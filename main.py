import argparse
import os
import json
import time
import pickle
from tqdm import tqdm
from transformers.modeling_outputs import Seq2SeqLMOutput
import torch
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from pytorch_lightning import seed_everything
from pytorch_lightning.loggers import TensorBoardLogger

from transformers import AdamW, T5ForConditionalGeneration, T5TokenizerFast
from transformers import get_linear_schedule_with_warmup

from data_utils import ABSADataset
from data_utils import write_results_to_log, read_line_examples_from_file
from eval_utils import compute_scores
from GAT import *

os.environ["TOKENIZERS_PARALLELISM"] = "false"
def init_args():
    parser = argparse.ArgumentParser()
    # basic settings
    parser.add_argument("--task", default='aste', type=str, required=True,
                        help="The name of the task, selected from: [uabsa, aste, tasd,acos]")
    parser.add_argument("--dataset", default='rest14', type=str, required=True,
                        help="The name of the dataset, selected from: [laptop14, rest14, rest15, rest16]")
    parser.add_argument("--model_name_or_path", default='t5-base', type=str,
                        help="Path to pre-trained model or shortcut name")
    parser.add_argument("--paradigm", default='extraction')
    parser.add_argument("--do_train", action='store_true', help="Whether to run training.")
    parser.add_argument("--do_eval", action='store_true', help="Whether to run eval on the dev/test set.")
    parser.add_argument("--do_direct_eval", action='store_true', 
                        help="Whether to run direct eval on the dev/test set.")

    # Other parameters
    parser.add_argument("--max_seq_length", default=128, type=int)
    parser.add_argument("--n_gpu", default=0)
    parser.add_argument("--train_batch_size", default=16, type=int,
                        help="Batch size per GPU/CPU for training.")
    parser.add_argument("--eval_batch_size", default=16, type=int,
                        help="Batch size per GPU/CPU for evaluation.")
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1,
                        help="Number of updates steps to accumulate before performing a backward/update pass.")
    parser.add_argument("--learning_rate", default=3e-4, type=float)
    parser.add_argument("--num_train_epochs", default=20, type=int, 
                        help="Total number of training epochs to perform.")
    parser.add_argument('--seed', type=int, default=42, help="random seed for initialization")

    # training details
    parser.add_argument("--weight_decay", default=0.0, type=float)
    parser.add_argument("--adam_epsilon", default=1e-8, type=float)
    parser.add_argument("--warmup_steps", default=0.0, type=float)
    parser.add_argument("--hidden_size",default=256,type=int)
    parser.add_argument("--enable_prompt",default = True, type= bool,required=True)
    args = parser.parse_args()

    if not os.path.exists('./outputs'):
        os.mkdir('./outputs')

    task_dir = f"./outputs/{args.task}"
    if not os.path.exists(task_dir):
        os.mkdir(task_dir)

    task_dataset_dir = f"{task_dir}/{args.dataset}"
    if not os.path.exists(task_dataset_dir):
        os.mkdir(task_dataset_dir)

    output_dir = f"{task_dataset_dir}/{args.paradigm}"
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)
    args.output_dir = output_dir

    return args


def get_dataset(tokenizer, type_path, args):
    return ABSADataset(tokenizer=tokenizer, data_dir=args.dataset, data_type=type_path, 
                       paradigm=args.paradigm, task=args.task, max_len=args.max_seq_length)



class T5FineTuner(pl.LightningModule):
    def __init__(self, hparams):
        super(T5FineTuner, self).__init__()
        self.hparams = hparams

        self.model = T5ForConditionalGeneration.from_pretrained(hparams.model_name_or_path)
        self.regulator = GAT(nfeat=hparams.hidden_size,nhid=128,nclass=hparams.hidden_size,dropout=0.4,alpha=0.05,nheads=4)
        self.tokenizer = T5TokenizerFast.from_pretrained(hparams.model_name_or_path)

    def is_logger(self):
        return True

    def forward(self, input_ids, tag = None,attention_mask=None, offset_mapping = None,decoder_input_ids=None,
                decoder_attention_mask=None, labels=None,matrix = None):
        if prefix_word !='':
            prefix_tokens = self.tokenizer(prefix_word, return_tensors='pt')['input_ids'][:,:-1].to(self.device)
            len = int(prefix_tokens.shape[1])
            prefix_tokens = prefix_tokens.expand(input_ids.shape[0], -1).to(self.device)
            #prefix
            input_ids = torch.cat([prefix_tokens,input_ids],dim=1).to(self.device)
            attention_mask = torch.cat([torch.full((attention_mask.shape[0], prefix_tokens.shape[1]), 1, dtype=torch.long).to(self.device),
                                       attention_mask.to(self.device)],dim=1)
            encoder_outputs = self.model.encoder(input_ids=input_ids, attention_mask=attention_mask)
            encoder_outputs_raw = encoder_outputs['last_hidden_state'][:,len:,:].clone().to(self.device)

            index = self.regulator(x=encoder_outputs['last_hidden_state'][:, len:, :].clone().to(self.device),
                             offset_mapping = offset_mapping,
                             tag=tag.to(self.device),
                             attention_mask=attention_mask[:, len:].clone().to(self.device),
                             adj=matrix.to(self.device)).to(self.device)
            ec = encoder_outputs_raw.clone()
            encoder_outputs_raw = ec.scatter_(dim=1, index=index.unsqueeze(2).repeat([1, 1, 768]),
                                              src=encoder_outputs_raw)
            prefix_embeds = encoder_outputs['last_hidden_state'][:,:len,:].clone().to(self.device)
            encoder_outputs['last_hidden_state'] = torch.cat([prefix_embeds,encoder_outputs_raw],dim=1).to(self.device)
        else:
            encoder_outputs = self.model.encoder(input_ids=input_ids, attention_mask=attention_mask)
            index = self.regulator(x=encoder_outputs['last_hidden_state'].clone().to(self.device),
                             offset_mapping = offset_mapping,
                             tag=tag.to(self.device),
                             attention_mask=attention_mask.to(self.device),
                             adj=matrix.to(self.device)).to(self.device)
            ec = encoder_outputs['last_hidden_state'].clone()
            encoder_outputs_raw = ec.scatter_(dim=1, index=index.unsqueeze(2).repeat([1, 1, 768]),src=encoder_outputs['last_hidden_state'])
            encoder_outputs['last_hidden_state'] = encoder_outputs_raw

        decoder_input_ids = self.model._shift_right(labels)
        decoder_attention_mask = self.model._shift_right(decoder_attention_mask)

        for i in range(decoder_attention_mask.shape[0]):
            decoder_attention_mask[i][0] = 1
        decoder_outputs = self.model.decoder(
                    encoder_hidden_states=encoder_outputs[0],
                    encoder_attention_mask=attention_mask,
                    input_ids=decoder_input_ids,
                    attention_mask = decoder_attention_mask
                )
        sequence_output = decoder_outputs[0] * (self.model.model_dim ** -0.5)
        lm_logits = self.model.lm_head(sequence_output)
        loss_fct = torch.nn.CrossEntropyLoss(ignore_index=-100)
        loss = loss_fct(lm_logits.view(-1,lm_logits.size(-1)),labels.view(-1))
        return Seq2SeqLMOutput(logits=lm_logits,loss=loss,
                                              past_key_values=decoder_outputs.past_key_values,
                                              )
    def _step(self, batch):
        lm_labels = batch["target_ids"]
        lm_labels[lm_labels[:, :] == self.tokenizer.pad_token_id] = -100

        outputs = self(
            input_ids=batch["source_ids"],
            attention_mask=batch["source_mask"],
            offset_mapping = batch["offset_mapping"],
            tag = batch["tag"],
            labels=lm_labels,
            matrix=batch["matrix"],
            decoder_attention_mask=batch['target_mask']
        )

        loss = outputs[0]
        return loss

    def training_step(self, batch, batch_idx):
        loss = self._step(batch)
        tensorboard_logs = {"train_loss": loss}
        return {"loss": loss, "log": tensorboard_logs}

    def training_epoch_end(self, outputs):
        avg_train_loss = torch.stack([x["loss"] for x in outputs]).mean()
        tensorboard_logs = {"avg_train_loss": avg_train_loss}
        self.logger.experiment.add_scalar("Loss/Train",avg_train_loss,self.current_epoch)
        return {"avg_train_loss": avg_train_loss, "log": tensorboard_logs, 'progress_bar': tensorboard_logs}

    def validation_step(self, batch, batch_idx):
        loss = self._step(batch)
        return {"val_loss": loss}

    def validation_epoch_end(self, outputs):
        avg_loss = torch.stack([x["val_loss"] for x in outputs]).mean()
        tensorboard_logs = {"val_loss": avg_loss}
        self.logger.experiment.add_scalar("Loss/Val", avg_loss, self.current_epoch)
        return {"avg_val_loss": avg_loss, "log": tensorboard_logs, 'progress_bar': tensorboard_logs}
        #return {"avg_val_loss": avg_loss, 'progress_bar': tensorboard_logs}
    def configure_optimizers(self):
        '''Prepare optimizer and schedule (linear warmup and decay)'''
        model = self.model
        regulator = self.regulator
        no_decay = ["bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {
                "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
                "weight_decay": self.hparams.weight_decay,
            },
            {
                "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay) ],
                "weight_decay": 0.0,
            },
            {
                "params": [p for n, p in regulator.named_parameters()],
                "weight_decay": self.hparams.weight_decay,
                "lr": self.hparams.learning_rate,
            },

        ]
        optimizer = AdamW(optimizer_grouped_parameters, lr=self.hparams.learning_rate, eps=self.hparams.adam_epsilon)
        self.opt = optimizer
        return [optimizer]

    def optimizer_step(self, epoch, batch_idx, optimizer, optimizer_idx, second_order_closure=None):
        if self.trainer.use_tpu:
            xm.optimizer_step(optimizer)
        else:
            optimizer.step()
        optimizer.zero_grad()
        self.lr_scheduler.step()

    def get_tqdm_dict(self):
        tqdm_dict = {"loss": "{:.4f}".format(self.trainer.avg_loss), "lr": self.lr_scheduler.get_last_lr()[-1]}
        return tqdm_dict

    def train_dataloader(self):
        train_dataset = get_dataset(tokenizer=self.tokenizer, type_path="train", args=self.hparams)
        dataloader = DataLoader(train_dataset, batch_size=self.hparams.train_batch_size, drop_last=True, shuffle=True, num_workers=4)
        t_total = (
            (len(dataloader.dataset) // (self.hparams.train_batch_size * max(1, len(self.hparams.n_gpu))))
            // self.hparams.gradient_accumulation_steps
            * float(self.hparams.num_train_epochs)
        )
        scheduler = get_linear_schedule_with_warmup(
            self.opt, num_warmup_steps=self.hparams.warmup_steps, num_training_steps=t_total
        )
        self.lr_scheduler = scheduler
        return dataloader

    def val_dataloader(self):
        val_dataset = get_dataset(tokenizer=self.tokenizer, type_path="dev", args=self.hparams)
        return DataLoader(val_dataset, batch_size=self.hparams.eval_batch_size, num_workers=4)

def get_constrained_decoding_token_ids_tasd(tokenizer, dataset, task_config):
    #tokens = ['|','&']
    token_list = {"rest15": ["restaurant prices", "drinks prices", "ambience general", "food prices", "drinks style_options",
               "food general", "food quality", "restaurant miscellaneous", "location general", "restaurant general",
               "food style_options", "service general", "drinks quality","NULL"],
                "rest16": ["restaurant prices", "drinks prices", "ambience general", "food prices", "drinks style_options",
               "food quality", "restarant miscellaneous", "location general", "restaurant general",
               "food style_options", "service general", "drinks quality","NULL"]}

    tokens = ' '.join(token_list[dataset])
    ids = tokenizer.encode(tokens)[:-1]     # not including the eos token
    return list(set(ids))

def get_constrained_decoding_token_ids_acos(tokenizer, dataset, task_config):
    token_list = {
        "laptop14": ["MEMORY#OPERATION_PERFORMANCE", "DISPLAY#OPERATION_PERFORMANCE", "LAPTOP#GENERAL", "PORTS#GENERAL",
                     "MULTIMEDIA_DEVICES#USABILITY", "MOTHERBOARD#OPERATION_PERFORMANCE",
                     "Out_Of_Scope#OPERATION_PERFORMANCE", "SOFTWARE#GENERAL", "MULTIMEDIA_DEVICES#DESIGN_FEATURES",
                     "CPU#QUALITY", "SHIPPING#QUALITY", "MULTIMEDIA_DEVICES#CONNECTIVITY", "MOUSE#GENERAL",
                     "SUPPORT#QUALITY", "CPU#DESIGN_FEATURES", "SOFTWARE#PRICE", "LAPTOP#PORTABILITY",
                     "OS#MISCELLANEOUS", "BATTERY#QUALITY", "LAPTOP#CONNECTIVITY", "MEMORY#GENERAL",
                     "MEMORY#DESIGN_FEATURES", "SUPPORT#GENERAL", "FANS&COOLING#DESIGN_FEATURES",
                     "POWER_SUPPLY#CONNECTIVITY", "SHIPPING#OPERATION_PERFORMANCE", "KEYBOARD#QUALITY",
                     "LAPTOP#QUALITY", "HARDWARE#QUALITY", "WARRANTY#QUALITY", "OPTICAL_DRIVES#USABILITY",
                     "OPTICAL_DRIVES#GENERAL", "HARDWARE#USABILITY", "CPU#GENERAL", "SUPPORT#PRICE",
                     "FANS&COOLING#OPERATION_PERFORMANCE", "KEYBOARD#USABILITY", "MULTIMEDIA_DEVICES#PRICE",
                     "OS#GENERAL", "DISPLAY#PRICE", "GRAPHICS#DESIGN_FEATURES", "FANS&COOLING#QUALITY",
                     "SUPPORT#DESIGN_FEATURES", "PORTS#QUALITY", "MULTIMEDIA_DEVICES#GENERAL",
                     "HARDWARE#OPERATION_PERFORMANCE", "KEYBOARD#PORTABILITY", "PORTS#DESIGN_FEATURES",
                     "BATTERY#GENERAL", "SOFTWARE#USABILITY", "Out_Of_Scope#USABILITY", "OS#DESIGN_FEATURES",
                     "PORTS#OPERATION_PERFORMANCE", "SOFTWARE#OPERATION_PERFORMANCE", "FANS&COOLING#GENERAL",
                     "BATTERY#DESIGN_FEATURES", "HARD_DISC#PRICE", "MULTIMEDIA_DEVICES#QUALITY", "OS#USABILITY",
                     "COMPANY#DESIGN_FEATURES", "PORTS#CONNECTIVITY", "COMPANY#QUALITY", "GRAPHICS#GENERAL",
                     "MULTIMEDIA_DEVICES#OPERATION_PERFORMANCE", "CPU#OPERATION_PERFORMANCE", "MOTHERBOARD#QUALITY",
                     "LAPTOP#MISCELLANEOUS", "LAPTOP#DESIGN_FEATURES", "KEYBOARD#OPERATION_PERFORMANCE",
                     "PORTS#USABILITY", "DISPLAY#DESIGN_FEATURES", "HARD_DISC#QUALITY", "OS#QUALITY",
                     "POWER_SUPPLY#QUALITY", "COMPANY#OPERATION_PERFORMANCE", "HARD_DISC#GENERAL", "WARRANTY#GENERAL",
                     "KEYBOARD#GENERAL", "HARD_DISC#OPERATION_PERFORMANCE", "LAPTOP#PRICE", "POWER_SUPPLY#GENERAL",
                     "MEMORY#USABILITY", "KEYBOARD#PRICE", "LAPTOP#USABILITY", "SOFTWARE#PORTABILITY",
                     "SHIPPING#GENERAL", "CPU#PRICE", "KEYBOARD#DESIGN_FEATURES", "OPTICAL_DRIVES#DESIGN_FEATURES",
                     "SUPPORT#OPERATION_PERFORMANCE", "OS#OPERATION_PERFORMANCE", "LAPTOP#OPERATION_PERFORMANCE",
                     "GRAPHICS#USABILITY", "SHIPPING#PRICE", "SOFTWARE#QUALITY", "COMPANY#GENERAL",
                     "POWER_SUPPLY#DESIGN_FEATURES", "HARD_DISC#DESIGN_FEATURES", "SOFTWARE#DESIGN_FEATURES",
                     "BATTERY"
                     "#OPERATION_PERFORMANCE", "OPTICAL_DRIVES#OPERATION_PERFORMANCE", "DISPLAY#GENERAL",
                     "DISPLAY#QUALITY", "DISPLAY#USABILITY", "HARDWARE#PRICE", "COMPANY#PRICE",
                     "GRAPHICS#OPERATION_PERFORMANCE", "PORTS#PORTABILITY", "HARD_DISC#USABILITY",
                     "Out_Of_Scope#GENERAL", "MEMORY#QUALITY", "POWER_SUPPLY#OPERATION_PERFORMANCE",
                     "HARDWARE#DESIGN_FEATURES", "HARDWARE#GENERAL"],
        "rest16": ["DRINKS#STYLE_OPTIONS", "FOOD#STYLE_OPTIONS", "RESTAURANT#MISCELLANEOUS", "LOCATION#GENERAL",
                   "RESTAURANT#GENERAL", "DRINKS#QUALITY", "RESTAURANT#PRICES", "DRINKS#PRICES", "FOOD#QUALITY",
                   "AMBIENCE#GENERAL", "FOOD#PRICES", "SERVICE#GENERAL"]
    }

    tokens = ' '.join(token_list[dataset]).lower().replace('#',' ').replace('_',' ')
    ids = tokenizer.encode(tokens)[:-1]  # not including the eos token
    return list(set(ids))

def evaluate(data_loader, model, paradigm, task, sents):
    """
    Compute scores given the predictions and gold labels
    """
    device = torch.device(f'cuda:{args.n_gpu}')
    model.to(device)
    model.eval()
    outputs, targets = [], []

    #for category term, multiple choices will be added
    constrained_token_id = ''
    if args.task =='tasd':
        constrained_token_id = get_constrained_decoding_token_ids_tasd(model.tokenizer, args.dataset, task)
    elif args.task =='acos':
        constrained_token_id = get_constrained_decoding_token_ids_acos(model.tokenizer,args.dataset,task)

    for batch in tqdm(data_loader):
        # need to push the data to device
        if prefix_word != '':
            prefix_tokens = model.tokenizer(prefix_word,return_tensors='pt')['input_ids'][:,:-1].to(device)
            prefix_tokens = prefix_tokens.expand(batch['source_ids'].shape[0], -1).to(device)
            len = int(prefix_tokens.shape[1])  #length of prompt
            input_ids = torch.cat([prefix_tokens.to(device), batch['source_ids'].to(device)], dim=1).to(device)
            attention_mask = torch.cat(
                [torch.full((batch['source_mask'].shape[0], prefix_tokens.shape[1]), 1, dtype=torch.long).to(device),
                 batch['source_mask'].to(device)], dim=1)
            #representations after encoder
            encoder_outputs = model.model.encoder(input_ids=input_ids, attention_mask=attention_mask)
            #text representations are sent to sequence regualtor
            index = model.regulator(x=encoder_outputs['last_hidden_state'][:, len:, :].clone().to(device),
                              offset_mapping = batch["offset_mapping"].to(device),
                              tag=batch['tag'].to(device),
                              attention_mask = attention_mask[:, len:].clone().to(device),
                              adj=batch['matrix'].to(device)).to(device)
            ec = encoder_outputs['last_hidden_state'][:, len:, :].clone()
            encoder_outputs_raw = ec.scatter_(dim=1, index=index.unsqueeze(2).repeat([1, 1, 768]),
                                              src=encoder_outputs['last_hidden_state'][:, len:, :])
            #prompt representations
            prefix_embeds = encoder_outputs['last_hidden_state'][:, :len, :].to(device)
            encoder_outputs['last_hidden_state'] = torch.cat(
                [prefix_embeds, encoder_outputs_raw], dim=1).to(device)
        else:
            encoder_outputs = model.model.encoder(input_ids=batch['source_ids'].to(device),
                                                  attention_mask=batch['source_mask'].to(device))
            index = model.regulator(x=encoder_outputs['last_hidden_state'].clone().to(device),
                              offset_mapping=batch["offset_mapping"].to(device),
                              tag=batch['tag'].to(device),
                              attention_mask=batch['source_mask'].clone().to(device),
                              adj=batch['matrix'].to(device)).to(device)
            ec = encoder_outputs['last_hidden_state'].clone()
            encoder_outputs_raw = ec.scatter_(dim=1, index=index.unsqueeze(2).repeat([1, 1, 768]),
                                              src=encoder_outputs['last_hidden_state'])
            attention_mask = batch['source_mask'].clone().to(device)
            encoder_outputs['last_hidden_state'] = encoder_outputs_raw.to(device)

        decoder_input_ids = torch.zeros([batch['source_ids'].shape[0], 1]).long().to(device)  #0 is the start

        def prefix_allowed_tokens_fn(batch_id, input_ids):
            allowed_tokens = batch['source_ids'][batch_id].tolist()
            if args.task == 'acos':
                special_tokens = model.tokenizer.encode('<extra_id_0> <extra_id_1> <extra_id_2> <extra_id_3> <extra_id_4> positive negative neutral NULL')
            elif args.task == 'uabsa':
                special_tokens = model.tokenizer.encode('<extra_id_0> <extra_id_1> <extra_id_2> positive negative neutral None')
            elif args.task in ['aste','tasd']:
                special_tokens = model.tokenizer.encode('<extra_id_3> <extra_id_0> <extra_id_1> <extra_id_2> positive negative neutral')
            allowed_tokens += special_tokens
            allowed_tokens += constrained_token_id
            allowed_tokens = list(set(allowed_tokens))
            return allowed_tokens
        outs = model.model.generate(encoder_outputs=encoder_outputs,
                                    attention_mask=attention_mask,
                                    decoder_input_ids=decoder_input_ids,
                                    max_length=128,
                                    prefix_allowed_tokens_fn = prefix_allowed_tokens_fn
                                    )

        dec = [model.tokenizer.decode(ids) for ids in outs]
        target = [model.tokenizer.decode(ids) for ids in batch["target_ids"]]
        #replace pad and eos with ''
        dec1 = [d.replace('<pad> ','').replace('<pad>','').replace('</s> ','').replace('</s>','') for d in dec]
        target1 = [t.replace('<pad> ','').replace('<pad>','').replace('</s> ','').replace('</s>','') for t in target]

        outputs.extend(dec1)
        targets.extend(target1)


    scores, all_labels, all_preds = compute_scores(outputs, targets, sents, paradigm, task)
    results = {'sents':sents,'scores': scores, 'labels': all_labels,
               'preds': all_preds}
    #pickle.dump(results, open(f"{args.output_dir}/results-{args.task}-{args.dataset}-{args.paradigm}_GCN.pickle", 'wb'))



    return results



logger = TensorBoardLogger('/root/tf-logs',name = 'my_model_run_name')

# initialization
args = init_args()
seed_everything(args.seed)
print("\n", "="*30, f"NEW EXP: {args.task.upper()} on {args.dataset}", "="*30, "\n")
if args.task == 'tasd':
    if args.enable_prompt== True:
        semantic_prompt = "aspect mean <extra_id_0> , category mean <extra_id_1> , sentiment mean <extra_id_2> . "
        fewshot_prompt = "Input : sushi is good. Target : <extra_id_0> sushi <extra_id_1> food general <extra_id_2> positive . "
        prefix_word = semantic_prompt + fewshot_prompt
    else:
        prefix_word = ''
elif args.task == 'aste':
    if args.enable_prompt== True:
        semantic_prompt = "aspect mean <extra_id_0> , opinion mean <extra_id_1> , sentiment mean <extra_id_2> . "
        if args.dataset == 'laptop14':
            fewshot_prompt = "Input : laptop runs fast . Target : <extra_id_0> laptop <extra_id_1> fast <extra_id_2> positive . "
        else:
            fewshot_prompt = "Input : sushi is good . Target : <extra_id_0> sushi <extra_id_1> good <extra_id_2> positive . "
        prefix_word = semantic_prompt + fewshot_prompt
    else:
        prefix_word = ''
elif args.task == 'uabsa':
    if args.enable_prompt== True:
        semantic_prompt = "aspect mean <extra_id_0> , sentiment mean <extra_id_1> . "
        if args.dataset == 'laptop14':
            fewshot_prompt = "Input : laptop runs fast . Target : <extra_id_0> laptop <extra_id_1> positive . "
        else:
            fewshot_prompt = "Input : sushi is good . Target : <extra_id_0> sushi <extra_id_1> positive . "
        prefix_word = semantic_prompt + fewshot_prompt
    else:
        prefix_word = ''
elif args.task == 'acos':
    if args.enable_prompt== True:
        semantic_prompt = "aspect mean <extra_id_0> , category mean <extra_id_1> , sentiment mean <extra_id_2> , opinion mean <extra_id_3> . "
        if args.dataset == 'laptop14':
            fewshot_prompt = "Input : laptop runs fast. Target : <extra_id_0> laptop <extra_id_1> laptop general <extra_id_2> positive <extra_id_3> fast . "
        else:
            fewshot_prompt = "Input : sushi is good. Target : <extra_id_0> sushi <extra_id_1> food general <extra_id_2> positive <extra_id_3> good . "
        prefix_word = semantic_prompt + fewshot_prompt
    else:
        prefix_word = ''

#training process
if args.do_train:
    print("\n****** Conduct Training ******")
    model = T5FineTuner(args)

    checkpoint_callback = pl.callbacks.ModelCheckpoint(
        filepath=args.output_dir, prefix="ckt", monitor='val_loss', mode='min',save_top_k=-1,save_weights_only=True,
    )

    # prepare for trainer
    train_params = dict(
        default_root_dir=args.output_dir,
        accumulate_grad_batches=args.gradient_accumulation_steps,
        gpus=args.n_gpu,
        gradient_clip_val=1.0,
        max_epochs=args.num_train_epochs,
        checkpoint_callback=checkpoint_callback,
        logger = logger
        #callbacks=[LoggingCallback()],
    )

    trainer = pl.Trainer(**train_params)
    start = time.time()
    trainer.fit(model)

    model.model.save_pretrained(args.output_dir)

    print("Finish training and saving the model!")
    during = time.time() - start
    print(during)

if args.do_eval:

    print("\n****** Conduct Evaluating ******")
    start = time.time()
    # model = T5FineTuner(args)
    dev_results, test_results = {}, {}
    best_f1, best_checkpoint, best_epoch = -999999.0, None, None
    all_checkpoints, all_epochs = [], []

    # retrieve all the saved checkpoints for model selection
    saved_model_dir = args.output_dir
    for f in os.listdir(saved_model_dir):
        file_name = os.path.join(saved_model_dir, f)
        if 'cktepoch' in file_name:
            all_checkpoints.append(file_name)

    # conduct some selection (or not)
    print(f"Load and evaluate checkpoint: {all_checkpoints}")

    model = T5FineTuner(args)
    test_dataset = ABSADataset(model.tokenizer, data_dir=args.dataset, data_type='test',
                    paradigm=args.paradigm, task=args.task, max_len=args.max_seq_length)
    test_loader = DataLoader(test_dataset, batch_size=args.eval_batch_size , num_workers=4)
    sents_test,_ ,_,_= read_line_examples_from_file(f'data/{args.task}/{args.dataset}/test.txt')
    all_scores = []
    for checkpoint in all_checkpoints:
        epoch = checkpoint.split('=')[-1][:-5] if len(checkpoint) > 1 else ""
        if 5<int(epoch)<41:
            all_epochs.append(epoch)

            # reload the model and conduct inference
            print(f"\nLoad the trained model frin om {checkpoint}...")
            model_ckpt = torch.load(checkpoint)

            model.load_state_dict(model_ckpt['state_dict'])
            test_result = evaluate(test_loader, model, args.paradigm, args.task,sents_test)
            logger.experiment.add_scalar("f1",test_result['scores']['f1'],epoch)
            all_scores.append((epoch,test_result['scores']))
    for epoch, scores in all_scores:
        if scores['f1'] > best_f1:
            best_epoch, best_f1 = epoch, scores['f1']
    print(f"best epoch={best_epoch}, f1={best_f1}\n")
    with open(os.path.join(args.output_dir, f'all_scores_{args.task}.txt'), "w") as writer:
        for epoch, scores in all_scores:
            if scores['f1'] > best_f1:
                best_epoch, best_f1 = epoch, scores['f1']
            writer.write(f"epoch={epoch}\n")
            writer.write(json.dumps(scores, indent=4) + "\n")
        writer.write(f"best epoch={best_epoch}, f1={best_f1}\n")
    during = time.time() - start
    print(during)


