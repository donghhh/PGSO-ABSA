# This file contains the evaluation functions

import re
import editdistance

sentiment_word_list = ['positive', 'negative', 'neutral']
senttag2wordl = {'positive': ['positive','great','wonderful','prefect','excellent','outstanding','fine'],
                 'negative': ['negative','bad','hurtful','disgusting','revolting','unpleasant','repulsive'],
                 'neutral': ['neutral','normal','ok','ordinary','usual','regular','typical']}
aspect_cate_list = ['location general',
 'food prices',
 'food quality',
 'ambience general',
 'service general',
 'restaurant prices',
 'drinks prices',
 'restaurant miscellaneous',
 'drinks quality',
 'drinks style_options',
 'restaurant general',
 'food style_options']
def return_key(val):
    for key,value in senttag2wordl.items():
        if val in value:
            return key
    return None

def extract_spans_extraction(task, seq):
    extractions = []
    if task == 'uabsa' and seq.lower() == 'none':
        return []
    else:
        if task in ['uabsa', 'aope']:
            # all_pt = seq.split('; ')
            all_pt = seq.split('<extra_id_2>')
            for pt in all_pt:
                try:
                    remain = pt.split('<extra_id_0> ')[-1]
                    a, b = remain.split('<extra_id_1> ')

                except ValueError:
                    a, b = '', ''
                extractions.append((a, b))
        elif task in ['tasd', 'aste']:
            all_pt = seq.split('<extra_id_3>')
            for pt in all_pt:
                try:
                    all = pt.split('<extra_id_0> ')[-1]
                    a,remain = all.split('<extra_id_1> ')
                    b,c = remain.split('<extra_id_2> ')

                except ValueError:
                    a, b, c = '', '', ''

                extractions.append((a, b, c))
        elif task =='acos':
            all_pt = seq.split('<extra_id_4>')
            for pt in all_pt:
                try:
                    all = pt.split('<extra_id_0> ')[-1]
                    a, remain = all.split('<extra_id_1> ')
                    b, last = remain.split('<extra_id_2> ')
                    c,d = last.split('<extra_id_3> ')
                except ValueError:
                    a, b, c,d = '', '', '',''
                extractions.append((a, b, c, d))

        return extractions





def compute_f1_scores(pred_pt, gold_pt):
    """
    Function to compute F1 scores with pred and gold pairs/triplets
    The input needs to be already processed
    """
    # number of true postive, gold standard, predicted aspect terms
    n_tp, n_gold, n_pred = 0, 0, 0

    for i in range(len(pred_pt)):
        n_gold += len(gold_pt[i])
        n_pred += len(pred_pt[i])
        pred_new = []
        for t in pred_pt[i]:
            if t in gold_pt[i] and t not in pred_new:  # delete same prediction
                n_tp += 1
                pred_new.append(t)
    precision = float(n_tp) / float(n_pred) if n_pred != 0 else 0
    recall = float(n_tp) / float(n_gold) if n_gold != 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if precision != 0 or recall != 0 else 0
    scores = {'precision': precision, 'recall': recall, 'f1': f1}

    return scores


def compute_scores(pred_seqs, gold_seqs, sents, io_format, task):
    """
    compute metrics for multiple tasks
    """
    assert len(pred_seqs) == len(gold_seqs) 
    num_samples = len(gold_seqs)

    all_labels, all_predictions = [], []

    for i in range(num_samples):
        gold_list = extract_spans_extraction(task, gold_seqs[i])
        pred_list = extract_spans_extraction(task, pred_seqs[i])

        all_labels.append(gold_list)
        all_predictions.append(pred_list)
    
    print("\nResults of raw output")
    scores = compute_f1_scores(all_predictions, all_labels)
    print(scores)

    
    return scores, all_labels, all_predictions