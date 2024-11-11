#!/usr/bin/env bash

python main.py --task uabsa \
            --dataset laptop14 \
            --model_name_or_path ./pretrained-models/t5-base \
            --n_gpu 0 \
            --do_train \
            --do_eval \
            --train_batch_size 48 \
            --gradient_accumulation_steps 2 \
            --eval_batch_size 40 \
            --learning_rate 3e-4 \
            --num_train_epochs 20 \
            --enable_prompt True