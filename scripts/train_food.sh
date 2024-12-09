#!/bin/bash
CUDA_VISIBLE_DEVICES=0

data_set=food
seed=1029
output_dir=exp/

python main.py \
    --data_set=$data_set \
    --output_dir=$output_dir/$data_set/"food_seed($seed)" \
    --seed=$seed \