CUDA_VISIBLE_DEVICES=0

data_set=cub
seed=1028
output_dir=exp/

python main.py \
    --data_set=$data_set \
    --output_dir=$output_dir/$data_set/"cub_iscap_seed($seed)" \
    --seed=$seed \