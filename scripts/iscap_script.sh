#!/bin/bash
#SBATCH -A IscrC_HDSCisLa
#SBATCH -p boost_usr_prod
#SBATCH --qos=normal
#SBATCH -N 1                # 1 node
#SBATCH --ntasks-per-node=1 # 4 tasks out of 32
#SBATCH --time 1-00:00:00     # format: HH:MM:SS
#SBATCH --gres=gpu:1        # 4 gpus per node out of 4
#SBATCH --mem=100000          # memory per node out of 494000MB (481GB)
#SBATCH --job-name=g_c_82
#SBATCH -o /leonardo_work/IscrC_Fed-GCD/hyzheng/PHE_release/log/cub_test.log

module load cuda/12.1
source /leonardo/home/userexternal/hzheng00/miniconda3/bin/activate fedgcd

CUDA_VISIBLE_DEVICES=0

model=deit_base_patch16_224
batch_size=128
num_gpus=1
use_port=2672
seed=1027

# Learning Rate
warmup_lr=1e-4
warmup_epochs=5
features_lr=1e-4
add_on_layers_lr=1e-3
prototype_vectors_lr=1e-3

# Optimizer & Scheduler
opt=adamw
sched=cosine
decay_epochs=10
decay_rate=0.1
weight_decay=0.05
epochs=200
output_dir=output_cosine/
input_size=224

use_global=True
use_ppc_loss=False
last_reserve_num=196
global_coe=0.5
ppc_cov_thresh=1.
ppc_mean_thresh=2.
global_proto_per_class=10
ppc_cov_coe=0.1
ppc_mean_coe=0.5
dim=768

if [ "$model" = "deit_tiny_patch16_224" ]
then
    reserve_layer_idx=11
elif [ "$model" = "deit_small_patch16_224" ]
then
    reserve_layer_idx=11
elif [ "$model" = "deit_base_patch16_224" ]
then
    reserve_layer_idx=11
elif [ "$model" = "cait_xxs24_224" ]
then
    reserve_layer_idx=1
fi

ft=protopformer
data_set=CD_CUB2011U
prototype_num=1000
labeled_nums=100
data_path=/db/pszzz/xxx


python main_uon_pszzz.py \
    --base_architecture=$model \
    --data_set=$data_set \
    --data_path=$data_path \
    --input_size=$input_size \
    --output_dir=$output_dir/$data_set/"cub_test_seed(1027)" \
    --batch_size=$batch_size \
    --seed=1027 \
    --opt=$opt \
    --sched=$sched \
    --warmup-epochs=$warmup_epochs \
    --warmup-lr=$warmup_lr \
    --decay-epochs=$decay_epochs \
    --decay-rate=$decay_rate \
    --weight_decay=$weight_decay \
    --epochs=$epochs \
    --finetune=$ft \
    --features_lr=$features_lr \
    --add_on_layers_lr=$add_on_layers_lr \
    --prototype_vectors_lr=$prototype_vectors_lr \
    --prototype_shape $prototype_num $dim 1 1 \
    --reserve_layers $reserve_layer_idx \
    --reserve_token_nums $last_reserve_num \
    --use_global=$use_global \
    --use_ppc_loss=$use_ppc_loss \
    --ppc_cov_thresh=$ppc_cov_thresh \
    --ppc_mean_thresh=$ppc_mean_thresh \
    --global_coe=$global_coe \
    --global_proto_per_class=$global_proto_per_class \
    --ppc_cov_coe=$ppc_cov_coe \
    --ppc_mean_coe=$ppc_mean_coe \
    --mask_theta=0.1 \
    --labeled_nums=$labeled_nums \