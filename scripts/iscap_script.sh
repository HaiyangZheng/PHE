#!/bin/bash
#SBATCH -A IscrC_HDSCisLa
#SBATCH -p boost_usr_prod
#SBATCH --qos=normal
#SBATCH -N 1                # 1 node
#SBATCH --ntasks-per-node=1 # 4 tasks out of 32
#SBATCH --time 1-00:00:00     # format: HH:MM:SS
#SBATCH --gres=gpu:1        # 4 gpus per node out of 4
#SBATCH --mem=100000          # memory per node out of 494000MB (481GB)
#SBATCH --job-name=ocdscar
#SBATCH -o /leonardo_work/IscrC_Fed-GCD/hyzheng/PHE_release/log/test_scars.log

module load cuda/12.1
source /leonardo/home/userexternal/hzheng00/miniconda3/bin/activate fedgcd

CUDA_VISIBLE_DEVICES=0

# Training Config
data_set=CD_Car
use_global=True
global_proto_per_class=10
prototype_dim=768
batch_size=128
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


python main.py \
    --data_set=$data_set \
    --output_dir=$output_dir/$data_set/"test" \
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
    --features_lr=$features_lr \
    --add_on_layers_lr=$add_on_layers_lr \
    --prototype_vectors_lr=$prototype_vectors_lr \
    --use_global=$use_global \
    --global_proto_per_class=$global_proto_per_class \
    --mask_theta=0.1 \
    --hash_code_length=12 \
    --prototype_dim=$prototype_dim \