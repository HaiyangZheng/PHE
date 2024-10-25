#!/bin/bash
#SBATCH -A IscrC_Fed-GCD
#SBATCH -p boost_usr_prod
#SBATCH --qos=normal
#SBATCH --time 1-00:00:00     # format: HH:MM:SS
#SBATCH -N 1                # 1 node
#SBATCH --ntasks-per-node=4 # 4 tasks out of 32
#SBATCH --gres=gpu:1        # 4 gpus per node out of 4
#SBATCH --mem=32000          # memory per node out of 494000MB (481GB)
#SBATCH --job-name=dccl_pseudo_test
#SBATCH -o /leonardo_work/IscrC_Fed-GCD/hyzheng/myocd/temp/before_rebuttal_car.txt

module load cuda/12.1
source /leonardo/home/userexternal/hzheng00/miniconda3/bin/activate protop

export CUDA_VISIBLE_DEVICES=0

# model=deit_tiny_patch16_224
# model=deit_small_patch16_224
model=deit_base_patch16_224
# model=cait_xxs24_224
batch_size=128
num_gpus=1
# model=$1
# batch_size=$2
# num_gpus=$3

use_port=2672
seed=1028
# seed=3796
# seed=0

# Learning Rate
warmup_lr=1e-4
warmup_epochs=5
features_lr=1e-4
add_on_layers_lr=1e-3
prototype_vectors_lr=1e-3
# warmup_lr=5e-5
# warmup_epochs=5
# features_lr=5e-5
# add_on_layers_lr=1e-3
# prototype_vectors_lr=1e-3

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
use_ppc_loss=False   # Whether use PPC loss
last_reserve_num=196 # Number of reserve tokens in the last layer
global_coe=0.5      # Weight of the global branch, 1 - global_coe is for local branch
ppc_cov_thresh=1.   # The covariance thresh of PPC loss
ppc_mean_thresh=2.  # The mean thresh of PPC loss
global_proto_per_class=10 # Number of global prototypes per class
ppc_cov_coe=0.1     # The weight of the PPC_{sigma} Loss
ppc_mean_coe=0.5    # The weight of the PPC_{mu} Loss
dim=768             # The dimension of each prototype

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

for data_set in CD_Car;
do
    prototype_num=980
    data_path=datasets
    python main.py \
        --base_architecture=$model \
        --data_set=$data_set \
        --data_path=$data_path \
        --input_size=$input_size \
        --output_dir=$output_dir/$data_set/"Current_baseline(10proto+dropout0.1)_iscap_before_rebuttal" \
        --batch_size=$batch_size \
        --seed=$seed \
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
        --ppc_mean_coe=$ppc_mean_coe
done