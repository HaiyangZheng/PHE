#!/bin/bash
#SBATCH -A IscrC_HDSCisLa
#SBATCH -p boost_usr_prod
#SBATCH --qos=normal
#SBATCH -N 1                # 1 node
#SBATCH --ntasks-per-node=1 # 4 tasks out of 32
#SBATCH --time 1-00:00:00     # format: HH:MM:SS
#SBATCH --gres=gpu:1        # 4 gpus per node out of 4
#SBATCH --mem=100000          # memory per node out of 494000MB (481GB)
#SBATCH --job-name=ocdcub
#SBATCH -o /leonardo_work/IscrC_Fed-GCD/hyzheng/temp3/log/cub0.log

module load cuda/12.1
source /leonardo/home/userexternal/hzheng00/miniconda3/bin/activate fedgcd

CUDA_VISIBLE_DEVICES=0

data_set=cub
seed=1027
output_dir=exp/

python main.py \
    --data_set=$data_set \
    --output_dir=$output_dir/$data_set/"cub_iscap_seed($seed)" \
    --seed=$seed \