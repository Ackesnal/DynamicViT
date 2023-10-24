#!/bin/bash
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --job-name=deit
#SBATCH --partition=gpu
#SBATCH --cpus-per-task=40
#SBATCH --gres=gpu:tesla-smx2:2
#SBATCH --mem-per-cpu=2G
#SBATCH -o ./log_mop2/kdnoprojattention_r1b5T1_0.5.txt
#SBATCH -e error.txt

python -m torch.distributed.launch --nproc_per_node=2 --use_env main_dynamic_vit.py --output_dir logs/dynamicvit_deit-s_0.5_kdnoprojattention_b5 --distill --mydistill kdnoprojattention --distillw 5 --arch deit_small --input-size 224 --batch-size 256 --data-path /scratch/itee/uqxxu16/data/imagenet/ --epochs 30 --base_rate 0.5 --lr 1e-3 --warmup-epochs 5