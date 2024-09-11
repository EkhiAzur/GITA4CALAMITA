#!/bin/bash
#SBATCH --job-name=gemma-2-9b
#SBATCH --cpus-per-task=8
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --mem=70GB
#SBATCH --gres=gpu:1
#SBATCH --constraint=a100
#SBATCH --output=log/gemma-2-9b.out
#SBATCH --error=error/gemma-2-9b.err

# activate virtual environment
source ../harness_env/bin/activate

path="google"
model="gemma-2-9b"
model_name=$path/$model
num_fewshot=3
tasks_path="../tasks"

srun python3 unified_code.py \
    --model hf \
    --model_args pretrained=$model_name,dtype=bfloat16,attn_implementation=flash_attention_2 \
    --batch_size 32 \
    --device cuda:0 \
    --num_fewshot ${num_fewshot} \
    --output_path ../results/${model} \
    --include_path ${tasks_path} \