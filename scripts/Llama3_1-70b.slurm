#!/bin/bash
#SBATCH --job-name=Meta-Llama-3.1-70B
#SBATCH --cpus-per-task=8
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --mem=70GB
#SBATCH --gres=gpu:6
#SBATCH --constraint=a100
#SBATCH --output=log/Meta-Llama-3.1-70B.out
#SBATCH --error=error/Meta-Llama-3.1-70B.err

# activate virtual environment
source ../harness_env/bin/activate

path="meta-llama"
model="Meta-Llama-3.1-70B"
model_name=$path/$model
num_fewshot=3
tasks_path="../tasks"

srun python3 unified_code.py \
    --model hf \
    --model_args pretrained=$model_name,dtype=bfloat16,attn_implementation=flash_attention_2,parallelize=True \
    --batch_size 32 \
    --device cuda \
    --num_fewshot ${num_fewshot} \
    --output_path ../results/${model} \
    --include_path ${tasks_path} \