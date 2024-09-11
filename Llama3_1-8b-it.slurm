#!/bin/bash
#SBATCH --job-name=Meta-Llama-3.1-8B-it
#SBATCH --cpus-per-task=8
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --mem=70GB
#SBATCH --gres=gpu:1
#SBATCH --constraint=a100
#SBATCH --output=log/Meta-Llama-3.1-8B-it.out
#SBATCH --error=error/Meta-Llama-3.1-8B-it.err

# activate virtual environment
source harness_env/bin/activate

path="meta-llama"
model="Meta-Llama-3.1-8B-Instruct"
model_name=$path/$model
num_fewshot=3

srun python3 unified_code.py \
    --model hf \
    --model_args pretrained=$model_name,dtype=bfloat16,attn_implementation=flash_attention_2 \
    --batch_size auto:64 \
    --device cuda:0 \
    --num_fewshot ${num_fewshot} \
    --output_path results/${model}_description.txt \
    --apply_chat_template \
    --fewshot_as_multiturn