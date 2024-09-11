#!/bin/bash
#SBATCH --job-name=gemma-2-27b-it
#SBATCH --cpus-per-task=8
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --mem=70GB
#SBATCH --gres=gpu:1
#SBATCH --constraint=a100
#SBATCH --output=log/gemma-2-27b-it.out
#SBATCH --error=error/gemma-2-27b-it.err

# activate virtual environment
source harness_env/bin/activate

path="google"
model="gemma-2-27b-it"
model_name=$path/$model
num_fewshot=3

srun python3 unified_code.py \
    --model hf \
    --model_args pretrained=$model_name,dtype=bfloat16,attn_implementation=flash_attention_2 \
    --batch_size 32 \
    --device cuda:0 \
    --num_fewshot ${num_fewshot} \
    --output_path results/${model}_description.txt \
    --data_path ../data/eval_data.jsonl \
    --apply_chat_template \
    --fewshot_as_multiturn