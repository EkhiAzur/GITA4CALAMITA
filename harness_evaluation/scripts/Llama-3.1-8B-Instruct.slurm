#!/bin/bash
#SBATCH --job-name=Meta-Llama-3.1-8B
#SBATCH --cpus-per-task=8
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --mem=70GB
#SBATCH --gres=gpu:1
#SBATCH --constraint=a100
#SBATCH --output=log/Meta-Llama-3.1-8B.out
#SBATCH --error=error/Meta-Llama-3.1-8B.err

source ../../harness_env/bin/activate

BATCH_SIZE=32
NUM_GPUS=1

MODEL="meta-llama/Llama-3.1-8B-Instruct"
OUTPUT_DIR="results/Llama-3.1-8B-Instruct"
GITA4CALAMITA_PREFIX="Llama-3.1-8B-Instruct" # Prefix for the temporary files to allow parallel evaluations
echo "Evaluating $MODEL with $NUM_GPUS GPUs and batch size $BATCH_SIZE"
sh eval-instruct.sh $MODEL $BATCH_SIZE $OUTPUT_DIR $NUM_GPUS $GITA4CALAMITA_PREFIX