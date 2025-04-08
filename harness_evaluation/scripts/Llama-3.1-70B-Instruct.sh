#!/bin/bash
#SBATCH --job-name=Llama-3.1-70B-Instruct
#SBATCH --cpus-per-task=8
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --mem=70GB
#SBATCH --gres=gpu:4
#SBATCH --constraint=a100
#SBATCH --output=log/Llama-3.1-70B-Instruct.out
#SBATCH --error=error/Llama-3.1-70B-Instruct.err

source ../../harness_env/bin/activate

BATCH_SIZE=4
NUM_GPUS=4
export VLLM_WORKER_MULTIPROC_METHOD=spawn # Flag to avoid errors with multiprocessing

MODEL="meta-llama/Llama-3.1-70B-Instruct"
OUTPUT_DIR="Llama-3.1-70B-Instruct"
echo "Evaluating $MODEL with $NUM_GPUS GPUs and batch size $BATCH_SIZE"
sh eval-instruct.sh $MODEL $BATCH_SIZE $OUTPUT_DIR $NUM_GPUS $GITA4CALAMITA_PREFIX