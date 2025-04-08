#!/bin/bash
#SBATCH --job-name=LLaMAntino-3-ANITA-8B-Inst-DPO-ITA
#SBATCH --cpus-per-task=8
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --mem=70GB
#SBATCH --gres=gpu:1
#SBATCH --constraint=a100
#SBATCH --output=log/LLaMAntino-3-ANITA-8B-Inst-DPO-ITA.out
#SBATCH --error=error/LLaMAntino-3-ANITA-8B-Inst-DPO-ITA.err

source ../../harness_env/bin/activate

BATCH_SIZE=32
NUM_GPUS=1

MODEL="swap-uniba/LLaMAntino-3-ANITA-8B-Inst-DPO-ITA"
OUTPUT_DIR="LLaMAntino-3-ANITA-8B-Inst-DPO-ITA"
GITA4CALAMITA_PREFIX="LLaMAntino-3-ANITA-8B-Inst-DPO-ITA" # Prefix for the temporary files to allow parallel evaluations
echo "Evaluating $MODEL with $NUM_GPUS GPUs and batch size $BATCH_SIZE"
sh eval-instruct.sh $MODEL $BATCH_SIZE $OUTPUT_DIR $NUM_GPUS $GITA4CALAMITA_PREFIX