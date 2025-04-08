#!/bin/bash
#SBATCH --job-name=Minerva-7B-instruct-v1.0
#SBATCH --cpus-per-task=8
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --mem=70GB
#SBATCH --gres=gpu:1
#SBATCH --constraint=a100
#SBATCH --output=log/Minerva-7B-instruct-v1.0.out
#SBATCH --error=error/Minerva-7B-instruct-v1.0.err

source ../../harness_env/bin/activate

BATCH_SIZE=32
NUM_GPUS=1

MODEL="sapienzanlp/Minerva-7B-instruct-v1.0"
OUTPUT_DIR="Minerva-7B-instruct-v1.0"
GITA4CALAMITA_PREFIX="Minerva-7B-instruct-v1.0" # Prefix for the temporary files to allow parallel evaluations
echo "Evaluating $MODEL with $NUM_GPUS GPUs and batch size $BATCH_SIZE"
sh eval-instruct.sh $MODEL $BATCH_SIZE $OUTPUT_DIR $NUM_GPUS $GITA4CALAMITA_PREFIX