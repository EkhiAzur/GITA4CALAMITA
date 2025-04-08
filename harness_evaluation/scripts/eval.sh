#!/bin/bash
MODEL=$1
BATCH_SIZE=$2
OUTPUT_DIR=$3
NUM_GPUS=$4
export GITA4CALAMITA_PREFIX=$5 # Prefix for the temporary files to allow parallel evaluations

lm_eval --model vllm \
    --model_args pretrained=${MODEL},dtype=bfloat16,tensor_parallel_size=$NUM_GPUS,gpu_memory_utilization=0.7,max_model_len=2048 \
    --tasks gita_story_class \
    --output_path ${OUTPUT_DIR}_story \
    --batch_size $BATCH_SIZE \
    --include_path ../tasks \
    --log_samples \
    --load_local \
    --local_base_dir ../datasets \

lm_eval --model vllm \
    --model_args pretrained=${MODEL},dtype=bfloat16,tensor_parallel_size=$NUM_GPUS,gpu_memory_utilization=0.7,max_model_len=2048 \
    --tasks gita_conflict_detect \
    --output_path ${OUTPUT_DIR}_conflict \
    --batch_size $BATCH_SIZE \
    --include_path ../tasks \
    --log_samples \
    --load_local \
    --local_base_dir ../datasets \

lm_eval --model vllm \
    --model_args pretrained=${MODEL},dtype=bfloat16,tensor_parallel_size=$NUM_GPUS,gpu_memory_utilization=0.7,max_model_len=2048 \
    --tasks gita_physical_state \
    --output_path ${OUTPUT_DIR}_physical \
    --batch_size $BATCH_SIZE \
    --log_samples \
    --include_path ../tasks \
    --load_local \
    --local_base_dir ../datasets \

rm -rf "${GITA4CALAMITA_PREFIX}_story_predata.json"
rm -rf "${GITA4CALAMITA_PREFIX}_story_data.json"
rm -rf "${GITA4CALAMITA_PREFIX}_conflict_data.json"