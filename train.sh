#!/bin/bash

# ========== Training Config ==========
export num_train_epochs=3
export save_strategy="epoch"
export evaluation_strategy="epoch"
export logging_strategy="steps"
export logging_steps=5
export seed=1234

# ========== Model ==========
export model_name="csebuetnlp/banglabert"

# ========== Optimization ==========
export learning_rate=2e-5 
export warmup_ratio=0.1
export gradient_accumulation_steps=4
export weight_decay=0.01
export lr_scheduler_type="linear"

# ========== Input / Output ==========
export dataset_dir="/Users/jbc/Documents/punc_restoration/bengali_punctuation_jsonl_data1/"
# export validation_file="/Users/jbc/Documents/punc_restoration/bengali_punctuation_jsonl_data1/validate/valid.jsonl"
export output_dir="/Users/jbc/Documents/punc_restoration/bengali_punctuation_jsonl_data1/Training_outputs/"

# ========== Batch & Seq ==========
export PER_DEVICE_TRAIN_BATCH_SIZE=8
export PER_DEVICE_EVAL_BATCH_SIZE=8
export MAX_SEQUENCE_LENGTH=512

# ========== Optional Arguments ==========
optional_arguments=(
    "--metric_for_best_model weighted_avg_f1"
    "--greater_is_better true"
    "--load_best_model_at_end true"
    "--logging_first_step"
    "--overwrite_cache"
    "--cache_dir cache_dir/"
    "--dataset_dir $dataset_dir"
    # "--validation_file $validation_file"
    "--do_eval"
    # "--fp16"
    # "--fp16_backend auto"
)

# ========== Weights & Biases ==========
export WANDB_DISABLED=true

# ========== Run Training ==========
python ./src/token_classification.py \
    --model_name_or_path $model_name \
    --output_dir $output_dir \
    --learning_rate $learning_rate \
    --warmup_ratio $warmup_ratio \
    --gradient_accumulation_steps $gradient_accumulation_steps \
    --weight_decay $weight_decay \
    --lr_scheduler_type $lr_scheduler_type \
    --per_device_train_batch_size $PER_DEVICE_TRAIN_BATCH_SIZE \
    --per_device_eval_batch_size $PER_DEVICE_EVAL_BATCH_SIZE \
    --max_seq_length $MAX_SEQUENCE_LENGTH \
    --logging_strategy $logging_strategy \
    --logging_steps $logging_steps \
    --seed $seed \
    --overwrite_output_dir \
    --num_train_epochs $num_train_epochs \
    --save_strategy $save_strategy \
    --evaluation_strategy $evaluation_strategy \
    --do_train \
    "${optional_arguments[@]}"
