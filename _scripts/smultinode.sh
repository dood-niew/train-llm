#!/bin/bash
echo myuser=`whoami`
echo COUNT_NODE=$COUNT_NODE
echo LD_LIBRARY_PATH = $LD_LIBRARY_PATH
echo PATH = $PATH
echo which mpicc `which mpicc`
echo HOSTNAMES = $HOSTNAMES
echo hostname = `hostname`
echo MASTER_ADDR= $MASTER_ADDR
echo MASTER_PORT= $MASTER_PORT

# export NCCL_TIMEOUT=3600000
# export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
# export HF_HOME=../.cache


accelerate launch \
    --num_processes $(( 4 * $COUNT_NODE )) \
    --num_machines $COUNT_NODE \
    --multi_gpu \
    --mixed_precision bf16 \
    --machine_rank $SLURM_PROCID \
    --main_process_ip $MASTER_ADDR \
    --main_process_port $MASTER_PORT \
    /project/lt200258-aithai/llm/code/train/instuction-finetune/lora_finetune/_scripts/train.py \
        --do_train \
        --model_name_or_path /project/lt200258-aithai/llm/base-model/google/gemma-2-27b \
        --attn_implementation flash_attention_2 \
        --data_train_path /project/lt200258-aithai/llm/process-dataset/llm_text/fine-tune/data_fine-tune/dataset_merge_all/apt3-filter/train \
        --seed 42 \
        --bf16 True \
        --model_max_length 2048 \
        --output_dir /project/lt200258-aithai/llm/code/train/instuction-finetune/lora_finetune/output/apt3-8k-gemma-2-27b-test5 \
        --num_train_epochs 1 \
        --per_device_train_batch_size 1 \
        --per_device_eval_batch_size 1 \
        --gradient_accumulation_steps 1 \
        --save_strategy "epoch" \
        --save_total_limit 1 \
        --logging_strategy 'steps' \
        --logging_steps 1 \
        --logging_first_step True \
        --learning_rate 1.0e-4 \
        --weight_decay 0.1 \
        --warmup_ratio 0.03 \
        --lr_scheduler_type 'cosine' \
        --deepspeed /project/lt200258-aithai/jack/playground/ds3.json \
        --bf16 True \
        --gradient_checkpointing True \
        --max_grad_norm 1.00 \
        --lora_target q_proj,v_proj \
        --lora_alpha 16 \
        --lora_dropout 0.1 \
        --lora_rank 8 \
        --max_position_embeddings 2048 \
        --eos_token_id 1 \
        --pad_token_id 0 \
        --is_lora True \
        --model_template gemma \
        --have_system False \
        --max_steps 5 \
        --optim "adamw_torch"

        #"/project/lt200258-aithai/meo/gemma/deepspeed_stage3.json"
        #         --eval_steps 1000 \
        # --save_steps 1000 \