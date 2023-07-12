# ------------------------------------------------------------------------------- #
# RRHF: Rank Responses to Align Language Models with Human Feedback without tears
# ------------------------------------------------------------------------------- #
export MODEL_PATH=chainyo/alpaca-lora-7b
export SAVE_PATH=./output
export DATA_PATH=../outputs/generated_data/train_data.json
export MASTER_ADDR="localhost"
export MASTER_PORT="7889"
export WANDB_DISABLED=true
export CUDA_VISIBLE_DEVICES=7
wandb offline

python3 -m torch.distributed.launch --master_addr ${MASTER_ADDR} --master_port ${MASTER_PORT} --nproc_per_node=1 --use_env train.py \
    --model_name_or_path $MODEL_PATH \
    --data_path $DATA_PATH \
    --bf16 True \
    --output_dir $SAVE_PATH \
    --num_train_epochs 3 \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 1 \
    --gradient_accumulation_steps 8 \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 1 \
    --save_total_limit 40 \
    --learning_rate 2e-5 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --tf32 True --model_max_length 192 # > train.log

    --fsdp "full_shard auto_wrap" \
    --fsdp_transformer_layer_cls_to_wrap 'LlamaDecoderLayer' \
