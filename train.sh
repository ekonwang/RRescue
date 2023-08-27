# ------------------------------------------------------------------------------- #
# RRHF: Rank Responses to Align Language Models with Human Feedback without tears
# ------------------------------------------------------------------------------- #
export SAVE_PATH=$1 #  /mnt/data/user/wang_yikun/instr/alpaca-7b/eos/esnli/$tag
export DATA_PATH=$2 #  /workspace/Rank/outputs/data/esnli/data_10k/train_partial.json
num_resp=$3 # 2-6
strategy=rank # rank / sft
use_eos_token=0 # 1 / 0

prompt_template=esnli_prompt.json
dataset=esnli
weight=0.1
export MODEL_PATH=chainyo/alpaca-lora-7b
export MASTER_ADDR="localhost"
export MASTER_PORT=$((RANDOM % 1000 + 10000))
export WANDB_DISABLED=true
prompt_file=./data_generation/configs/$prompt_template
wandb offline
mkdir -p ./logs

IFS=',' read -ra DEVICES <<< "$CUDA_VISIBLE_DEVICES"
NPROC=${#DEVICES[@]}

python3 -m torch.distributed.launch --master_addr ${MASTER_ADDR} --master_port ${MASTER_PORT} --nproc_per_node=$NPROC --use_env train.py \
    --model_name_or_path $MODEL_PATH \
    --data_path $DATA_PATH \
    --prompt_file $prompt_file --num_resp ${num_resp} \
    --use_eos_token $use_eos_token \
    --bf16 True \
    --output_dir $SAVE_PATH \
    --optim "adamw_torch" \
    --num_train_epochs 1 \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 1 \
    --gradient_accumulation_steps 1 \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 8000 \
    --save_total_limit 40 \
    --learning_rate 2e-5 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --fsdp "full_shard auto_wrap" \
    --fsdp_transformer_layer_cls_to_wrap 'LlamaDecoderLayer' \
    --model_max_length 512 \
    --rrhf_weight ${weight} --train_strategy $strategy | tee ./logs/test.log

