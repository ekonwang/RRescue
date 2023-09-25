# ------------------------------------------------------------------------------- #
# RRHF: Rank Responses to Align Language Models with Human Feedback without tears
# ------------------------------------------------------------------------------- #
export WANDB_DISABLED=true

SAVE_PATH=$1
DATA_PATH=$2
num_resp=$3 # 2-6
strategy=rank # rank / sft
use_eos_token=0 # 1 / 0

prompt_template=esnli_prompt.json
dataset=esnli
MODEL_PATH=$MODEL_FOR_TRAIN # chainyo/alpaca-lora-7b / NousResearch/Llama-2-7b-hf
MASTER_ADDR="localhost"
MASTER_PORT=$((RANDOM % 1000 + 10000))
prompt_file=./data_generation/configs/$prompt_template
wandb offline
mkdir -p ./logs

# ------------- useful work arounds -------------- #

# if environmental variable RRESCUE_SEED is not set, set it to 42
if [ -z ${RRESCUE_SEED+x} ]; then
    export RRESCUE_SEED=42
fi

# if environmental variable RRHF_WEIGHT is not set, set it to 0.1
if [ -z ${RRHF_WEIGHT+x} ]; then
    export RRHF_WEIGHT=0.1
fi

# if environmental variable GLOBAL_BATCH_SIZE is not set, set it to 8
if [ -z ${GLOBAL_BATCH_SIZE+x} ]; then
    export GLOBAL_BATCH_SIZE=8
fi

# ------------------------------------------------ #
IFS=',' read -ra DEVICES <<< "$CUDA_VISIBLE_DEVICES"
NPROC=${#DEVICES[@]}
gradient_accumulation_steps=$(( (GLOBAL_BATCH_SIZE + NPROC - 1) / NPROC ))

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
    --gradient_accumulation_steps ${gradient_accumulation_steps} \
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
    --rrhf_weight $RRHF_WEIGHT --train_strategy $strategy --seed ${RRESCUE_SEED} | tee ./logs/test.log
