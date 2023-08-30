#/bin/bash

DATA="esnli"
OUTPUT_DIR=$1
MODEL_DIR=$OUTPUT_DIR
prompt_file=./data_generation/configs/esnli_prompt_with_examples.json

# while $MODEL_DIR/tokenizer.model does not exists, then sleep for 10 seconds
while [ ! -f $MODEL_DIR/tokenizer.model ]
do
    echo "Waiting for $MODEL_DIR/tokenizer.model"
    sleep 1200
done

port=$((RANDOM % 1000 + 10000))
IFS=',' read -ra DEVICES <<< "$CUDA_VISIBLE_DEVICES"
NPROC=${#DEVICES[@]}

if [ ! -f $OUTPUT_DIR/eval.json ]; then
    echo $port
    torchrun --nproc_per_node $NPROC --master_port $port load_for_eval.py \
    --model_name_or_path $MODEL_DIR \
    --data_path $DATA --prompt_file $prompt_file \
    --output_path $OUTPUT_DIR \
    --batch_size 8 --truncate $2 \
    --fp16 \
    --tag eval
fi

python -u esnli_eval.py \
--input_file $OUTPUT_DIR/eval.json \
--data_path $DATA | tee $OUTPUT_DIR/eval.log

python -u esnli_eval.py \
--input_file $OUTPUT_DIR/eval.json \
--data_path $DATA --truncate 1000 | tee $OUTPUT_DIR/eval_1k.log
