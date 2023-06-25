export MASTER_ADDR=localhost
export MASTER_PORT=7834
export CUDA_VISIBLE_DEVICES="0"
MODEL_DIR=$1
OUT_DIR=$2
mkdir -p $OUT_DIR
torchrun --nproc_per_node 1 --master_port 7834 response_gen.py \
                        --base_model $MODEL_DIR \
                        --data_path "Dahoas/rm-static" \
                        --out_path $OUT_DIR \
                        --diverse_beam 4 \
                        --batch_size 1

# python ./split_files.py $OUT_DIR $OUT_DIR
# bash ./scoring_responses.sh $OUT_DIR
# python make_data.py $OUT_DIR

# bash response_gen.sh chainyo/alpaca-lora-7b /workspace/RL/generated_data
