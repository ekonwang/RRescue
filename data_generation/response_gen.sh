export MASTER_ADDR=localhost
export MASTER_PORT=7834
export CUDA_VISIBLE_DEVICES="0"
MODEL_DIR=chainyo/alpaca-lora-7b
OUT_DIR=../generated_data
DATA="esnli"
NPROC=1
mkdir -p $OUT_DIR
# torchrun --nproc_per_node $NPROC --master_port 7834 response_gen.py \
#                         --base_model $MODEL_DIR \
#                         --data_path $DATA \
#                         --out_path $OUT_DIR \
#                         --diverse_beam 4 \
#                         --batch_size 4

# python ./split_files.py --dataset $DATA --out_path $OUT_DIR --num_process $NPROC
python ./run_scoring_responses.py --dataset $DATA --num_process $NPROC
# python make_data.py $OUT_DIR
