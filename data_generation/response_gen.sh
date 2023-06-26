export MASTER_ADDR=localhost
export MASTER_PORT=7834
export CUDA_VISIBLE_DEVICES="0"
MODEL_DIR=$1
OUT_DIR=$2
# DATA="Dahoas/rm-static"
DATA="esnli"
NPROC=1
mkdir -p $OUT_DIR
# torchrun --nproc_per_node $NPROC --master_port 7834 response_gen.py \
#                         --base_model $MODEL_DIR \
#                         --data_path $DATA \
#                         --out_path $OUT_DIR \
#                         --diverse_beam 4 \
#                         --batch_size 4

python ./split_files.py --dataset $DATA --data_path $OUT_DIR --num_process $NPROC
# bash ./scoring_responses.sh $OUT_DIR
# python make_data.py $OUT_DIR

# bash response_gen.sh chainyo/alpaca-lora-7b ../generated_data
