# -----------------------------------------------------------------------------------
# RRHF: Rank Responses to Align Language Models with Human Feedback without tears
# https://github.com/GanjinZero/RRHF
# -----------------------------------------------------------------------------------

export MASTER_ADDR=localhost
export MASTER_PORT=7834
export CUDA_VISIBLE_DEVICES="7"

MODEL_DIR=chainyo/alpaca-lora-7b
OUT_DIR=../outputs/generated_data
DATA="esnli"
NPROC=1
diverse_beam=2
expansion=3

mkdir -p $OUT_DIR
# torchrun --nproc_per_node $NPROC --master_port 7835 response_gen.py \
#                         --base_model $MODEL_DIR \
#                         --data_path $DATA \
#                         --out_path $OUT_DIR \
#                         --diverse_beam $diverse_beam \
#                         --expansion $expansion \
#                         --batch_size 1

# python ./split_files.py --dataset $DATA --out_path $OUT_DIR --num_process $NPROC --expansion $expansion
python ./run_scoring_responses.py --num_process $NPROC --expansion $expansion --output $OUT_DIR
python make_data.py $OUT_DIR
python ./sent_check.py --diverse_beam $diverse_beam --expansion $expansion --output $OUT_DIR > $OUT_DIR/test.log
