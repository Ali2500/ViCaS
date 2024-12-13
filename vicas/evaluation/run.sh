PRED_DIR=$1
shift

GT_DIR=$1
shift

# Evaluate LG-VIS (single process)
python3 vicas/evaluation/main.py --pred_dir ${PRED_DIR} --gt_dir ${GT_DIR} --skip_captions $@

# Evaluate captions (multi-GPU) and append the scores to the output from the previous step
torchrun --nproc_per_node=8 --master_port 2222 vicas/evaluation/main.py --pred_dir ${PRED_DIR} --gt_dir ${GT_DIR} --skip_masks --append_metrics $@