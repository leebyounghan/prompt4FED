# datafile : DATA_DIR/dataset.csv
DATA_DIR="./dataset"
CUDA_VISIBLE_DEVICES=0 \
python -m partitions  \
    --client_number 5 \
    --alpha 5 \
    --val_ratio 0.2 \
    --random_state 0 \
    --data_file ${DATA_DIR}/dataset.csv \
    --imbalance True