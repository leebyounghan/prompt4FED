DATA_DIR="./fednlp_data/"
CUDA_VISIBLE_DEVICES=0 \
python -m dataloader  \
    --cluster_number 10 \
    --data_file ${DATA_DIR}/data_files/cornell_movie_dialogue_data.h5 \
    --batch 4 \
    --embedding_file ${DATA_DIR}/embedding_files/data_embedding.pkl \
    --task_type sequence_to_sequence \
    --overwrite