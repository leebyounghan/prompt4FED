# datafile : DATA_DIR/datafiles/cornell_movie_dialogue_data.h5
# embeddingfile : DATA_DIR/embedding_files/cornell_movie_dialogue_embedding.pkl

DATA_DIR="./fednlp_data"
CUDA_VISIBLE_DEVICES=0 \
python -m dataloader  \
    --cluster_number 5 \
    --data_file ${DATA_DIR}/data_files/cornell_movie_dialogue_data.h5 \
    --batch 4 \
    --embedding_file ${DATA_DIR}/embedding_files/cornell_movie_dialogue_embedding.pkl \
    --task_type seq2seq \
    --overwrite