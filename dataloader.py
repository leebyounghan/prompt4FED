import h5py
import argparse
import os
import pickle
import json
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split


# dataloader.py 실행 시 아래 코드를 터미널에 입력하시면 clients 폴더에 clients 수만큼 train.csv와 test.csv 파일이 생성됩니다

# DATA_DIR="dataloader 경로"

# attributes : input(sentence1) label(sentence2) assigned(클러스터링 결과)

# CUDA_VISIBLE_DEVICES=0 \
# python -m dataloader  \
#     --cluster_number 10 \
#     --data_file ${DATA_DIR}/data_files/data.h5 \
#     --batch 32 \
#     --embedding_file ${DATA_DIR}/embedding_files/data_embedding.pkl \
#     --task_type sequence_to_sequence \
#     --overwrite


# FedML의 Bert-Clustering
def get_embedding_Kmeans(embedding_exist, corpus, N_clients, batch=16):
    embedding_data = {}
    corpus_embeddings = []
    if embedding_exist == False:
        embedder = SentenceTransformer(
            "distilbert-base-nli-stsb-mean-tokens", device="cuda:0"
        )  # server only
        corpus_embeddings = embedder.encode(
            corpus, show_progress_bar=True, batch_size=batch
        )  # smaller batch size for gpu

        embedding_data["data"] = corpus_embeddings
    else:
        corpus_embeddings = corpus
        
    ### KMEANS clustering
    print("start Kmeans")
    num_clusters = N_clients
    clustering_model = KMeans(n_clusters=num_clusters)
    clustering_model.fit(corpus_embeddings)
    cluster_assignment = clustering_model.labels_
    print("end Kmeans")
    # TODO: read the center points

    return cluster_assignment, embedding_data

def arguments():
    
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--cluster_number",
        type=int,
        default="10",
        metavar="CN",
        help="client number for lda partition",
    )

    parser.add_argument(
        "--batch",
        type=int,
        default="32",
        metavar="CN",
        help="batch size for sentenceBERT",
    )

    parser.add_argument(
        "--data_file",
        type=str,
        default="/data_files/data.h5",
        metavar="DF",
        help="data pickle file path",
    )

    parser.add_argument(
        "--embedding_file",
        type=str,
        default="/data_files/data_embedding.h5",
        metavar="EF",
        help="embedding pickle file path",
    )

    parser.add_argument(
        "--task_type",
        type=str,
        metavar="TT",
        default="sequence_to_sequence",
        help="task type",
    )

    parser.add_argument(
        "--overwrite",
        action="store_false",
        default=True,
        help="True if embedding data file does not exist False if it does exist",
    )
    
    return parser


def load_datasets():
    
    # Argument 설정
    parser = arguments()
    args = parser.parse_args()
    corpus = []
    
    N_Clients = args.cluster_number
    
    print("start reading data")
    data = h5py.File(args.data_file, "r") # Task Type 확인 시, csv파일의 경우 attributes에 따라 파일 읽기가 수행되지 않을 수 있기에 h5py로 읽기
    
    
    # Task Type 확인 / 파일 읽기를 csv 파일로 할 경우, 아래 코드를 수정하시면 됩니다
    if (
        args.task_type == "name_entity_recognition"
    ):  # specifically wnut and wikiner datesets
        for i in data["X"].keys():
            sentence = data["X"][i][()]
            sentence = [i.decode("UTF-8") for i in sentence]
            corpus.append(" ".join(sentence))

    elif args.task_type == "reading_comprehension":  # specifically Squad1.1 dataset
        for i in data["context_X"].keys():
            question_components = []
            # context = f['context_X'][i][()].decode('UTF-8')
            question = data["question_X"][i][()].decode("UTF-8")
            answer_start = data["Y"][i][()][0]
            answer_end = data["Y"][i][()][1]
            answer = data["context_X"][i][()].decode("UTF-8")[answer_start:answer_end]

            question_components.append(question)
            question_components.append(answer)
            corpus.append(" ".join(question_components))

    elif args.task_type == "sequence_to_sequence":
        keys = list(data["Y"].keys())  # f["Y"]의 키들을 리스트로 변환
        num_entries = len(keys)
        for i in range(0, num_entries, 2):
            if i + 1 < num_entries:  # 입력과 출력이 모두 있는지 확인
                input_sentence = data["Y"][keys[i]][()].decode("UTF-8")
                output_sentence = data["Y"][keys[i + 1]][()].decode("UTF-8")
                corpus.append((input_sentence, output_sentence))

    else:
        for i in data["X"].keys():
            sentence = data["X"][i][()].decode("UTF-8")
            corpus.append(sentence)
            
    data.close()
    
    
    
    print("start process embedding data and kmeans partition")
    cluster_assignment = []
    embedding_data = []
    
    if args.overwrite == False:
        cluster_assignment, corpus_embedding = get_embedding_Kmeans(
            False, corpus, N_Clients, args.batch
        )
        embedding_data = {}
        embedding_data["data"] = corpus_embedding
        with open(args.embedding_file, "wb") as f:
            pickle.dump(embedding_data, f, pickle.HIGHEST_PROTOCOL)
    else:
        with open(args.embedding_file, "rb") as f:
            embedding_data = pickle.load(f)
            embedding_data = embedding_data["data"]
            if isinstance(embedding_data, dict):
                embedding_data = embedding_data["data"]
            cluster_assignment, corpus_embedding = get_embedding_Kmeans(
                True, embedding_data, N_Clients, args.batch
            )
    
    df = pd.DataFrame(corpus, columns=["Input", "Label"])
    df['assigned'] = cluster_assignment
    
    # 클러스터링 결과를 기반으로 클라이언트 별로 데이터 인덱스를 저장할 딕셔너리 생성
    cluster_indices = {i: [] for i in range(N_Clients)}

    # 각 클러스터별로 데이터 인덱스를 딕셔너리에 저장
    for idx, row in df.iterrows():
        cluster_idx = row['assigned']
        cluster_indices[cluster_idx].append(idx)

    # 각 클라이언트에 대해 train과 test 데이터로 랜덤하게 분할하는 함수 정의
    def split_train_test_data(indices, test_size=0.2):
        train_indices, test_indices = train_test_split(indices, test_size=test_size, random_state=42)
        return train_indices, test_indices

    # 클러스터링 결과를 기반으로 클라이언트 별로 데이터 분할
    client_data_splits = {}
    for client_id in range(N_Clients):
        client_indices = cluster_indices[client_id]
        train_indices, test_indices = split_train_test_data(client_indices)
        client_data_splits[client_id] = {
            'train': train_indices,
            'test': test_indices
        }

    # 클라이언트 데이터를 폴더별로 저장
    output_folder = 'clients'
    os.makedirs(output_folder, exist_ok=True)  # clients 폴더가 없으면 생성

    for client_id, data_splits in client_data_splits.items():
        train_indices = data_splits['train']
        test_indices = data_splits['test']
        train_df = df.iloc[train_indices]
        test_df = df.iloc[test_indices]
    
        client_folder = os.path.join(output_folder, f"client_{client_id}")
        os.makedirs(client_folder, exist_ok=True)  # 클라이언트 폴더 생성

        train_filename = os.path.join(client_folder, "train.csv")
        test_filename = os.path.join(client_folder, "test.csv")
    
        train_df.to_csv(train_filename, index=False)
        test_df.to_csv(test_filename, index=False)
        
load_datasets()