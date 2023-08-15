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

# 디렉토리 생성 
def make_dir(num_clients, data_dir, df, val_ratio=0.2):
    
    if not os.path.exists(data_dir):
        os.mkdir(data_dir)
        
    assigned_groups = df.groupby('assigned')
    
    client_data = {i: {'train': [], 'val': []} for i in range(1, num_clients + 1)}
    
    for assigned_value, group_df in assigned_groups:
        client_idx = 0
        for _, row in group_df.iterrows():
            client_idx = client_idx % num_clients + 1  # Cycle through clients
            client_data[client_idx]['train'].append(row)
            client_idx += 1
    
    for i, client in client_data.items():
        client_dir = os.path.join(data_dir, f"client_{i}")
        os.makedirs(client_dir, exist_ok=True)
    
        train_data = pd.DataFrame(client['train'])
        train_set, val_set = train_test_split(train_data, test_size=val_ratio, random_state=42)
    
        train_path = os.path.join(client_dir, "train.csv")
        val_path = os.path.join(client_dir, "val.csv")
    
        train_set.to_csv(train_path, index=False)
        val_set.to_csv(val_path, index=False)
        
    print("Data splitting and saving complete.")
    
# 데이터셋 로드
def load_datasets(client_num, data_file, task_type, embedding_file, overwrite=False, batch=32):
    
    corpus = []
    
    print("start reading data")
    data = h5py.File(data_file, "r") # Task Type 확인 시, csv파일의 경우 attributes에 따라 파일 읽기가 수행되지 않을 수 있기에 h5py로 읽기
    
    
    # Task Type 확인 / 파일 읽기를 csv 파일로 할 경우, 아래 코드를 수정하시면 됩니다
    if (
        task_type == "name_entity_recognition"
    ):  # specifically wnut and wikiner datesets
        for i in data["X"].keys():
            sentence = data["X"][i][()]
            sentence = [i.decode("UTF-8") for i in sentence]
            corpus.append(" ".join(sentence))

    elif task_type == "reading_comprehension":  # specifically Squad1.1 dataset
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

    elif task_type == "seq2seq":
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
    
    if overwrite == False:
        cluster_assignment, corpus_embedding = get_embedding_Kmeans(
            False, corpus, client_num, batch
        )
        embedding_data = {}
        embedding_data["data"] = corpus_embedding
        with open(embedding_file, "wb") as f:
            pickle.dump(embedding_data, f, pickle.HIGHEST_PROTOCOL)
    else:
        with open(embedding_file, "rb") as f:
            embedding_data = pickle.load(f)
            embedding_data = embedding_data["data"]
            if isinstance(embedding_data, dict):
                embedding_data = embedding_data["data"]
            cluster_assignment, corpus_embedding = get_embedding_Kmeans(
                True, embedding_data, client_num, batch
            )
    
    df = pd.DataFrame(corpus, columns=["input", "label"])
    df['assigned'] = cluster_assignment
    
    return df
