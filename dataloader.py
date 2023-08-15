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
import matplotlib.pyplot as plt

# Bert-Clustering of FedML
def bert_clustering(embedding_exist, corpus, N_clients, batch=16):
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


# Data partition
def data_partition(num_clients, data_dir, df, alpha, val_ratio=0.2):
    if not os.path.exists(data_dir):
        os.mkdir(data_dir)
        
    assigned_groups = df.groupby('assigned')
    client_data = {i: {'train': [], 'val': []} for i in range(1, num_clients + 1)}
    
    if alpha == 1: # IID, Uniformly distributed
        for assigned_value, group_df in assigned_groups:
            client_idx = 0
            for _, row in group_df.iterrows():
                client_idx = client_idx % num_clients + 1  # Cycle through clients
                client_data[client_idx]['train'].append(row)
                client_idx += 1
                
    else: # Non-IID
        for client_id in range(1, num_clients + 1):
            assigned_value = client_id - 1
            cluster_data = assigned_groups.get_group(assigned_value)
        
            num_selected = int(len(cluster_data) * alpha)
            classes_in_cluster = cluster_data['assigned'].nunique()
            
            # proportion per class
            class_sampling_ratios = [1.0 / classes_in_cluster] * classes_in_cluster
            
            # sampling num per class
            num_samples_per_class = [int(num_selected * ratio) for ratio in class_sampling_ratios]
            
            selected_data_per_class = []
            
            for _, num_samples in enumerate(num_samples_per_class):
                class_data = cluster_data[cluster_data['assigned'] == assigned_value]
                selected_data = class_data.sample(n=num_samples, replace=False)
                selected_data_per_class.append(selected_data)
            
            selected_data = pd.concat(selected_data_per_class)
            remaining_data = df[df['assigned'] != assigned_value].sample(n=len(cluster_data) - num_selected, replace=False)
            cluster_data = pd.concat([selected_data, remaining_data])
            
            for _, row in cluster_data.iterrows():
                client_data[client_id]['train'].append(row)
            
        
    # generate each client's file  
    for i, client in client_data.items():
        client_dir = os.path.join(data_dir, f"client_{i}")
        os.makedirs(client_dir, exist_ok=True)
    
        train_data = pd.DataFrame(client['train'])
        train_set, val_set = train_test_split(train_data, test_size=val_ratio, random_state=42)
    
        train_path = os.path.join(client_dir, "train.csv")
        val_path = os.path.join(client_dir, "test.csv")
    
        train_set.to_csv(train_path, index=False)
        val_set.to_csv(val_path, index=False)
        
    print("Data splitting and saving complete.")
    
    
# load dataset
def load_datasets(client_num, data_file, task_type, embedding_file, overwrite=False, batch=16):
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
        cluster_assignment, corpus_embedding = bert_clustering(
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
            cluster_assignment, corpus_embedding = bert_clustering(
                True, embedding_data, client_num, batch
            )
    
    df = pd.DataFrame(corpus, columns=["input", "label"])
    df['assigned'] = cluster_assignment
    
    return df


def visualize_distribution(data_dir):
    # 클라이언트 폴더 경로 설정
    client_folder = data_dir

    # 각 클라이언트 별 데이터 저장을 위한 리스트 초기화
    client_train_distributions = []
    client_test_distributions = []

    # 각 클라이언트 폴더 순회
    for client_subfolder in os.listdir(client_folder):
        client_subfolder_path = os.path.join(client_folder, client_subfolder)
    
        # 클라이언트 폴더 내의 train.csv와 test.csv 파일 경로 설정
        train_csv_path = os.path.join(client_subfolder_path, "train.csv")
        test_csv_path = os.path.join(client_subfolder_path, "test.csv")
    
        # train.csv 파일 읽기
        if os.path.exists(train_csv_path):
            train_df = pd.read_csv(train_csv_path)
            assigned_train_counts = train_df['assigned'].value_counts()
            client_train_distributions.append(assigned_train_counts)
    
        # test.csv 파일 읽기
        if os.path.exists(test_csv_path):
            test_df = pd.read_csv(test_csv_path)
            assigned_test_counts = test_df['assigned'].value_counts()
            client_test_distributions.append(assigned_test_counts)

    plt.figure(figsize=(12, 6))
    for i in range(len(client_train_distributions)):
        plt.subplot(1, len(client_train_distributions), i + 1)
        assigned_values = sorted(list(client_train_distributions[i].index))  # assigned 값 순서대로 정렬
        plt.bar(assigned_values, client_train_distributions[i][assigned_values], alpha=0.5, label='Train')
        plt.bar(assigned_values, client_test_distributions[i][assigned_values], alpha=0.5, label='Test')
        plt.title(f"Client{i+1} Assigned")
        plt.xlabel("Assigned Value")
        plt.ylabel("Count")
        plt.xticks(rotation=0)
        plt.legend()

    plt.tight_layout()
    plt.show()
