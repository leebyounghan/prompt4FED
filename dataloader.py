import h5py
import os
import pickle
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sentence_transformers import SentenceTransformer
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
from common import create_lda_partitions
from collections import Counter
class CustomDataset:
    def __init__(self, args):
        self.args = args

    def bert_clustering(self, corpus):
        embedding_data = {}
        corpus_embeddings = []
        if not self.args.embedding_exist:
            embedder = SentenceTransformer(
                "distilbert-base-nli-stsb-mean-tokens", device="cuda:0"
            )  # server only
            corpus_embeddings = embedder.encode(
                corpus, show_progress_bar=True, batch_size=self.args.batch
            )  # smaller batch size for GPU

            embedding_data["data"] = corpus_embeddings
        else:
            corpus_embeddings = corpus

        ### KMEANS clustering
        print("start Kmeans")
        num_clusters = self.args.client_num
        clustering_model = KMeans(n_clusters=num_clusters)
        clustering_model.fit(corpus_embeddings)
        cluster_assignment = clustering_model.labels_
        print("end Kmeans")
        # TODO: read the center points

        return cluster_assignment, embedding_data

    def load_datasets(self):
        corpus = []

        print("start reading data")
        data = h5py.File(self.args.data_file, "r") # Task Type 확인 시, csv파일의 경우 attributes에 따라 파일 읽기가 수행되지 않을 수 있기에 h5py로 읽기

        # Task Type 확인 / 파일 읽기를 csv 파일로 할 경우, 아래 코드를 수정하시면 됩니다
        if (
            self.args.task_type == "name_entity_recognition"
        ):  # specifically wnut and wikiner datesets
            for i in data["X"].keys():
                sentence = data["X"][i][()]
                sentence = [i.decode("UTF-8") for i in sentence]
                corpus.append(" ".join(sentence))

        elif self.args.task_type == "reading_comprehension":  # specifically Squad1.1 dataset
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

        elif self.args.task_type == "seq2seq":
            keys = list(data["Y"].keys())  # f["Y"]의 키들을 리스트로 변환
            num_entries = len(keys)
            for i in range(0, num_entries, 2):
                if i + 1 < num_entries:  # 입력과 출력이 모두 있는지 확인
                    input_sentence = data["Y"][keys[i]][()].decode("UTF-8")
                    output_sentence = data["Y"][keys[i + 1]][()].decode("UTF-8")

                    # 입력 또는 출력 텍스트가 비어있는 경우 해당 쌍 제거
                    if input_sentence.strip() and output_sentence.strip():
                        corpus.append((input_sentence, output_sentence))

        else:
            for i in data["X"].keys():
                sentence = data["X"][i][()].decode("UTF-8")
                corpus.append(sentence)

        data.close()

        print("start process embedding data and kmeans partition")
        cluster_assignment = []
        embedding_data = []

        if not self.args.overwrite:
            cluster_assignment, corpus_embedding = self.bert_clustering(corpus)
            embedding_data = {}
            embedding_data["data"] = corpus_embedding
            with open(self.args.embedding_file, "wb") as f:
                pickle.dump(embedding_data, f, pickle.HIGHEST_PROTOCOL)
        else:
            with open(self.args.embedding_file, "rb") as f:
                embedding_data = pickle.load(f)
                embedding_data = embedding_data["data"]
                if isinstance(embedding_data, dict):
                    embedding_data = embedding_data["data"]
                cluster_assignment, corpus_embedding = self.bert_clustering(corpus)

        df = pd.DataFrame(corpus, columns=["Input", "Label"])
        df['assigned'] = cluster_assignment

        return df




class CustomPartition:
    def __init__(self, args):
        self.num_clients = args.num_clients
        self.dataframe = args.dataframe
        self.alpha = args.alpha
        self.val_ratio = args.val_ratio
        self.imbalance = args.imbalance
        self.random_state = args.random_state

    def data_partition(self):
        idx = np.array(range(len(self.dataframe)))
        labels = self.dataframe.assigned.to_numpy()
        dataset = [idx, labels]
        unique_labels = np.unique(labels)

        partition = [[[], []] for _ in range(self.num_clients)]

        if not self.imbalance:  # IID, Uniformly distributed
            for label in unique_labels:
                label_indices = np.where(labels == label)[0]
                np.random.shuffle(label_indices)

                split_size = len(label_indices) // self.num_clients

                for i in range(self.num_clients):
                    start_idx = i * split_size
                    end_idx = (i + 1) * split_size if i < self.num_clients - 1 else len(label_indices)
                    client_indices = label_indices[start_idx:end_idx]
                    partition[i][0].extend(idx[client_indices])
                    partition[i][1].extend(labels[client_indices])
        else:
            # Non-iid partition
            partition, _ = create_lda_partitions(
                dataset, num_partitions=self.num_clients, concentration=self.alpha, accept_imbalanced=True
            )

        return partition

    def data_distribution(self, partitions):
        client_dataframes = []

        for partition in partitions:
            idx_list = partition[0]
            label_list = partition[1]

            client_df = self.dataframe.iloc[idx_list].copy()
            client_df['assigned'] = label_list

            train_df, test_df = train_test_split(client_df, test_size=self.val_ratio, random_state=self.random_state)

            client_dataframes.append((train_df, test_df))

        return client_dataframes

    def data_visualization(self, partitions):
        plt.figure(figsize=(12, 6))

        for i, client_partition in enumerate(partitions):
            plt.subplot(1, len(partitions), i + 1)
            labels = client_partition[1]
            label_counts = Counter(labels)
            assigned_values = sorted(label_counts.keys())
            counts = [label_counts[value] for value in assigned_values]

            plt.bar(assigned_values, counts, alpha=0.5, label='Data')
            plt.title(f"Client {i + 1} Assigned")
            plt.xlabel("Assigned Value")
            plt.ylabel("Count")
            plt.xticks(rotation=0)
            plt.legend()

        plt.tight_layout()
        plt.show()

    def split_visualization(self, client_dataframes):
        plt.figure(figsize=(12, 6))

        for i, (train_df, test_df) in enumerate(client_dataframes):
            plt.subplot(1, len(client_dataframes), i + 1)

            train_labels = train_df['assigned']
            train_label_counts = Counter(train_labels)
            train_assigned_values = sorted(train_label_counts.keys())
            train_counts = [train_label_counts[value] for value in train_assigned_values]

            test_labels = test_df['assigned']
            test_label_counts = Counter(test_labels)
            test_assigned_values = sorted(test_label_counts.keys())
            test_counts = [test_label_counts[value] for value in test_assigned_values]

            plt.bar(train_assigned_values, train_counts, alpha=0.5, label='Train')
            plt.bar(test_assigned_values, test_counts, alpha=0.5, label='Test')

            plt.title(f"Client {i + 1} Assigned")
            plt.xlabel("Assigned Value")
            plt.ylabel("Count")
            plt.xticks(rotation=0)
            plt.legend()

        plt.tight_layout()
        plt.show()

    def count_labels(self, partitions):
        # 클라이언트 별 레이블 수 세기
        client_label_counts = []

        for client_partition in partitions:
            labels = client_partition[1]
            label_counts = Counter(labels)
            client_label_counts.append(label_counts)

        for label_count in client_label_counts:
            print(label_count)
