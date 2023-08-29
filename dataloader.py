import h5py
import os
import pickle
import json
import pandas as pd
import numpy as np
import argparse

from sentence_transformers import SentenceTransformer
from sklearn.cluster import KMeans

class CustomDataset:
    def __init__(self, args):
        self.args = args

    def bert_clustering(self, corpus):
        embedding_data = {}
        corpus_embeddings = []
        embedder = SentenceTransformer(
                "distilbert-base-nli-stsb-mean-tokens", device="cuda:0"
        )  # server only
        corpus_embeddings = embedder.encode(
            corpus, show_progress_bar=True, batch_size=self.args.batch
        )  # smaller batch size for GPU

        embedding_data["data"] = corpus_embeddings
            

        ### KMEANS clustering
        print("start Kmeans")
        num_clusters = self.args.cluster_number
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
            
    
def data_load():
    # load dataset
    parser = arguments()
    args = parser.parse_args()
    
    # Create CustomDataset instance
    custom_dataset = CustomDataset(args)
    # Load datasets
    dataframe = custom_dataset.load_datasets()
    
    folder = 'dataset'
    os.makedirs(folder, exist_ok=True)  
    
    filename = os.path.join(folder, "dataset.csv")
        
    dataframe.to_csv(filename, index=False)
    
def arguments():
    
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--cluster_number",
        type=int,
        default="5",
        metavar="CN",
        help="client number for lda partition",
    )

    parser.add_argument(
        "--batch",
        type=int,
        default="4",
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
        default="seq2seq",
        help="task type",
    )

    parser.add_argument(
        "--overwrite",
        action="store_false",
        default=True,
        help="True if embedding data file does not exist False if it does exist",
    )
    
    return parser

data_load()
