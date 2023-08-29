import h5py
import os
import pickle
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import argparse
from sklearn.model_selection import train_test_split
from common import create_lda_partitions
from collections import Counter

class CustomPartition:
    def __init__(self, args):
        self.num_clients = args.client_number
        self.datafile = args.data_file
        self.alpha = args.alpha
        self.val_ratio = args.val_ratio
        self.imbalance = args.imbalance
        self.random_state = args.random_state

    def data_partition(self):
        dataframe = pd.read_csv(self.datafile)
        
        idx = np.array(range(len(dataframe)))
        labels = dataframe.assigned.to_numpy()
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
        dataframe = pd.read_csv(self.datafile)

        output_folder = 'clients'
        os.makedirs(output_folder, exist_ok=True)  
        
        for client_id, partition in enumerate(partitions):
            idx_list = partition[0]
            label_list = partition[1]

            client_df = dataframe.iloc[idx_list].copy()
            client_df['assigned'] = label_list

            train_df, test_df = train_test_split(client_df, test_size=self.val_ratio, random_state=self.random_state)

            client_dataframes.append((train_df, test_df))
            
            client_folder = os.path.join(output_folder, f"client_{client_id}")
            os.makedirs(client_folder, exist_ok=True)  # 클라이언트 폴더 생성   

            train_filename = os.path.join(client_folder, "train.csv")
            test_filename = os.path.join(client_folder, "test.csv")
        
            train_df.to_csv(train_filename, index=False)
            test_df.to_csv(test_filename, index=False)


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
            
            
def partition():
    # partition dataset
    parser = arguments()
    args = parser.parse_args()
    
    # Create CustomPartition instance
    cp = CustomPartition(args)
    partitions = cp.data_partition()
    
    # distribute clients'data
    cp.data_distribution(partitions)
    

def arguments():
    
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--client_number",
        type=int,
        default=5,
        metavar="CN",
        help="client number for lda partition",
    )

    parser.add_argument(
        "--alpha",
        type=float,
        default=5,
        metavar="A",
        help="lda partition controllable param",
    )
    
    parser.add_argument(
        "--val_ratio",
        type=float,
        default=0.2,
        metavar="VR",
        help="lda partition controllable param",
    )
    
    parser.add_argument(
        "--random_state",
        type=int,
        default=0,
        metavar="RS",
        help="data split seed",
    )

    parser.add_argument(
        "--data_file",
        type=str,
        default="/data_files/dataset.csv",
        metavar="DF",
        help="data file with csv",
    )

    parser.add_argument(
        "--imbalance",
        metavar="IM",
        default=True,
        help="True if Non-iid distribution else iid distribution",
    )
    
    return parser

partition()