import argparse
import os

from collections import OrderedDict
from typing import List, Tuple

from datasets import Dataset
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split

import flwr as fl
from flwr.common import Metrics

from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from src.peft import get_peft_config, get_peft_model, PeftConfig, LoraConfig, PeftModel, TaskType, PrefixTuningConfig

from transformers import DataCollatorForSeq2Seq

from transformers import AdamW
from transformers import logging
from datasets import load_dataset
from evaluate import load as load_metric
from nltk.tokenize import sent_tokenize
import evaluate
import pdb


DEVICE = torch.device("cuda")  # Try "cuda" to train on GPU
print(f"Training on {DEVICE} using PyTorch {torch.__version__} and Flower {fl.__version__}")

def init_args() :
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", default=42, type=int)
    parser.add_argument("--model_path", default="google/Flan-t5-base", type=str)
    parser.add_argument("--num_rounds", default=10, type=int)
    parser.add_argument("--num_clients", default=5, type=int)
    parser.add_argument("--visible_gpu", default= "2,3,4,5,6,7", type = str)
    parser.add_argument("--model_type", default= "PREFIX", choices = ['FT', 'PREFIX', 'PROMPT'])
    
    args = parser.parse_args()
    
    
    os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"]= args.visible_gpu

    return args



def get_model(args):
    
    CHECKPOINT = args.model_path
    
    if args.model_type == "PREFIX" :    
        peft_config = PrefixTuningConfig(task_type=TaskType.SEQ_2_SEQ_LM, 
                                        inference_mode=False, 
                                        prefix_projection = True,
                                        num_virtual_tokens=30)

        model = AutoModelForSeq2SeqLM.from_pretrained(CHECKPOINT)
        model = get_peft_model(model, peft_config)
        model.print_trainable_parameters()

    else :
        model = AutoModelForSeq2SeqLM.from_pretrained(CHECKPOINT)
    
    tokenizer = AutoTokenizer.from_pretrained(CHECKPOINT, padding=True,truncation=True)
    
    return model, tokenizer




def preprocess_function(examples):
    max_input_length = 64
    max_target_length = 30
    
    model_inputs = tokenizer(
        examples["Input"],
        max_length=max_input_length,
        truncation=True,
    )
    
    labels = tokenizer(
        examples["Label"], 
        max_length=max_target_length, 
        truncation=True
    )
    
    model_inputs["labels"] = labels["input_ids"]

    return model_inputs




def load_data(clients_num, model, tokenizer):
    
    trainloader = []
    testloader = []
    data_types = {"Input" : str, 'Label' : str}
        
    dataset_train = [Dataset.from_pandas(pd.read_csv(f"./clients/client_{i}/train.csv",dtype=data_types, usecols = ["Input", "Label"])) for i in range(clients_num)]
    dataset_test = [Dataset.from_pandas(pd.read_csv(f"./clients/client_{i}/test.csv",dtype=data_types, usecols = ["Input", "Label"])) for i in range(clients_num)]
    



    tokenized_train = [dataset.map(preprocess_function) for dataset in dataset_train]
    tokenized_test = [dataset.map(preprocess_function) for dataset in dataset_test]
        
    
    data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)    
    for dataset in tokenized_train:
        dataset = dataset.remove_columns(["Input", "Label"])
        
        loader = DataLoader(
            dataset,
            shuffle=True,
            batch_size=8,
            collate_fn=data_collator,
        )
        
        trainloader.append(loader)

        
        
    for dataset in tokenized_test:
        dataset = dataset.remove_columns(["Input", "Label"])
        loader = DataLoader(
            dataset,
            batch_size=8,
            collate_fn=data_collator,
        )
        
        testloader.append(loader)
        

    return trainloader, testloader



def train(net, trainloader, epochs):
    optimizer = AdamW(net.parameters(), lr=5e-5)
    net.train()
    losses = 0
    for _ in range(epochs):
        for batch in trainloader:
            batch = {k: v.to(DEVICE) for k, v in batch.items()}
            outputs = net(**batch)
            loss = outputs.loss
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            losses += loss.item()
        
        print(f"loss : {loss / len(trainloader)}")

def test(net, testloader):
    rouge_score = evaluate.load("rouge")
    loss = 0
    net.eval()
    score_list = []
    for batch in testloader:
        batch = {k: v.to(DEVICE) for k, v in batch.items()}
        labels = batch['labels'].cpu()
        
        with torch.no_grad():
            predictions = net.generate(**batch, max_length = 70)
        
        decoded_preds = tokenizer.batch_decode(predictions, skip_special_tokens=True)
        labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
        decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
        decoded_preds = ["\n".join(sent_tokenize(pred.strip())) for pred in decoded_preds]
        decoded_labels = ["\n".join(sent_tokenize(label.strip())) for label in decoded_labels]
        rouge_score.add_batch(predictions=decoded_preds, references=decoded_labels)       
        
        
    # Extract the median ROUGE scores
    result = rouge_score.compute()
    print(decoded_preds)
    print(decoded_labels)
    result = {key: value for key, value in result.items()}
    result = {k: round(v, 4) for k, v in result.items()}
    return result['rouge2'], result['rouge2']


class BaseClient(fl.client.NumPyClient):
    def __init__(self, net, trainloader, testloader):
        self.net = net
        self.trainloader = trainloader
        self.testloader = testloader

    def get_parameters(self, config):
        return [val.cpu().numpy() for _, val in self.net.state_dict().items()]

    def set_parameters(self, parameters):
        params_dict = zip(self.net.state_dict().keys(), parameters)
        state_dict = OrderedDict({k: torch.Tensor(v) for k, v in params_dict})
        self.net.load_state_dict(state_dict, strict=True)

    def fit(self, parameters, config):
        self.set_parameters(parameters)
        print("Training Started...")
        train(self.net, self.trainloader, epochs=1)
        print("Training Finished.")
        return self.get_parameters(config={}), len(self.trainloader), {}

    def evaluate(self, parameters, config):
        self.set_parameters(parameters)
        loss, accuracy = test(self.net, self.testloader)
        return float(loss), len(self.testloader), {"accuracy": float(accuracy), "loss": float(loss)}
 
 
   
class PrefixClient(fl.client.NumPyClient):
    def __init__(self, net, trainloader, testloader):
        self.net = net
        self.trainloader = trainloader
        self.testloader = testloader

    def get_parameters(self, config):
        return self.net.get_prompt_parameters()

    def set_parameters(self, parameters):
        self.net.set_prompt_parameters(parameters)

    def fit(self, parameters, config):
        self.set_parameters(parameters)
        print("Training Started...")
        train(self.net, self.trainloader, epochs=1)
        print("Training Finished.")
        return self.get_parameters(config={}), len(self.trainloader), {}

    def evaluate(self, parameters, config):
        self.set_parameters(parameters)
        loss, accuracy = test(self.net, self.testloader)
        return float(loss), len(self.testloader), {"accuracy": float(accuracy), "loss": float(loss)}

    




if __name__ == "__main__":
    
    args = init_args()
    NUM_CLIENTS = args.num_clients
    model, tokenizer = get_model(args)

    trainloaders, testloaders = load_data(5, model, tokenizer)
    
    def client_fn(cid: str):

    # Load model
        net = model.to(DEVICE)
        
        trainloader = trainloaders[int(cid)]
        valloader = testloaders[int(cid)]

        # Create a  single Flower client representing a single organization
        if args.model_type == 'PREFIX':
            client = PrefixClient(net, trainloader, valloader)
        else :
            client = BaseClient(net, trainloader, valloader)
        return client #chage code for baseclient
    
    # Create FedAvg strategy
    strategy = fl.server.strategy.FedAvg(
        fraction_fit=1.0,  # Sample 100% of available clients for training
        fraction_evaluate=1.0,  # Sample 100% of available clients for evaluation
        min_fit_clients=5,  # Never sample less than 5 clients for training
        min_evaluate_clients=5,  # Never sample less than 5 clients for evaluation
        min_available_clients=5,  # Wait until all 5 clients are available
    )

    # Specify client resources if you need GPU (defaults to 1 CPU and 0 GPU)
    if DEVICE.type == "cuda":
        client_resources = {"num_cpus" : 0.5,"num_gpus": 0.5}

    # Start simulation
    history = fl.simulation.start_simulation(
        client_fn=client_fn,
        num_clients=NUM_CLIENTS,
        config=fl.server.ServerConfig(num_rounds=args.num_rounds),
        strategy=strategy,
        client_resources=client_resources,
        ray_init_args={"include_dashboard": True}
    )

    print(history)



