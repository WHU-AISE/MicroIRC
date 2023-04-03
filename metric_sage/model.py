import sys
import os
curPath = os.path.abspath(os.path.dirname(__file__))
rootPath = os.path.split(curPath)[0]
sys.path.append(rootPath)
import torch
import torch.nn as nn
from torch.nn import init
from torch.autograd import Variable

import numpy as np
import time
import random
from sklearn.metrics import f1_score
from collections import defaultdict

from metric_sage.encoders import Encoder
from metric_sage.aggregators import MeanAggregator
import pandas as pd
from metric_sage.index_map import IndexMap
from metric_sage.index_map import index_map

import wandb

"""
Simple supervised GraphSAGE model as well as examples running the model
on the Cora and Pubmed datasets.
"""

class SupervisedGraphSage(nn.Module):

    def __init__(self, num_classes, enc):
        super(SupervisedGraphSage, self).__init__()
        self.enc = enc
        self.xent = nn.CrossEntropyLoss()

        self.weight = nn.Parameter(torch.FloatTensor(num_classes, enc.embed_dim))
        init.xavier_uniform(self.weight)

    def forward(self, nodes, metric, is_node_train_index = True):
        embeds = self.enc(nodes, metric, is_node_train_index)
        scores = self.weight.mm(embeds)
        return scores.t()

    def loss(self, nodes, metric, labels, is_node_train_index = True):
        scores = self.forward(nodes, metric, is_node_train_index)
        return self.xent(scores, labels.squeeze())

def load_RCA(node_num, feat_num, df, time_data, time_list):
    num_nodes = node_num
    num_feats = feat_num
    feat_data = np.zeros((num_nodes, num_feats))
    labels = np.empty((num_nodes,1), dtype=np.int64)
    node_map = {}
    label_map = {}
    count = 0
    index_map_list = []
    for i,row in df.iterrows():
        for j, time in enumerate(time_list):
            if time.in_time(int(time_data[i:i+1]['timestamp']), i):
                labels[count] = j
                feat_data[count,:] = df[i:i+1]
                index_map_list.append(IndexMap(count, i))
                count += 1

    adj_lists = defaultdict(set)
    label_map = defaultdict(set)
    for i, label in enumerate(labels):
        label_map[label[0]].add(i)
    for s in label_map:
        for i in range(len(label_map[s])):
            for j in range(len(label_map[s])):
                adj_lists[list(label_map[s])[i]].add(list(label_map[s])[j])
    
    return feat_data, labels, adj_lists, index_map_list

def run_RCA(node_num, feat_num, df, time_data, time_list, metric, folder, class_num, label_file, train_=False):
    np.random.seed(1)
    random.seed(1)
    num_nodes = node_num
    feat_data, labels, adj_lists, index_map_list = load_RCA(node_num, feat_num, df, time_data, time_list)

    features = nn.Embedding(node_num, feat_num)

    agg1 = MeanAggregator(features, metric, index_map_list, cuda=True)
    enc1 = Encoder(features, node_num, feat_num, adj_lists, agg1, metric, index_map_list, gcn=True, cuda=False)
    agg2 = MeanAggregator(lambda nodes, metric : enc1(nodes, metric).t(), metric, index_map_list, cuda=False)
    enc2 = Encoder(lambda nodes, metric : enc1(nodes, metric).t(), enc1.embed_dim, feat_num, adj_lists, agg2, metric, index_map_list,
            base_model=enc1, gcn=True, cuda=False)
    
    # train parameters
    num_sample = 10
    batch_size = 10000
    epochs = 400
    learning_rate = 0.7
    enc1.num_sample = num_sample
    enc2.num_sample = num_sample

    graphsage = SupervisedGraphSage(class_num, enc2)
#    graphsage.cuda()
    rand_indices = np.random.permutation(num_nodes)
    test = rand_indices[:1600]
    val = rand_indices[1600:3200]
    train = list(rand_indices[3200:])
    # model diy name
    # suffix_diy = "data_modify"
    suffix_diy = ""
    suffix = label_file + "_" + str(class_num) + "_" + str(num_sample) + "_" + str(batch_size) + ("" if suffix_diy == "" else "_" + suffix_diy)
    if train_:
        wandb.init(project="MicroIRC_" + suffix)
        wandb.config = {
        "learning_rate": learning_rate,
        "epochs": epochs,
        "batch_size": batch_size
        }
        optimizer = torch.optim.SGD(filter(lambda p : p.requires_grad, graphsage.parameters()), lr=learning_rate)
        times = []
        for batch in range(batch_size):
            batch_nodes = train[:epochs]
            random.shuffle(train)
            start_time = time.time()
            optimizer.zero_grad()
            loss = graphsage.loss(batch_nodes, metric.loc[index_map(batch_nodes, index_map_list)],
                    Variable(torch.LongTensor(labels[np.array(batch_nodes)])))
            loss.backward()
            optimizer.step()
            end_time = time.time()
            times.append(end_time-start_time)
            print(batch, loss.data.item())
            
            wandb.log({"loss": loss})
            # Optional
            wandb.watch(graphsage)

        val_output = graphsage.forward(val, metric.loc[index_map(val, index_map_list)], True) 
        print("Validation F1:", f1_score(labels[val], val_output.data.numpy().argmax(axis=1), average="micro"))
        print("Average batch time:", np.mean(times))
        torch.save(graphsage.state_dict(), "./model/model_parameters_" + suffix + ".pkl")
        return graphsage
    else:
        trained_model = SupervisedGraphSage(class_num, enc2)                                                    
        trained_model.load_state_dict(torch.load("./model/model_parameters_" + suffix + ".pkl"))
        val_output = trained_model.forward(val, metric.loc[index_map(val, index_map_list)], True)
        print("Validation F1:", f1_score(labels[val], val_output.data.numpy().argmax(axis=1), average="micro"))
        return trained_model

if __name__ == "__main__":
    pass
