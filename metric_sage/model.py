import sys
import os

from torch.optim.lr_scheduler import StepLR

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
from metric_sage.Config import Config
import wandb
from util import *

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

def load_RCA_with_label(node_num, feat_num, df, time_data, time_list):
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
            ts = time_string_2_timestamp_utc(time_data['timestamp'][i])
            if time.in_time(ts, i):
                labels[count] = time.label
                feat_data[count, :] = row
                index_map_list.append(IndexMap(count, i))
                count += 1

    adj_lists = defaultdict(set)
    label_map = defaultdict(set)
    for i, label in enumerate(labels):
        label_map[label[0]].add(i)
    for s in label_map:
        for i in range(len(label_map[s])):
            for j in range(i, len(label_map[s])):
                adj_lists[list(label_map[s])[i]].add(list(label_map[s])[j])

    return feat_data, labels, adj_lists, index_map_list

def run_RCA(node_num, feat_num, time_data, time_list, train_metric, test_metric, val_metric, class_num, label_file, time_index, folder, config: Config):
    np.random.seed(1)
    random.seed(1)
    num_nodes = node_num
    feat_data, labels, adj_lists, index_map_list = load_RCA_with_label(node_num, feat_num, train_metric, time_data, time_list)

    agg1 = MeanAggregator(config, 'agg1', None, train_metric, index_map_list, feat_num, 64, cuda=config.cuda)
    enc1 = Encoder(config, 'enc1', None, 64, 32, adj_lists, agg1, train_metric, index_map_list, gcn=True, cuda=config.cuda)
    agg2 = MeanAggregator(config, 'agg2', lambda nodes, metric, is_train_index: enc1(nodes, metric, is_train_index).t(), train_metric, index_map_list, feat_num, 32, cuda=config.cuda)
    enc2 = Encoder(config, 'enc2', lambda nodes, metric, is_train_index: enc1(nodes, metric, is_train_index).t(), enc1.embed_dim, class_num, adj_lists, agg2, train_metric, index_map_list,
                   base_model=enc1, gcn=True, cuda=config.cuda)
    
    # train parameters
    epochs = config.epochs
    learning_rate = config.learning_rate

    graphsage = SupervisedGraphSage(class_num, enc2)
    if config.cuda:
        graphsage = graphsage.cuda()
    rand_indices = np.random.permutation(num_nodes)
    train = list(rand_indices)
    batch_size = config.batch_size

    val = []
    val_labels = []
    for v in val_metric:
        t = v.tt
        for i in range(t.begin_index, t.end_index + 1):
            val.append(i)
            val_labels.append(t.label)

    # model diy name
    # suffix_diy = "data_modify"
    suffix_diy = ""
    time_index_str = ""
    for ti in time_index:
        time_index_str = time_index_str + (str(ti) + ".")
    suffix = label_file + "_" + str(class_num) + "_" + str(config.rate) + "_" + str(config.num_sample) + "_" + str(epochs) + \
             "_" + str(learning_rate) + ("" if suffix_diy == "" else "_" + suffix_diy) + "_" + time_index_str
    if config.is_train:
        wandb.init(project="MicroIRC_" + suffix)
        wandb.config = {
        "learning_rate": learning_rate,
        "epochs": epochs,
        "batch_size": batch_size
        }
        optimizer = torch.optim.SGD(filter(lambda p : p.requires_grad, graphsage.parameters()), lr=learning_rate)
        scheduler = StepLR(optimizer, step_size=1, gamma=0.5)
        times = []
        epoch_size = config.epoch_size
        total_epochs = 0
        for epoch in range(epoch_size):
            print('train with epoch:' + str(epoch + 1))
            for batch in range(epochs):
                total_epochs += 1
                if total_epochs % 2000 == 0:
                    scheduler.step()
                batch_nodes = train[:batch_size]
                random.shuffle(train)
                start_time = time.time()
                optimizer.zero_grad()
                labels_v = torch.LongTensor(labels[np.array(batch_nodes)])
                if config.cuda:
                    labels_v = labels_v.cuda()
                loss = graphsage.loss(batch_nodes, train_metric.loc[index_map(batch_nodes, index_map_list)], labels_v)
                loss.backward()
                optimizer.step()
                end_time = time.time()
                times.append(end_time-start_time)
                if batch % 10 == 0:
                    print(batch, loss.data.item())
                    wandb.log({"loss": loss})
                    wandb.watch(graphsage)

        val_output = graphsage.forward(val, val_metric, False)
        print("Validation F1:", f1_score(val_labels, val_output.data.numpy().argmax(axis=1), average="micro"))
        print("Average batch time:", np.mean(times))
        _dir = folder + "/model"
        if not os.path.exists(_dir):
            os.makedirs(_dir)
        torch.save(graphsage.state_dict(), _dir + "/model_parameters_" + suffix + ".pkl")
        return graphsage
    else:
        trained_model = SupervisedGraphSage(class_num, enc2)
        trained_model.load_state_dict(torch.load(folder + "/model/model_parameters_" + suffix + ".pkl"))
        val_output = trained_model.forward(val, val_metric, False)
        print("Validation F1:", f1_score(val_labels, val_output.data.numpy().argmax(axis=1), average="micro"))
        return trained_model


if __name__ == "__main__":
    pass
