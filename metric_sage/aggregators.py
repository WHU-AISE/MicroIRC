import torch
import torch.nn as nn
from torch.autograd import Variable

import random
import numpy as np
from metric_sage.index_map import index_map
from metric_sage.Config import Config
import torch.nn.functional as F

"""
Set of modules for aggregating embeddings of neighbors.
"""


class MeanAggregator(nn.Module):
    """
    Aggregates a node's embeddings using mean of neighbors' embeddings
    """

    def __init__(self, config: Config, name, features, metric, index_map_list, feature_num, embed_num, cuda=False,
                 gcn=False):
        """
        Initializes the aggregator for a specific graph.

        features -- function mapping LongTensor of node ids to FloatTensor of feature values.
        cuda -- whether to use GPU
        gcn --- whether to perform concatenation GraphSAGE-style, or add self-loops GCN-style
        """

        super(MeanAggregator, self).__init__()

        self.config = config
        self.name = name
        self.features = features
        self.cuda = cuda
        self.gcn = gcn
        self.metric = metric
        self.index_map_list = index_map_list
        self.embed_num = embed_num
        self.feature_num = feature_num
        self.fc1 = nn.Linear(feature_num, embed_num)

    def flattenlist(self, _2dlist):
        # defining an empty list
        flatlist = []
        # Iterating through the outer list
        for item in _2dlist:
            if type(item) is list or type(item) is np.ndarray:
                # If the item is of the list type, iterating through the sub-list
                for i in self.flattenlist(list(item)):
                    flatlist.append(i)
            else:
                flatlist.append(item)
        return flatlist

    def fc(self, x):
        relu = nn.ReLU()
        return relu(self.fc1(x))

    def forward(self, nodes, neigh_nodes, metric, is_node_train_index=True):
        """
        nodes --- list of nodes in a batch
        to_neighs --- list of sets, each set is the set of neighbors for node in batch
        num_sample --- number of neighbors to sample. No sampling if None.
        """
        # Local pointers to functions (speed hack)
        num_sample = self.config.num_sample
        _set = set
        if num_sample is not None:
            _sample = random.sample
            samp_neighs = [_set(_sample(neigh_node,
                                        num_sample,
                                        )) if len(neigh_node) >= num_sample else neigh_node for neigh_node in
                           neigh_nodes]
        else:
            samp_neighs = neigh_nodes

        if self.gcn:
            samp_neighs = [samp_neigh + set([nodes[i]]) for i, samp_neigh in enumerate(samp_neighs)]
        unique_nodes_list = list(set.union(*samp_neighs))
        if is_node_train_index:
            true_unique_nodes_list = index_map(unique_nodes_list, self.index_map_list)
        else:
            true_unique_nodes_list = unique_nodes_list
        unique_nodes = {n: i for i, n in enumerate(true_unique_nodes_list)}
        mask = Variable(torch.zeros(len(samp_neighs), len(true_unique_nodes_list)))
        if is_node_train_index:
            column_indices = [unique_nodes[index_map([n], self.index_map_list)[0]] for samp_neigh in samp_neighs for n
                              in samp_neigh]
        else:
            column_indices = [unique_nodes[n] for samp_neigh in samp_neighs for n in samp_neigh]
        row_indices = [i for i in range(len(samp_neighs)) for j in range(len(samp_neighs[i]))]
        mask[row_indices, column_indices] = 1
        if self.cuda:
            mask = mask.cuda()
        num_neigh = mask.sum(1, keepdim=True)
        mask = mask.div(num_neigh)
        mask[torch.isnan(mask)] = 0
        if self.features:
            if self.cuda:
                embed_matrix = self.features(torch.LongTensor(unique_nodes_list).cuda(), metric, is_node_train_index)
            else:
                embed_matrix = self.features(torch.LongTensor(unique_nodes_list), metric, is_node_train_index)
        else:
            if is_node_train_index:
                if self.cuda:
                    embed_matrix = self.fc(
                        torch.from_numpy(np.float32(self.metric.loc[true_unique_nodes_list].values)).cuda())
                else:
                    embed_matrix = self.fc(torch.from_numpy(np.float32(self.metric.loc[true_unique_nodes_list].values)))
            else:
                if self.cuda:
                    embed_matrix = self.fc(torch.from_numpy(np.float32(metric.loc[unique_nodes_list].values)).cuda())
                else:
                    embed_matrix = self.fc(torch.from_numpy(np.float32(metric.loc[unique_nodes_list].values)))
        to_feats = mask.mm(embed_matrix)
        return F.relu(to_feats)
