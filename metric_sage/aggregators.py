import torch
import torch.nn as nn
from torch.autograd import Variable

import random
import numpy as np
from metric_sage.index_map import index_map

"""
Set of modules for aggregating embeddings of neighbors.
"""

class MeanAggregator(nn.Module):
    """
    Aggregates a node's embeddings using mean of neighbors' embeddings
    """
    def __init__(self, features, metric, index_map_list,cuda=False, gcn=False): 
        """
        Initializes the aggregator for a specific graph.

        features -- function mapping LongTensor of node ids to FloatTensor of feature values.
        cuda -- whether to use GPU
        gcn --- whether to perform concatenation GraphSAGE-style, or add self-loops GCN-style
        """

        super(MeanAggregator, self).__init__()

        self.features = features
        self.cuda = cuda
        self.gcn = gcn
        self.metric = metric
        self.index_map_list = index_map_list

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
        
    def forward(self, nodes, neigh_nodes, metric, num_sample=10, is_node_train_index=True):
        """
        nodes --- list of nodes in a batch
        to_neighs --- list of sets, each set is the set of neighbors for node in batch
        num_sample --- number of neighbors to sample. No sampling if None.
        """
        # Local pointers to functions (speed hack)
        _set = set
        if not num_sample is None:
            _sample = random.sample
            samp_neighs = [_set(_sample(neigh_node, 
                            num_sample,
                            )) if len(neigh_node) >= num_sample else neigh_node for neigh_node in neigh_nodes]
        else:
            samp_neighs = neigh_nodes

        if self.gcn:
            samp_neighs = [samp_neigh + set([nodes[i]]) for i, samp_neigh in enumerate(samp_neighs)]
        unique_nodes_list = list(set.union(*samp_neighs))
        if is_node_train_index:
            true_unique_nodes_list = index_map(unique_nodes_list, self.index_map_list)
        else:
            true_unique_nodes_list = unique_nodes_list
        unique_nodes = {n:i for i,n in enumerate(true_unique_nodes_list)}
        mask = Variable(torch.zeros(len(samp_neighs), len(true_unique_nodes_list)))
        if is_node_train_index:
            column_indices = [unique_nodes[index_map([n], self.index_map_list)[0]] for samp_neigh in samp_neighs for n in samp_neigh]  
        else:
            column_indices = [unique_nodes[n] for samp_neigh in samp_neighs for n in samp_neigh]  
        row_indices = [i for i in range(len(samp_neighs)) for j in range(len(samp_neighs[i]))]
        mask[row_indices, column_indices] = 1
        if self.cuda:
            mask = mask.cuda()
        num_neigh = mask.sum(1, keepdim=True)
        mask = mask.div(num_neigh)
        try:
            if self.cuda:
                embed_matrix = self.features(torch.LongTensor(unique_nodes_list).cuda(), self.metric.loc[true_unique_nodes_list], is_node_train_index)
            else:
                embed_matrix = self.features(torch.LongTensor(unique_nodes_list), self.metric.loc[true_unique_nodes_list], is_node_train_index)
        except:
            if self.cuda:
                embed_matrix = self.features(torch.tensor(self.metric.loc[true_unique_nodes_list].values).cuda())
            else:
                # metric_feat = self.flattenlist([m.values for m in neigh_metric])
                # embed_matrix = self.features(torch.tensor(metric_feat))
                if is_node_train_index:
                    embed_matrix = torch.from_numpy(np.float32(self.metric.loc[true_unique_nodes_list].values)).view(len(true_unique_nodes_list), 146)
                else:
                    # TODO 新指标需要传入neighbor metric,无法从self.metric获取
                    embed_matrix = torch.from_numpy(np.float32(self.metric.loc[true_unique_nodes_list].values)).view(len(true_unique_nodes_list), 146)
        to_feats = mask.mm(embed_matrix)
        return to_feats
