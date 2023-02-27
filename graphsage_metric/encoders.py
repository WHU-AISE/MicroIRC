import torch
import torch.nn as nn
from torch.nn import init
import torch.nn.functional as F

class Encoder(nn.Module):
    """
    Encodes a node's using 'convolutional' GraphSage approach
    """
    def __init__(self, features, feature_dim, 
            embed_dim, adj_lists, aggregator,
            metric,
            index_map_list,
            num_sample=10,
            base_model=None, gcn=False, cuda=False, 
            feature_transform=False): 
        super(Encoder, self).__init__()

        self.features = features
        self.feat_dim = feature_dim
        self.adj_lists = adj_lists
        self.aggregator = aggregator
        self.num_sample = num_sample
        if base_model != None:
            self.base_model = base_model

        self.gcn = gcn
        self.embed_dim = embed_dim
        self.cuda = cuda
        self.aggregator.cuda = cuda
        self.weight = nn.Parameter(
                torch.FloatTensor(embed_dim, self.feat_dim if self.gcn else 2 * self.feat_dim))
        init.xavier_uniform(self.weight)
        self.metric = metric
        self.index_map_list = index_map_list

    def forward(self, nodes, metric, is_node_train_index = True):
        """
        Generates embeddings for a batch of nodes.

        nodes     -- list of nodes
        """
        neigh_nodes = []
        for node in nodes:
            if is_node_train_index:
                neigh_nodes.append(self.adj_lists[int(node)])
            else:
                neigh_nodes.append({i for i in range(int(node) - 60, int(node) + 60)})
        # metric传neigh_nodes区间内的
        neigh_feats = self.aggregator.forward(nodes, neigh_nodes, metric,
                self.num_sample, is_node_train_index)
        try:
            # TODO metric问题
            if not self.gcn:
                if self.cuda:
                    self_feats = self.features(torch.LongTensor(nodes), metric).cuda()
                else:
                    self_feats = self.features(torch.LongTensor(nodes), metric)
                combined = torch.cat([self_feats, neigh_feats], dim=1)
            else:
                combined = neigh_feats
        except:
            if not self.gcn:
                if self.cuda:
                    self_feats = self.features(torch.from_numpy(metric.values).cuda())
                else:
                    self_feats = self.features(torch.from_numpy(metric.values))
                combined = torch.cat([self_feats, neigh_feats], dim=1)
            else:
                combined = neigh_feats
        combined = F.relu(self.weight.mm(combined.t()))
        return combined
