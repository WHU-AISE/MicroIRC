import torch
import torch.nn as nn
from metric_sage.Config import Config

"""
Set of modules for pooling embeddings of instances to their service.
"""


class InstancesPooling(nn.Module):
    def __init__(self, config: Config, name, node_num, embed_num, cuda=False):
        super(InstancesPooling, self).__init__()

        self.config = config
        self.root_cause_service_2_columns = config.root_cause_service_2_columns
        self.name = name
        self.cuda = cuda
        self.service_count = len(self.root_cause_service_2_columns)
        self.node_num = node_num
        self.service_pooling_map = {}
        self.fc1 = nn.Linear(self.service_count, embed_num)
        for service in self.root_cause_service_2_columns:
            service_pooling = nn.MaxPool2d(kernel_size=2, stride=len(self.root_cause_service_2_columns[service]))
            self.service_pooling_map[service] = service_pooling

    def fc(self, x):
        return self.fc1(x)

    def forward(self, metric):
        tensor_index_map = self.config.combine_columns_index_map
        all_services_metric = torch.empty(metric.shape[0], self.service_count)
        if self.cuda:
            all_services_metric = all_services_metric.to('cuda')
        for i, service in enumerate(self.root_cause_service_2_columns):
            service_instances_metric = torch.rand(metric.shape[0], len(self.root_cause_service_2_columns[service]))
            if self.cuda:
                service_instances_metric = service_instances_metric.to('cuda')
            instance_metrics_count = 0
            for column in tensor_index_map:
                instance = tensor_index_map[column]
                if service in instance:
                    service_instances_metric[:, instance_metrics_count] = metric[:, column]
                    instance_metrics_count += 1
            max_instance_pooling, _ = torch.max(service_instances_metric, dim=1)
            all_services_metric[:, i] = max_instance_pooling
        return self.fc(all_services_metric)
