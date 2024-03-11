#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: zhuyuhan2333
"""

import os
import time
import pandas as pd
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import random

from sklearn.cluster import Birch
from util import time_string_2_timestamp
from utils.PageRank import pageRank
from metric_sage.model import run_RCA
from metric_sage.time import Time
import torch

from util import *
from metric_sage.Config import Config
import warnings

warnings.filterwarnings('ignore')

smoothing_window = 12


# Anomaly Detection

# The anomaly detection using Birch
def birch_ad_with_smoothing(latency_df, threshold):
    # anomaly detection on response time of service invocation.
    # input: response times of service invocations, threshold for birch clustering
    # output: anomalous service invocation

    anomalies = []
    for svc, latency in latency_df.iteritems():
        # No anomaly detection in db
        # if svc != 'timestamp' and 'Unnamed' not in svc and 'rabbitmq' not in svc and 'db' not in svc:
        if svc != 'timestamp' and 'Unnamed' not in svc and 'node' not in svc and 'tcp' not in svc:
            latency = latency.rolling(
                window=smoothing_window, min_periods=1).mean()
            x = np.array(latency)
            x = np.where(np.isnan(x), 0, x)
            normalized_x = preprocessing.normalize([x])

            X = normalized_x.reshape(-1, 1)
            brc = Birch(branching_factor=50, n_clusters=None,
                        threshold=threshold, compute_labels=True)
            brc.fit(X)
            brc.predict(X)

            labels = brc.labels_
            #            centroids = brc.subcluster_centers_
            n_clusters = np.unique(labels).size
            if n_clusters > 1:
                anomalies.append(svc)
    return anomalies


class Node:
    def __init__(self, name, ip, node_name, cni_ip, status, center):
        self.name = name
        self.ip = ip
        self.node_name = node_name
        self.cni_ip = cni_ip
        self.status = status
        self.center = center


def collect_graph(_dir: str):
    graphs_at_timestamp = {}
    svc_timestamp_map = {}
    pod_timestamp_map = {}
    path = os.path.join(_dir, 'graph.csv')
    k8s_nodes = [Node('izbp193ioajdcnpofhlr1hz', '172.26.146.178', 'izbp193ioajdcnpofhlr1hz', '', 'Ready', 'cloud'),
                 Node('izbp16opgy3xucvexwqp9dz', '172.23.182.14', 'izbp16opgy3xucvexwqp9dz', '', 'Ready', 'cloud'),
                 Node('izbp1gwb52uyj3g0wn52lez', '172.26.146.180', 'izbp1gwb52uyj3g0wn52lez', '', 'Ready', 'cloud'),
                 Node('izbp1gwb52uyj3g0wn52lfz', '172.26.146.179', 'izbp1gwb52uyj3g0wn52lfz', '', 'Ready', 'cloud'),
                 Node('server-1', '192.168.31.74', 'server-1', '', 'Ready', 'edge-1'),
                 Node('server-2', '192.168.31.85', 'server-2', '', 'Ready', 'edge-1'),
                 Node('server-3', '192.168.31.128', 'server-3', '', 'Ready', 'edge-2'),
                 Node('dell2018', '192.168.31.208', 'dell2018', '', 'Ready', 'edge-2')]
    graph_df = pd.read_csv(path)
    # graph_df['timestamp'] = pd.to_datetime(graph_df['timestamp'])
    grouped = graph_df.groupby('timestamp')

    def is_pod_node(row, nodes: [Node]):
        for n in nodes:
            if row['destination'] == n.node_name:
                return True
        return False

    for group_name, group_data in grouped:
        timestamp_str = str(time_string_2_timestamp(str(group_name)))
        for idx, row in group_data.iterrows():
            if is_pod_node(row, k8s_nodes):
                t_list = pod_timestamp_map.get(timestamp_str, [])
                t_list.append((row['source'], row['destination']))
                pod_timestamp_map[timestamp_str] = t_list
                config.pod_set.add(row['source'])
            else:
                t_list = svc_timestamp_map.get(timestamp_str, [])
                t_list.append((row['source'], row['destination']))
                svc_timestamp_map[timestamp_str] = t_list
                config.svc_set.add(row['source'])
                config.svc_set.add(row['destination'])

    combine_timestamp = pod_timestamp_map.copy()
    combine_timestamp.update(svc_timestamp_map.copy())
    masks = ['ingress', 'unknown', 'istio-ingressgateway']
    for timestamp in combine_timestamp:
        g = nx.DiGraph()
        svc_call_list = svc_timestamp_map.get(timestamp, None)
        svc_exist = []
        if svc_call_list:
            for svc_call in svc_call_list:
                source = svc_call[0]
                destination = svc_call[1]
                if source in masks or destination in masks:
                    continue
                g.add_edge(source, destination)
                g.nodes[source]['type'] = 'service'
                g.nodes[destination]['type'] = 'service'
                svc_exist.append(source)
                svc_exist.append(destination)
        svc_exist = list(set(svc_exist))
        pod_list = pod_timestamp_map.get(timestamp, None)
        if pod_list:
            svc_pods_map = {}
            for pod in pod_list:
                source = pod[0]
                destination = pod[1]
                if source in masks or destination in masks:
                    continue
                # pod node 双向边
                g.add_edge(source, destination)
                g.add_edge(destination, source)
                g.nodes[source]['type'] = 'instance'
                g.nodes[source]['center'] = destination
                g.nodes[destination]['type'] = 'host'
                for svc in svc_exist:
                    if svc in source:
                        svc_pods = svc_pods_map.get(svc, [])
                        svc_pods.append(source)
                        svc_pods = list(set(svc_pods))
                        svc_pods_map[svc] = svc_pods
            for svc in svc_pods_map:
                svc_pods = svc_pods_map[svc]
                for svc_pod in svc_pods:
                    g.add_edge(svc, svc_pod)
                    g.add_edge(svc_pod, svc)
        graphs_at_timestamp[timestamp] = g
    return graphs_at_timestamp


def combine_graph(graphs: [nx.DiGraph]) -> nx.DiGraph:
    g = nx.DiGraph()
    for graph in graphs:
        for edge in graph.edges:
            source = edge[0]
            destination = edge[1]
            g.add_edge(source, destination)
            g.nodes[source]['type'] = graph.nodes[source]['type']
            g.nodes[destination]['type'] = graph.nodes[destination]['type']
            try:
                g.nodes[source]['center'] = graph.nodes[source]['center']
                g.nodes[destination]['center'] = graph.nodes[destination]['center']
            except:
                pass
            try:
                g.nodes[source]['data'] = graph.nodes[source]['data']
                g.nodes[destination]['data'] = graph.nodes[destination]['data']
            except:
                pass
    return g


# draw topology graph containing instances and return the svc-instance two-way correspondence map
def attributed_graph(folder):
    graphs_at_timestamp = collect_graph(folder)
    graph = combine_graph(graphs_at_timestamp.values())
    svc_instances_map = {}
    instance_svc_map = {}
    for svc in config.svc_set:
        svc_instancs = []
        for instance in config.pod_set:
            if svc in instance:
                svc_instancs.append(instance)
                instance_svc_map.setdefault(instance, svc)
        svc_instances_map.setdefault(svc, svc_instancs)

    # draw and output file
    # draw(graph, "all_network" + "-" + root_cause)

    # printDGNodes(graph)
    # printDGEdges(graph)

    #    plt.figure(figsize=(9,9))
    #    nx.draw(graph, with_labels=True, font_weight='bold')
    #    pos = nx.spring_layout(graph)
    #    nx.draw(graph, pos, with_labels=True, cmap = plt.get_cmap('jet'), node_size=1500, arrows=True, )
    #    labels = nx.get_edge_attributes(graph,'weight')
    #    nx.draw_networkx_edge_labels(graph,pos,edge_labels=labels)
    #    plt.show()

    return graph, svc_instances_map, instance_svc_map


# draw and output file
def draw(DG, file_name):
    pos = nx.spring_layout(DG)
    nx.draw(DG,
            pos,
            node_color='#B0C4DE',
            edge_color=(0, 0, 0, 0.5),
            font_color='b',
            with_labels=True,
            font_size=10,
            node_size=600,
            width=2,
            font_weight='bold')
    labels = nx.get_edge_attributes(DG, 'weight')
    nx.draw_networkx_edge_labels(DG, pos, edge_labels=labels, font_size=12)
    plt.title(file_name)
    plt.savefig('picture/' + file_name + '.svg', format='svg', dpi=150)


def printDGNodes(DG):
    for node in DG.nodes(data=True):
        print(node)


def printDGEdges(DG):
    for edge in DG.edges(data=True):
        print(edge)


# calculate nodes' weights
def node_weight(svc, anomaly_graph, baseline_df, faults_name, instance, begin_timestamp, end_timestamp):
    # Get the average weight of the in_edges
    in_edges_weight_avg = 0.0
    num = 0
    for u, v, data in anomaly_graph.in_edges(svc, data=True):
        num = num + 1
        in_edges_weight_avg = in_edges_weight_avg + data['weight']
    if num > 0:
        in_edges_weight_avg = in_edges_weight_avg / num

    filename = faults_name + '/' + instance + '.csv'
    df = pd.read_csv(filename)
    df = dfTimelimit(df, begin_timestamp, end_timestamp)
    node_cols = df.columns[-3:]
    max_corr = 0.01
    metric = node_cols[0]
    for col in node_cols:
        # the correlation between the instance and its node
        temp = abs((pd.Series(formalize(baseline_df[instance].fillna(0)).squeeze())).corr(
            pd.Series(formalize(df[col].fillna(0)).squeeze())))
        if temp > max_corr:
            max_corr = temp
            metric = col
    data = in_edges_weight_avg * max_corr
    return data, metric


def dfTimelimit(df, begin_timestamp, end_timestamp):
    begin_index = 0
    end_index = 1
    for index, row in df.iterrows():
        if time_string_2_timestamp_utc(row['timestamp']) >= begin_timestamp:
            begin_index = index
            break
    for index, row in df.iterrows():
        if index > begin_index and time_string_2_timestamp_utc(row['timestamp']) >= end_timestamp:
            end_index = index
            break
    df = df.loc[begin_index:end_index]
    return df


# Get the instance baseline
def getInstanceBaseline(svc, instance, baseline_df, instance_data):
    max = 0
    for column in instance_data.columns:
        if instance in column:
            piece = abs((pd.Series(formalize(baseline_df[svc].fillna(0)).squeeze())).corr(
                pd.Series(formalize(instance_data[column].fillna(0)).squeeze())))
            if piece > max:
                max = piece
                max_col = column
    return instance_data[max_col]


# the correlation between the instance and its service
def corrSvcAndInstances(svc, instance, baseline_df, instance_data, svc_latency):
    max = 0
    if svc in baseline_df.columns:
        for column in instance_data.columns:
            if instance in column:
                piece = abs((pd.Series(formalize(baseline_df[svc].fillna(0)).squeeze())).corr(
                    pd.Series(formalize(instance_data[column].fillna(0)).squeeze())))
                if piece > max:
                    max = piece
    else:
        for column in svc_latency.columns:
            if svc in column:
                for instance_column in instance_data.columns:
                    if instance in instance_column:
                        piece = abs((pd.Series(formalize(svc_latency[column].fillna(0)).squeeze())).corr(
                            pd.Series(formalize(instance_data[instance_column].fillna(0)).squeeze())))
                        if piece > max:
                            max = piece
    return max


# the correlation between the instance and its node
def corrNodeAndInstances(instance, host, latency, instance_data):
    max = 0
    for column in instance_data.columns:
        if instance in instance_data:
            for node_column in latency.columns:
                if host in node_column:
                    piece = abs((pd.Series(formalize(instance_data[column].fillna(0)).squeeze())).corr(
                        pd.Series(formalize(latency[node_column].fillna(0)).squeeze())))
                    if piece > max:
                        max = piece
    return max


def corr_svcs(target_svc, edge, baseline_df, latency_df, svc_latency):
    max = 0
    if target_svc in baseline_df.columns:
        return abs(baseline_df[target_svc].corr(latency_df[edge + '&p50']))
    for column in svc_latency.columns:
        if target_svc in column:
            piece = abs((pd.Series(formalize(svc_latency[column].fillna(0)).squeeze())).corr(
                latency_df[edge + '&p50'].squeeze()))
            if piece > max:
                max = piece
    return max


def instance_personalization(svc, anomaly_graph, baseline_df, instance, instance_data, svc_latency):
    max_corr = corrSvcAndInstances(svc, instance, baseline_df, instance_data, svc_latency)

    # The total value of statistical services
    edges_weight_avg = 0.0
    num = 0
    for u, v, data in anomaly_graph.in_edges(svc, data=True):
        num = num + 1
        edges_weight_avg = edges_weight_avg + data['weight']

    svc_instance_data = 0.01
    for u, v, data in anomaly_graph.out_edges(svc, data=True):
        if v == instance:
            svc_instance_data = data['weight']

    # The total value of svc to instance conversion
    if num != 0:
        edges_weight_avg = edges_weight_avg * svc_instance_data / num + max_corr
    else:
        edges_weight_avg = max_corr
    personalization = edges_weight_avg

    return personalization, max_corr


def svc_personalization(svc, anomaly_graph):
    # The total value of statistical svc
    edges_weight_avg = 0.0
    num = 0
    for u, v, data in anomaly_graph.in_edges(svc, data=True):
        num = num + 1
        edges_weight_avg = edges_weight_avg + data['weight']

    # The total value of svc to instance conversion
    if num != 0:
        edges_weight_avg = edges_weight_avg / num
    else:
        edges_weight_avg = 1 / len(anomaly_graph.nodes)

    personalization = edges_weight_avg

    return personalization


def node_personalization(node, anomaly_graph):
    # Count the total value of instances on the node
    edges_weight_avg = 0.0
    num = 0
    for u, v, data in anomaly_graph.in_edges(node, data=True):
        num = num + 1
        edges_weight_avg = edges_weight_avg + data['weight']

        # Total value of svc to instance conversion
    edges_weight_avg = edges_weight_avg / num
    personalization = edges_weight_avg
    return personalization


# draw anomaly subgraph and execute personalized randow walk
def anomaly_subgraph(DG, anomalies, latency_df, alpha, svc_instances_map, instance_svc_map, anomalie_instances, instance_data, svc_latency):
    # Get all the svc nodes and instance nodes associated with the exception detection
    edges = []
    nodes = []
    edge_walk = []
    baseline_df = pd.DataFrame()
    edge_df = {}
    # Draw anomaly subgraphs from anomaly nodes
    for anomaly in anomalies:
        edge = anomaly.split('_')
        edge[1] = edge[1][:len(edge[1]) - 4]
        if edge not in edge_walk:
            edge_walk.append(edge)
        edges.append(tuple(edge))
        source = edge[0]
        target = edge[1]
        nodes.append(target)
        nodes.append(source)
        baseline_df[source] = latency_df[anomaly]
        baseline_df[target] = latency_df[anomaly]

        # add the edge[0], i.e, instance，latency impact due to caller instance
        for u, v, data in DG.out_edges(source, data=True):
            if u in v:
                nodes.append(v)
                if v in anomalie_instances:
                    edges.append(tuple([u, v]))
                baseline_df[v] = getInstanceBaseline(u, v, baseline_df, instance_data)

        edge_df[target] = anomaly
        # Add the called party instance node to the node to be processed in the subgraph
        for u, v, data in DG.out_edges(target, data=True):
            if u in v:
                nodes.append(v)
                if v in anomalie_instances:
                    edges.append(tuple([u, v]))
                baseline_df[v] = getInstanceBaseline(u, v, baseline_df, instance_data)
                edge_df[v] = anomaly
    # Benchmarking of abnormal metrics
    baseline_df = baseline_df.fillna(0)
    nodes.extend(anomalie_instances)
    svcs = [instance_svc_map[instance] for instance in anomalie_instances]
    nodes.extend(svcs)
    nodes = set(nodes)
    # Modify anomaly node svc, edge name
    nodes = cutSvcNameForAnomalyNodes(nodes)

    # draw anomaly subgraph
    anomaly_graph = nx.DiGraph()
    for node in nodes:
        # Skip if an instance node
        if node == 'istio-ingressgateway' or DG.nodes[node]['type'] == 'instance' or node == 'unknown' or DG.nodes[node]['type'] == 'host':
            continue
        # Set incoming edge weights
        for u, v, data in DG.in_edges(node, data=True):
            edge = (u, v)
            # If it is an abnormal edge, assign alpha directly
            if edge in edges:
                data = alpha
            # If it is an instance edge, skip it first and assign it synchronously by its svc assignment
            elif DG.nodes[u]['type'] == 'instance':
                continue
            else:
                normal_edge = u + '_' + v
                # data = abs(baseline_df[v].corr(latency_df[normal_edge]))
                data = corr_svcs(v, normal_edge, baseline_df, latency_df, svc_latency)
            data = 0 if np.isnan(data) else data
            data = round(data, 3)
            anomaly_graph.add_edge(u, v, weight=data)
            anomaly_graph.nodes[u]['type'] = DG.nodes[u]['type']
            anomaly_graph.nodes[v]['type'] = DG.nodes[v]['type']

        # Set out edge weights
        # u is the anomaly node
        for u, v, data in DG.out_edges(node, data=True):
            edge = (u, v)
            if edge in edges:
                data = alpha
                if DG.nodes[v]['type'] == 'instance':
                    instance_host = DG.nodes[v]['center']
                    anomaly_graph.add_edge(v, instance_host,
                                           weight=corrNodeAndInstances(v, instance_host, latency_df, instance_data))
                    anomaly_graph.nodes[instance_host]['type'] = 'host'
            else:
                if DG.nodes[v]['type'] == 'instance':
                    # Assign weights based on similarity of metrics
                    data = corrSvcAndInstances(u, v, baseline_df, instance_data, svc_latency)
                    instance_host = DG.nodes[v]['center']
                    anomaly_graph.add_edge(v, instance_host,
                                           weight=corrNodeAndInstances(v, instance_host, latency_df, instance_data))
                    anomaly_graph.nodes[instance_host]['type'] = 'host'
                else:
                    normal_edge = u + '_' + v
                    # Calculate the correlation between the delay of this node and the anomaly node
                    # data = abs(baseline_df[u].corr(latency_df[normal_edge + "&p50"]))
                    data = corr_svcs(u, normal_edge, baseline_df, latency_df, svc_latency)
            data = 0 if np.isnan(data) else data
            data = round(data, 3)
            anomaly_graph.add_edge(u, v, weight=data)
            anomaly_graph.nodes[u]['type'] = DG.nodes[u]['type']
            anomaly_graph.nodes[v]['type'] = DG.nodes[v]['type']

    # for u, v in edges:
    #     if anomaly_graph.nodes[v]['type'] == 'host' and anomaly_graph.nodes[u]['type'] != 'instance':
    #         anomaly_graph.remove_edge(u, v)

    personalization = {}
    for node in DG.nodes():
        if node in nodes:
            personalization[node] = 0

    svc_personalization_map = {}
    svc_personalization_count = {}
    # Assigning weights to personalized arrays
    for node in nodes:
        if node == 'unknown' or node == 'istio-ingressgateway':
            continue
        if DG.nodes[node]['type'] == 'service':
            personalization[node] = round(svc_personalization(
                node, anomaly_graph), 3)
        elif DG.nodes[node]['type'] == 'host':
            personalization[node] = round(node_personalization(
                node, anomaly_graph), 3)
        elif DG.nodes[node]['type'] == 'instance':
            target = instance_svc_map[node]
            svc_personalization_map.setdefault(target, 0)
            svc_personalization_count.setdefault(target, 0)
            p, max_corr = instance_personalization(
                target, anomaly_graph, baseline_df, node, instance_data, svc_latency)
            personalization[node] = p / anomaly_graph.degree(node)
            personalization[node] = round(p, 3)

    for node in personalization.keys():
        if np.isnan(personalization[node]):
            personalization[node] = 0

    # The personalized random walk algrithm
    try:
        anomaly_score = nx.pagerank(
            anomaly_graph, alpha=0.85, personalization=personalization, max_iter=10000)
    except:
        anomaly_score = nx.pagerank(
            anomaly_graph, alpha=0.85, personalization=personalization, max_iter=10000, tol=1.0e-1)

    anomaly_score = sorted(anomaly_score.items(),
                           key=lambda x: x[1], reverse=True)
    return anomaly_score


def remove_host_score(anomaly_score, anomaly_graph):
    for node in anomaly_graph.nodes():
        if anomaly_graph.nodes[node]['type'] == 'host':
            for score in anomaly_score:
                if score[0] == node:
                    anomaly_score.remove(score)


def count_rank(anomaly_score, target, target_svc, svc_instances_map, instance_svc_map):
    num = 0
    svc_num = 0
    for idx, anomaly_target in enumerate(anomaly_score):
        if target in anomaly_target[0]:
            num = idx + 1
            break
    for idx, anomaly_target in enumerate(anomaly_score):
        if target_svc in anomaly_target[0]:
            svc_num = idx + 1
            break
    # If the service-level anomaly
    num_relation = 0
    # if target == target_svc:
    #     instance_rank = 0
    #     instance_count = len(svc_instances_map[target])
    #     true_instance_count = 0
    #     min_rank = 0
    #     for idx, anomaly_target in enumerate(anomaly_score):
    #         if target in anomaly_target[0] and target != anomaly_target[0]:
    #             if min_rank == 0:
    #                 min_rank = (idx + 1)
    #             instance_rank += (idx + 1)
    #             true_instance_count += 1
    #     if true_instance_count / instance_count >= 0.6:
    #         num_relation = 1 if (instance_rank - 3 * true_instance_count) <= 0 else min_rank
    # If the instance-level anomaly
    # if target != target_svc:
    #     if len(svc_instances_map[instance_svc_map[target]]) == 1:
    #         for idx, anomaly_target in enumerate(anomaly_score):
    #             if anomaly_target[0] in target:
    #                 num_relation = idx + 1
    #                 break
    #
    # if num_relation != 0:
    #     num = min(num, num_relation)
    print(target, ' Top K: ', num)
    return num, svc_num


def print_pr(nums):
    pr1 = 0
    pr2 = 0
    pr3 = 0
    pr4 = 0
    pr5 = 0
    pr6 = 0
    pr7 = 0
    pr8 = 0
    pr9 = 0
    pr10 = 0
    fill_nums = []
    for num in nums:
        # if num != 0 and num < 10:
        if num != 0:
            fill_nums.append(num)
    for num in fill_nums:
        if num <= 10:
            pr10 += 1
            if num <= 9:
                pr9 += 1
                if num <= 8:
                    pr8 += 1
                    if num <= 7:
                        pr7 += 1
                        if num <= 6:
                            pr6 += 1
                            if num <= 5:
                                pr5 += 1
                                if num <= 4:
                                    pr4 += 1
                                    if num <= 3:
                                        pr3 += 1
                                        if num <= 2:
                                            pr2 += 1
                                            if num == 1:
                                                pr1 += 1
    pr_1 = round(pr1 / len(fill_nums), 3)
    pr_2 = round(pr2 / len(fill_nums), 3)
    pr_3 = round(pr3 / len(fill_nums), 3)
    pr_4 = round(pr4 / len(fill_nums), 3)
    pr_5 = round(pr5 / len(fill_nums), 3)
    pr_6 = round(pr6 / len(fill_nums), 3)
    pr_7 = round(pr7 / len(fill_nums), 3)
    pr_8 = round(pr8 / len(fill_nums), 3)
    pr_9 = round(pr9 / len(fill_nums), 3)
    pr_10 = round(pr10 / len(fill_nums), 3)
    print('PR@1:' + str(pr_1))
    print('PR@3:' + str(pr_3))
    print('PR@5:' + str(pr_5))
    print('PR@10:' + str(pr_10))
    avg_1 = pr_1
    avg_3 = round((pr_1 + pr_2 + pr_3) / 3, 3)
    avg_5 = round((pr_1 + pr_2 + pr_3 + pr_4 + pr_5) / 5, 3)
    avg_10 = round((pr_1 + pr_2 + pr_3 + pr_4 + pr_5 + pr_6 + pr_7 + pr_8 + pr_9 + pr_10) / 10, 3)
    print('AVG@1:' + str(avg_1))
    print('AVG@3:' + str(avg_3))
    print('AVG@5:' + str(avg_5))
    print('AVG@10:' + str(avg_10))
    return pr_1, pr_3, pr_5, pr_10, avg_1, avg_3, avg_5, avg_10


def my_acc(scoreList, rightOne, n=None):
    node_rank = [_[0] for _ in scoreList]
    if n is None:
        n = len(scoreList)
    s = 0.0
    for i in range(len(rightOne)):
        if rightOne[i] in node_rank:
            rank = node_rank.index(rightOne[i]) + 1
            s += (n - max(0, rank - len(rightOne))) / n
        else:
            s += 0
    s /= len(rightOne)
    return s


def getInstancesName(folder):
    success_rate_file_name = folder + '/' + 'instance.csv'
    success_rate_source_data = pd.read_csv(success_rate_file_name)
    headers = success_rate_source_data.columns
    instances = []
    for header in headers:
        if 'timestamp' in header: continue
        instances.append(header[:header.rfind('_')])
    instancesSet = set(instances)
    return instancesSet


def cutSvcNameForAnomalyNodes(anomaly_nodes):
    anomaly_nodes_cut = []
    for node in anomaly_nodes:
        if "&p" in node:
            node = node[:-4]
        anomaly_nodes_cut.append(node)
    return anomaly_nodes_cut


def getRootCauseSvc(root_cause):
    if '-' not in root_cause: return root_cause
    return root_cause[:root_cause.find('-')]


def getCandidateList(root_cause_list, count, svc_instances_map, instance_svc_map, DG):
    root_cause_candidate_list = []
    for i in range(min(count, len(root_cause_list))):
        root_cause_candidate_list.append(root_cause_list[i])
    for i in range(min(count, len(root_cause_list))):
        root_cause = root_cause_list[i]
        if DG.nodes[root_cause]['type'] == 'instance':
            # Instance root cause candidates plus services
            if instance_svc_map[root_cause] not in root_cause_candidate_list:
                root_cause_candidate_list.append(instance_svc_map[root_cause])
        elif DG.nodes[root_cause]['type'] == 'service':
            for i in svc_instances_map[root_cause]:
                if i not in root_cause_candidate_list:
                    root_cause_candidate_list.append(i)
    return root_cause_candidate_list


def trainGraphSage(time_data, time_list, folder, train_metric, test_metric, val_metric, class_num, label_file,
                   time_index, config: Config):
    node_num = 0
    for t in time_list:
        node_num += t.count

    return run_RCA(node_num, len(train_metric.columns), time_data, time_list, train_metric, test_metric, val_metric,
                   class_num, label_file, time_index,
                   folder, config)


def rank(classification_count, root_cause_list, label_map_revert):
    rank_list = {}
    for i, root_cause in enumerate(root_cause_list):
        total_value = 0
        for item in enumerate(classification_count):
            key = item[1][0]
            value = item[1][1]
            try:
                metric_root_cause = label_map_revert[key]
                # if root_cause in metric_root_cause or metric_root_cause in root_cause:
                if metric_root_cause[:metric_root_cause.index('_')] in root_cause:
                    total_value += value
                    continue
            except:
                pass
        if total_value == 0:
            rank_list.setdefault(root_cause, len(root_cause_list) - i)
        else:
            rank_list.setdefault(root_cause, (len(root_cause_list) - i) * total_value)
    return rank_list


class Simple:
    def __init__(self, label, begin, end):
        self.label = label
        self.begin = begin
        self.end = end


def read_label_logs(label_file, simple_list: [Simple], minute):
    if simple_list is None:
        simple_list = []
    file_path = label_file
    try:
        with open(file_path, 'r') as f:
            lines = f.readlines()
            # 如果文件为空则跳过
            if not lines:
                return simple_list
            for line in lines:
                if 'cpu_' in line or 'mem_' in line or 'net_' in line:
                    root_cause = line.strip()
                    simple = Simple(root_cause, None, None)
                elif 'start create' in line:
                    begin = line[:18]
                elif 'finish delete' in line:
                    end = line[:18]
                    simple.begin = timestamp_2_time_string(time_string_2_timestamp(begin) - 30 * (minute - 3))
                    simple.end = timestamp_2_time_string(time_string_2_timestamp(end) + 30 * (minute - 3))
                    simple_list.append(simple)
    except Exception as e:
        print(f"Error reading file {file_path}: {e}")


class TMetric:
    def __init__(self, tt: Time, metric: pd.DataFrame):
        self.tt = tt
        self.metric = metric


if __name__ == '__main__':

    folder_list = ['data/data3']
    label_file_list = ['topoChange']
    i_t_pr_1 = 0
    i_t_pr_3 = 0
    i_t_pr_5 = 0
    i_t_pr_10 = 0
    i_t_avg_1 = 0
    i_t_avg_3 = 0
    i_t_avg_5 = 0
    i_t_avg_10 = 0
    s_t_pr_1 = 0
    s_t_pr_3 = 0
    s_t_pr_5 = 0
    s_t_pr_10 = 0
    s_t_avg_1 = 0
    s_t_avg_3 = 0
    s_t_avg_5 = 0
    s_t_avg_10 = 0

    i_t_pr_1_a = 0
    i_t_pr_3_a = 0
    i_t_pr_5_a = 0
    i_t_pr_10_a = 0
    i_t_avg_1_a = 0
    i_t_avg_3_a = 0
    i_t_avg_5_a = 0
    i_t_avg_10_a = 0
    s_t_pr_1_a = 0
    s_t_pr_3_a = 0
    s_t_pr_5_a = 0
    s_t_pr_10_a = 0
    s_t_avg_1_a = 0
    s_t_avg_3_a = 0
    s_t_avg_5_a = 0
    s_t_avg_10_a = 0

    config = Config()
    data_count = len(folder_list)
    # params
    minute = config.minute
    alpha = config.alpha
    instance_tolerant = config.instance_tolerant
    service_tolerant = config.service_tolerant
    train = config.is_train
    candidate_count = config.candidate_count
    # rate=1 means training all anomaly types, you can set 0 < rate <= 1, e.g., {0.8, 0.6, 0.4} mentioned in paper
    rate = config.rate
    # metrics sample time interval:5s
    time_interval_minute = config.sample_interval
    node_overflow = config.node_overflow
    for i in range(data_count):
        folder = folder_list[i]
        # read root_causes
        label_file_name = folder + '/' + 'label' + '.txt'
        simple_list: [Simple] = []
        read_label_logs(label_file_name, simple_list, minute)
        label_set = set()
        label_revert_set = set()
        label_map = {}
        label_map_revert = {}
        root_cause_services = []
        for simple in simple_list:
            label_set.add(simple.label[:simple.label.index('_') + 4])
            root_cause_services.append(simple.label[:simple.label.index('_')])
        root_cause_services = list(set(root_cause_services))
        label_list = sorted(list(label_set))
        for label in list(label_set):
            label_map[label] = label_list.index(label)
            label_map_revert[label_list.index(label)] = label
        print(label_map_revert)
        class_num = len(label_set)

        label_sampling_map = {}
        for simple in simple_list:
            failure = simple.label[:simple.label.index('_') + 4]
            if failure not in label_sampling_map:
                label_sampling_map[failure] = [simple.label]
            else:
                label_sampling_map[failure].append(simple.label)
        training_sampling_map = {}
        test_sampling_map = {}
        val_sampling_map = {}
        training_labels = []
        test_labels = []
        val_labels = []
        for key in label_sampling_map:
            random.shuffle(label_sampling_map[key])
            training_labels_key = label_sampling_map[key][:int(config.train_rate * len(label_sampling_map[key]))]
            training_sampling_map[key] = training_labels_key
            for trl in training_labels_key:
                training_labels.append(trl)
            test_labels_key = label_sampling_map[key][int(config.train_rate * len(label_sampling_map[key])):int(
                (config.train_rate + config.test_rate) * len(label_sampling_map[key]))]
            test_sampling_map[key] = test_labels_key
            for tl in test_labels_key:
                test_labels.append(tl)
            val_labels_key = label_sampling_map[key][
                             int((config.train_rate + config.test_rate) * len(label_sampling_map[key])):]
            val_sampling_map[key] = val_labels_key
            for vl in val_labels_key:
                val_labels.append(vl)

        train_metric_source_data = pd.DataFrame()
        test_metric_source_data = pd.DataFrame()
        val_metric_source_data = pd.DataFrame()
        for dir in os.listdir(folder):
            if dir != 'model' and os.path.isdir(folder + '/' + dir):
                metric_data_single = pd.read_csv(folder + '/' + dir + '/bookinfo/instance.csv')
                if dir in training_labels:
                    if train_metric_source_data.empty:
                        train_metric_source_data = metric_data_single
                    else:
                        train_metric_source_data = pd.concat([train_metric_source_data, metric_data_single])
                    train_metric_source_data = train_metric_source_data.fillna(-1)
                if dir in test_labels:
                    if test_metric_source_data.empty:
                        test_metric_source_data = metric_data_single
                    else:
                        test_metric_source_data = pd.concat([test_metric_source_data, metric_data_single])
                    test_metric_source_data = test_metric_source_data.fillna(-1)
                if dir in val_labels:
                    if val_metric_source_data.empty:
                        val_metric_source_data = metric_data_single
                    else:
                        val_metric_source_data = pd.concat([val_metric_source_data, metric_data_single])
                    val_metric_source_data = val_metric_source_data.fillna(-1)
        train_metric_source_data = train_metric_source_data.sort_values(by='timestamp')
        train_metric_source_data = train_metric_source_data.reset_index(drop=True)
        test_metric_source_data = test_metric_source_data.sort_values(by='timestamp')
        test_metric_source_data = test_metric_source_data.reset_index(drop=True)
        val_metric_source_data = val_metric_source_data.sort_values(by='timestamp')
        val_metric_source_data = val_metric_source_data.reset_index(drop=True)

        # time_data
        train_time_data = train_metric_source_data.iloc[:, 0:1]
        test_time_data = test_metric_source_data.iloc[:, 0:1]
        val_time_data = val_metric_source_data.iloc[:, 0:1]


        # normalize
        def normalize(metric_source_data):
            for cc, column in metric_source_data.items():
                x = np.array(column)
                valid_rows = np.where(x != -1)

                # 仅保留不含 -1 值的行进行归一化
                valid_data = x[valid_rows]
                normalized_data = preprocessing.normalize([valid_data])

                # 将归一化后的数据还原到原始数据中
                x[valid_rows] = normalized_data[0].reshape(-1, 1).T[0]
                # normalized_x = preprocessing.normalize([x])

                metric_source_data[cc] = x
            return metric_source_data


        train_metric_data_normalize = normalize(train_metric_source_data.iloc[:, 1:])
        test_metric_data_normalize = normalize(test_metric_source_data.iloc[:, 1:])
        val_metric_data_normalize = normalize(val_metric_source_data.iloc[:, 1:])

        # combine columns
        combine_columns = list(set(test_metric_data_normalize).union(set(train_metric_data_normalize)).union(
            set(val_metric_data_normalize)))
        combine_columns.append('timestamp')
        combine_columns_index = {c: k for k, c in enumerate(combine_columns)}
        combine_columns_index_map = {k: c for k, c in enumerate(combine_columns)}
        config.combine_columns_index_map = combine_columns_index_map
        root_cause_service_2_columns = {}
        for rs in root_cause_services:
            for cic in combine_columns:
                if rs in cic:
                    if rs not in root_cause_service_2_columns:
                        root_cause_service_2_columns[rs] = [combine_columns.index(cic)]
                    else:
                        root_cause_service_2_columns[rs].append(combine_columns.index(cic))
        config.root_cause_service_2_columns = root_cause_service_2_columns


        def combine_miss_columns(data_normalize, cc):
            miss_columns = list(set(combine_columns).difference(set(data_normalize.columns)))
            for miss_column in miss_columns:
                data_normalize[miss_column] = -1
            return data_normalize.reindex(columns=cc)


        train_metric_data_normalize = combine_miss_columns(train_metric_data_normalize, combine_columns)
        test_metric_data_normalize = combine_miss_columns(test_metric_data_normalize, combine_columns)
        val_metric_data_normalize = combine_miss_columns(val_metric_data_normalize, combine_columns)


        def time_list(simples_label, time_data):
            j = 0
            tt_list = []
            for sl in simples_label:
                root_cause = sl.label[:sl.label.index('_')]
                root_cause_level = 'pod'
                begin_timestamp = time_string_2_timestamp(sl.begin)
                end_timestamp = time_string_2_timestamp(sl.end)
                failure_type = sl.label[:sl.label.index('_') + 4]
                lb = label_map[failure_type]
                t = Time(sl, begin_timestamp, end_timestamp, root_cause, root_cause_level, failure_type, lb, j + 1)
                tt_list.append(t)
                j += 1
            for ti, row in time_data.iterrows():
                for t in tt_list:
                    t.in_time(time_string_2_timestamp_utc(row['timestamp']), ti)
            return tt_list


        train_simples = [simple for simple in simple_list if simple.label in training_labels]
        test_simples = [simple for simple in simple_list if simple.label in test_labels]
        val_simples = [simple for simple in simple_list if simple.label in val_labels]
        train_time_list = time_list(train_simples, train_time_data)
        test_time_list = time_list(test_simples, test_time_data)
        val_time_list = time_list(val_simples, val_time_data)

        val_t_metrics = [TMetric(val_time, val_metric_data_normalize) for val_time in val_time_list]
        test_t_metrics = [TMetric(te_time, test_metric_data_normalize) for te_time in test_time_list]

        time_index = []

        if 0 < rate < 1:
            if train:
                random.shuffle(train_time_list)
                time_list_shuffle = train_time_list[0:int(len(train_time_list) * rate)]
                time_index = [t.index for t in time_list_shuffle]
            else:
                # input the index of model file suffix separated by "."
                # time_index = []
                time_list_shuffle = [t for t in train_time_list if t.index in time_index]
        else:
            time_list_shuffle = train_time_list

        # train GNN
        graphsage = trainGraphSage(train_time_data, time_list_shuffle, folder, train_metric_data_normalize,
                                   test_t_metrics, val_t_metrics, class_num, label_file_list[i], time_index,
                                   config)

        # ablation result
        nums_ablation = []
        svc_nums_ablation = []

        nums = []
        svc_nums = []
        instance_level_nums = []
        svc_level_nums = []
        failure_type_map = {}
        acc = 0
        acc_count = 0
        acc_ablation = 0
        acc_ablation_count = 0
        for t_metrics in test_t_metrics:
            t = t_metrics.tt
            sb = t.simple
            label_folder = folder + '/' + sb.label + '/bookinfo/'
            # build svc call
            call_file_name = label_folder + 'call.csv'
            call_data = pd.read_csv(call_file_name)
            call_set = []
            for head in call_data.columns:
                if 'timestamp' in head: continue
                call_set.append(head[:head.find('&')])

            root_cause = t.root_cause
            root_cause_level = t.root_cause_level
            begin_timestamp = t.begin
            end_timestamp = t.end
            failure_type = t.failure_type

            print('#################root_cause:' + root_cause + '#################')
            anomaly_source = root_cause
            file_dir = folder
            # collect instance names
            # instances = getInstancesName(label_folder)

            # instance.csv
            instance_file_name = label_folder + 'instance.csv'
            instance_data = pd.read_csv(instance_file_name)
            instance_data = dfTimelimit(instance_data, begin_timestamp, end_timestamp)
            anomalie_instances = [a_instance[:a_instance.rfind('_')] for a_instance in birch_ad_with_smoothing(instance_data, instance_tolerant)]
            anomalie_instances = list(set(anomalie_instances))
            # read latency data
            call_latency = pd.read_csv(label_folder + 'call.csv')

            # qps data
            # qps_file_name = label_folder + 'svc_qps.csv'
            # qps_source_data = pd.read_csv(qps_file_name)
            # qps_source_data = dfTimelimit(qps_source_data, begin_timestamp, end_timestamp)
            # anomalie_instances = birch_ad_with_smoothing(qps_source_data, service_tolerant)

            # success rate data
            # success_rate_file_name = label_folder + 'success_rate.csv'
            # success_rate_source_data = pd.read_csv(success_rate_file_name)
            # success_rate_source_data = dfTimelimit(success_rate_source_data, begin_timestamp, end_timestamp)
            # anomalie_instances += birch_ad_with_smoothing(success_rate_source_data, service_tolerant)

            # node data
            node_file_name = folder + '/' + sb.label + '/node/node.csv'
            node_source_data = pd.read_csv(node_file_name)
            for head in node_source_data.columns:
                if 'node' not in head:
                    node_source_data = node_source_data.drop([head], axis=1)

            call_latency = call_latency.join(node_source_data)

            call_latency = dfTimelimit(call_latency, begin_timestamp, end_timestamp)

            svc_latency = pd.read_csv(label_folder + 'latency.csv')
            svc_latency = dfTimelimit(svc_latency, begin_timestamp, end_timestamp)

            # anomaly detection
            anomalies = birch_ad_with_smoothing(call_latency, service_tolerant)

            anomaly_nodes = []
            for anomaly in anomalies:
                edge = anomaly.split('_')
                anomaly_nodes.append(edge[1][:edge[1].find('&')])

            anomaly_nodes = set(anomaly_nodes)
            # Build the call graph with examples for subsequent PageRank
            DG, svc_instances_map, instance_svc_map = attributed_graph(label_folder)

            # Building anomaly subgraphs and scoring with personalized PageRank
            anomaly_score = anomaly_subgraph(
                DG, anomalies, call_latency, alpha,
                svc_instances_map, instance_svc_map,
                anomalie_instances, instance_data, svc_latency)

            # ablation
            print('ablation Top K:')
            num, svc_num = count_rank(anomaly_score, root_cause, getRootCauseSvc(root_cause), svc_instances_map,
                                      instance_svc_map)
            nums_ablation.append(num)
            svc_nums_ablation.append(svc_num)
            acc_temp = my_acc(anomaly_score, [root_cause])
            if acc_temp > 0:
                acc_ablation_count += 1
            acc_ablation += my_acc(anomaly_score, [root_cause])

            root_cause_list = list(map(lambda p: p[0], anomaly_score))
            root_cause_list = getCandidateList(root_cause_list, candidate_count, svc_instances_map, instance_svc_map,
                                               DG)
            test = []
            test_labels = []
            for i in range(t.begin_index, t.end_index + 1):
                test.append(i)
                test_labels.append(t.label)
            val_output = graphsage.forward(test, test_metric_data_normalize, is_node_train_index=False)
            classification = val_output.data.cpu().numpy().argmax(axis=1)

            classification_count = {}
            for c in classification:
                try:
                    classification_count[c] += 1
                except:
                    classification_count.setdefault(c, 1)
            classification_count = sorted(classification_count.items(),
                                          key=lambda x: x[1], reverse=True)

            # Calculate the ranking results
            rank_list = rank(classification_count, root_cause_list, label_map_revert)
            rank_list = sorted(rank_list.items(),
                               key=lambda x: x[1], reverse=True)
            print('MicroIRC Top K:')
            num, svc_num = count_rank(rank_list, root_cause, getRootCauseSvc(root_cause), svc_instances_map,
                                      instance_svc_map)
            nums.append(num)
            svc_nums.append(svc_num)
            acc_temp = my_acc(rank_list, [root_cause])
            if acc_temp > 0:
                acc_count += 1
            acc += my_acc(rank_list, [root_cause])
            try:
                failure_type_nums = failure_type_map[failure_type]
            except:
                failure_type_nums = []
            if root_cause_level == 'pod':
                instance_level_nums.append(num)
                failure_type_nums.append(num)
            elif root_cause_level == 'service':
                svc_level_nums.append(svc_num)
                failure_type_nums.append(svc_num)
            failure_type_map[failure_type] = failure_type_nums

        print('exception level: ' + root_cause_level)
        print('params: ')
        print('minute: ' + str(minute))
        print('alpha: ' + str(alpha))
        print('service_tolerant: ' + str(service_tolerant))
        print('instance_tolerant: ' + str(instance_tolerant))
        print('acc: ' + str(acc / acc_count))
        print('acc_ablation: ' + str(acc_ablation / acc_ablation_count))
        print('instance_pr: ')
        i_pr_1, i_pr_3, i_pr_5, i_pr_10, i_avg_1, i_avg_3, i_avg_5, i_avg_10 = print_pr(nums)
        i_t_pr_1 += i_pr_1
        i_t_pr_3 += i_pr_3
        i_t_pr_5 += i_pr_5
        i_t_pr_10 += i_pr_10
        i_t_avg_1 += i_avg_1
        i_t_avg_3 += i_avg_3
        i_t_avg_5 += i_avg_5
        i_t_avg_10 += i_avg_10

        # print('svc_pr:')
        # s_pr_1, s_pr_3, s_pr_5, s_pr_10, s_avg_1, s_avg_3, s_avg_5, s_avg_10 = print_pr(svc_nums)
        # s_t_pr_1 += s_pr_1
        # s_t_pr_3 += s_pr_3
        # s_t_pr_5 += s_pr_5
        # s_t_pr_10 += s_pr_10
        # s_t_avg_1 += s_avg_1
        # s_t_avg_3 += s_avg_3
        # s_t_avg_5 += s_avg_5
        # s_t_avg_10 += s_avg_10

        # ablation
        print('instance_pr_ablation:')
        i_pr_1, i_pr_3, i_pr_5, i_pr_10, i_avg_1, i_avg_3, i_avg_5, i_avg_10 = print_pr(nums_ablation)
        i_t_pr_1_a += i_pr_1
        i_t_pr_3_a += i_pr_3
        i_t_pr_5_a += i_pr_5
        i_t_pr_10_a += i_pr_10
        i_t_avg_1_a += i_avg_1
        i_t_avg_3_a += i_avg_3
        i_t_avg_5_a += i_avg_5
        i_t_avg_10_a += i_avg_10

        # print('svc_pr_ablation:')
        # s_pr_1, s_pr_3, s_pr_5, s_pr_10, s_avg_1, s_avg_3, s_avg_5, s_avg_10 = print_pr(svc_nums_ablation)
        # s_t_pr_1_a += s_pr_1
        # s_t_pr_3_a += s_pr_3
        # s_t_pr_5_a += s_pr_5
        # s_t_pr_10_a += s_pr_10
        # s_t_avg_1_a += s_avg_1
        # s_t_avg_3_a += s_avg_3
        # s_t_avg_5_a += s_avg_5
        # s_t_avg_10_a += s_avg_10

        # PR@K in different levels
        print('level_instance_pr:')
        l_i_pr_1, l_i_pr_3, l_i_pr_5, l_i_pr_10, l_i_avg_1, l_i_avg_3, l_i_avg_5, l_i_avg_10 = print_pr(
            instance_level_nums)
        print('level_svc_pr:')
        # l_s_pr_1, l_s_pr_3, l_s_pr_5, l_s_pr_10, l_s_avg_1, l_s_avg_3, l_s_avg_5, l_s_avg_10 = print_pr(svc_level_nums)

        # PR@K in different anomaly types
        for key in failure_type_map:
            print('failure_type:' + str(key))
            print_pr(failure_type_map[key])

    print('instance_pr_total:')
    print('i_t_pr_1:' + str(round(i_t_pr_1 / data_count, 3)))
    print('i_t_pr_3:' + str(round(i_t_pr_3 / data_count, 3)))
    print('i_t_pr_5:' + str(round(i_t_pr_5 / data_count, 3)))
    print('i_t_pr_10:' + str(round(i_t_pr_10 / data_count, 3)))
    print('i_t_avg_1:' + str(round(i_t_avg_1 / data_count, 3)))
    print('i_t_avg_3:' + str(round(i_t_avg_3 / data_count, 3)))
    print('i_t_avg_5:' + str(round(i_t_avg_5 / data_count, 3)))
    print('i_t_avg_10:' + str(round(i_t_avg_10 / data_count, 3)))
    # print('svc_pr_total:')
    # print('s_t_pr_1:' + str(round(s_t_pr_1 / data_count, 3)))
    # print('s_t_pr_3:' + str(round(s_t_pr_3 / data_count, 3)))
    # print('s_t_pr_5:' + str(round(s_t_pr_5 / data_count, 3)))
    # print('s_t_pr_10:' + str(round(s_t_pr_10 / data_count, 3)))
    # print('s_t_avg_1:' + str(round(s_t_avg_1 / data_count, 3)))
    # print('s_t_avg_3:' + str(round(s_t_avg_3 / data_count, 3)))
    # print('s_t_avg_5:' + str(round(s_t_avg_5 / data_count, 3)))
    # print('s_t_avg_10:' + str(round(s_t_avg_10 / data_count, 3)))

    print('instance_pr_ablation_total:')
    print('i_t_pr_1_a:' + str(round(i_t_pr_1_a / data_count, 3)))
    print('i_t_pr_3_a:' + str(round(i_t_pr_3_a / data_count, 3)))
    print('i_t_pr_5_a:' + str(round(i_t_pr_5_a / data_count, 3)))
    print('i_t_pr_10_a:' + str(round(i_t_pr_10_a / data_count, 3)))
    print('i_t_avg_1_a:' + str(round(i_t_avg_1_a / data_count, 3)))
    print('i_t_avg_3_a:' + str(round(i_t_avg_3_a / data_count, 3)))
    print('i_t_avg_5_a:' + str(round(i_t_avg_5_a / data_count, 3)))
    print('i_t_avg_10_a:' + str(round(i_t_avg_10_a / data_count, 3)))
    # print('svc_pr_ablation_total:')
    # print('s_t_pr_1_a:' + str(round(s_t_pr_1_a / data_count, 3)))
    # print('s_t_pr_3_a:' + str(round(s_t_pr_3_a / data_count, 3)))
    # print('s_t_pr_5_a:' + str(round(s_t_pr_5_a / data_count, 3)))
    # print('s_t_pr_10_a:' + str(round(s_t_pr_10_a / data_count, 3)))
    # print('s_t_avg_1_a:' + str(round(s_t_avg_1_a / data_count, 3)))
    # print('s_t_avg_3_a:' + str(round(s_t_avg_3_a / data_count, 3)))
    # print('s_t_avg_5_a:' + str(round(s_t_avg_5_a / data_count, 3)))
    # print('s_t_avg_10_a:' + str(round(s_t_avg_10_a / data_count, 3)))
