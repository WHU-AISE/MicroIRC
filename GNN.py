#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: zhuyuhan2333
"""

import os
from subprocess import call
import time
import pandas as pd
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt

from sklearn.cluster import Birch
from sklearn import base, preprocessing
from MicroIRC_data_process import svc_personalization
from utils.PageRank import pageRank
from graphsage.model import run_RCA
from graphsage.time import Time

import warnings
warnings.filterwarnings('ignore')


smoothing_window = 12

# Anomaly Detection

# birch异常检测
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

#            threshold = 0.05

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

# 读取异常数据
def rt_invocations(faults_name):
    # retrieve the response time of each invocation from data collection
    # input: prefix of the csv file
    # output: round-trip time

    latency_filename = faults_name + '/latency_source_50.csv'  # inbound
    latency_df_source = pd.read_csv(latency_filename)
    # latency_df_source = latency_df_source.fillna(latency_df_source.mean()).fillna(0)
    latency_df_source = latency_df_source.fillna(0)
    latency_df_source['unknown_front-end'] = 0

    latency_filename = faults_name + '/latency_destination_50.csv'  # outbound
    latency_df_destination = pd.read_csv(latency_filename)
    latency_df_destination = latency_df_destination.fillna(0)
    latency_df_destination = latency_df_destination.fillna(
        latency_df_destination.mean()).fillna(0)

    try:
        latency_df = (latency_df_source.add(latency_df_destination)).fillna(0)
        latency_df = latency_df_source.add(latency_df)
    except:
        latency_df = latency_df_source

    latency_df.set_index('timestamp')

    # latency_df.to_csv(faults_name+'/latency.csv')
    return latency_df

# 绘制调用关系图：含实例的；并返回svc instance的双向对应map
def attributed_graph(instances, call_set, root_cause):
    # build the attributed graph
    # input: prefix of the file
    # output: attributed graph

    # filename = faults_name + '/mpg.csv'
    # filename = base_folder + 'call.csv'
    # df = pd.read_csv(filename)

    DG = nx.DiGraph()
    svc_list = []
    for row in call_set:
        split = row.split('_')
        source = split[0]
        destination = split[1]
        # if 'rabbitmq' not in source and 'rabbitmq' not in destination and 'db' not in destination and 'db' not in source:
        if 'rabbitmq' not in source and 'rabbitmq' not in destination :
            if 'jaeger' not in source and 'jaeger' not in destination :
                DG.add_edge(source, destination)
                svc_list.append(source)
                svc_list.append(destination)

    # 服务列表
    # svc_list = []
    # for index, row in call_set:
    #     source = row['source']
    #     destination = row['destination']
    #     svc_list.append(source)
    #     svc_list.append(destination)
    svc_set = set(svc_list)
    svc_instances_map = {}
    instance_svc_map = {}
    # 添加实例与服务之间的边
    for svc in svc_set:
        svc_instancs = []
        for instance in instances:
            # 添加实例-节点边
            DG.add_edge(instance, 'node')
            if svc in instance:
                DG.add_edge(svc, instance)
                svc_instancs.append(instance)
                instance_svc_map.setdefault(instance, svc)
        svc_instances_map.setdefault(svc, svc_instancs)

    # node 打标签
    for node in DG.nodes():
        if 'node' in node:
            DG.nodes[node]['type'] = 'host'
        elif '-' in node and 'redis-cart' not in node:
            DG.nodes[node]['type'] = 'instance'
        else:
            DG.nodes[node]['type'] = 'service'

    # 画图并输出成图片文件
    draw(DG, "all_network" + "-" + root_cause)

    # printDGNodes(DG)
    # printDGEdges(DG)

#    plt.figure(figsize=(9,9))
#    nx.draw(DG, with_labels=True, font_weight='bold')
#    pos = nx.spring_layout(DG)
#    nx.draw(DG, pos, with_labels=True, cmap = plt.get_cmap('jet'), node_size=1500, arrows=True, )
#    labels = nx.get_edge_attributes(DG,'weight')
#    nx.draw_networkx_edge_labels(DG,pos,edge_labels=labels)
#    plt.show()

    return DG, svc_instances_map, instance_svc_map

# 画图并输出成图片文件
def draw(DG, file_name):
    # 画图并输出成图片文件
    pos = nx.spring_layout(DG)
    nx.draw(DG,
        pos, # pos 指的是布局,主要有spring_layout,random_layout,circle_layout,shell_layout
        node_color = '#B0C4DE',   # node_color指节点颜色,有rbykw,同理edge_color 
        edge_color = (0,0,0,0.5),
        font_color = 'b',
        with_labels = True, # with_labels指节点是否显示名字 
        font_size = 10,  # font_size表示字体大小,font_color表示字的颜色
        node_size = 600,
        width = 2,
        font_weight = 'bold')  # font_size表示字体大小,font_color表示字的颜色
    labels = nx.get_edge_attributes(DG,'weight')
    nx.draw_networkx_edge_labels(DG,pos,edge_labels=labels, font_size=12)
    plt.title(file_name)
    #nx.write_gexf(DG, 'network.gexf')  # gexf格式文件可以导入gephi中进行分析
    # plt.show()
    plt.savefig(file_name + ".svg",format='svg',dpi=150)

def printDGNodes(DG):
    for node in DG.nodes(data=True):
        print(node)

def printDGEdges(DG):
    for edge in DG.edges(data=True):
        print(edge)

# 计算node节点的权值
def node_weight(svc, anomaly_graph, baseline_df, faults_name, instance, begin_timestamp, end_timestamp):

    # Get the average weight of the in_edges
    in_edges_weight_avg = 0.0
    num = 0
    for u, v, data in anomaly_graph.in_edges(svc, data=True):
        #        print(u, v)
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
        # temp = abs(baseline_df[svc].corr(df[col]))
        # 每个实例的指标跟node的相关性
        temp = abs(baseline_df[instance].corr(df[col]))
        if temp > max_corr:
            max_corr = temp
            metric = col
    data = in_edges_weight_avg * max_corr
    return data, metric

def dfTimelimit(df, begin_timestamp, end_timestamp):
    begin_index = 0
    end_index = 1
    for index, row in df.iterrows():
        if row['timestamp'] >= begin_timestamp:
            begin_index = index
            break
    for index, row in df.iterrows():
        if index > begin_index and row['timestamp'] >= end_timestamp:
            end_index = index
            break
    df = df.loc[begin_index:end_index]
    return df

# 实例baseline
def getInstanceBaseline(svc, instance, baseline_df, faults_name, begin_timestamp, end_timestamp):
    filename = faults_name + '/' + instance + '.csv'
    df = pd.read_csv(filename)
    # 取滑动窗口
    df = dfTimelimit(df, begin_timestamp, end_timestamp)

    total = 0
    max = 0
    max_col = df.columns[3]
    for column in df.columns[3:-3]:
        piece = abs(baseline_df[svc].corr(df[column]))
        if piece > max:
            max = piece
            max_col = column
    return df[max_col]

# 计算实例与服务的关联度
def corrSvcAndInstances(svc, instance, baseline_df, faults_name, begin_timestamp, end_timestamp):
    filename = faults_name + '/' + instance + '.csv'
    df = pd.read_csv(filename)
    df = dfTimelimit(df, begin_timestamp, end_timestamp)
    total = 0
    max = 0.01
    for column in df.columns[3:-3]:
        piece = abs(baseline_df[svc].corr(df[column]))
        if piece > max:
            max = piece
    return max

# 计算实例与节点的关联度
def corrNodeAndInstances(instance, faults_name, begin_timestamp, end_timestamp):
    filename = faults_name + '/' + instance + '.csv'
    df = pd.read_csv(filename)
    df = dfTimelimit(df, begin_timestamp, end_timestamp)
    total = 0
    max = 0.01
    for column in df.columns[3:-3]:
        for node_column in df.columns[-3:]:
            piece = abs(df[column].corr(df[node_column]))
            if piece > max:
                max = piece
    return max

def instance_personalization(svc, anomaly_graph, baseline_df, faults_name, instance, begin_timestamp, end_timestamp):
    filename = faults_name + '/' + instance + '.csv'
    df = pd.read_csv(filename)
    df = dfTimelimit(df, begin_timestamp, end_timestamp)
    ctn_cols = df.columns[3:-3]
    max_corr = 0.01
    metric = ctn_cols[0]
    total = 0
    for col in ctn_cols:
        temp = abs(baseline_df[svc].corr(df[col]))
        # total += temp
        if temp > max_corr:
            max_corr = temp
            metric = col

    # max_corr = total / len(ctn_cols)

    # 统计svc的总值
    edges_weight_avg = 0.0
    num = 0
    for u, v, data in anomaly_graph.in_edges(svc, data=True):
        num = num + 1
        edges_weight_avg = edges_weight_avg + data['weight'] 

    svc_instance_data = 0.01
    for u, v, data in anomaly_graph.out_edges(svc, data=True):
        if anomaly_graph.nodes[v]['type'] == 'service':
            num = num + 1
            edges_weight_avg = edges_weight_avg + data['weight']
        elif v == instance:
            svc_instance_data = data['weight']

    # svc转化为instance的总值
    edges_weight_avg = edges_weight_avg * svc_instance_data

    # 统计instance自身的值，主要是跟其宿主机的关联度
    for u, v, data in anomaly_graph.in_edges(instance, data=True):
        if anomaly_graph.nodes[u]['type'] == 'host':
            num = num + 1
            edges_weight_avg = edges_weight_avg + data['weight']

    edges_weight_avg = edges_weight_avg / num

    personalization = edges_weight_avg * max_corr

    return personalization, metric

# 异常子图绘制及PageRank算法
def anomaly_subgraph(DG, anomalies, latency_df, faults_name, alpha, svc_instances_map, instance_svc_map, begin_timestamp, end_timestamp, anomalie_instances, root_cause_level, root_cause, bias, call_set):
    # 获取异常检测相关联的所有svc节点以及instance节点
    edges = []
    nodes = []
    edge_walk = []
    # print(DG.nodes())
    baseline_df = pd.DataFrame()
    edge_df = {}
    # 异常source集合
    anomaly_source = []
    source_alpha = 0.2
    # 由异常节点出发绘制异常子图
    # 异常检测出发点为svc
    for anomaly in anomalies:
        edge = anomaly.split('_')
        edge[1] = edge[1][:len(edge[1])-4]
        if edge not in edge_walk:
            edge_walk.append(edge)
        edges.append(tuple(edge))

        svc = edge[1]
        if svc == 'redis-cart' or svc == 'unknown':
            continue
        nodes.append(svc)

        # 加上anomaly_source
        source = edge[0]
        nodes.append(source)
        anomaly_source.append(source)
        baseline_df[source] = latency_df[anomaly]

        # 补充edge[0]的实例，由于调用方实例导致的延时影响
        for u, v, data in DG.out_edges(source, data=True):
            if u in v:
                nodes.append(v)
                # TODO 可以替换为实例延时数据
                baseline_df[v] = getInstanceBaseline(u, v, baseline_df, faults_name, begin_timestamp, end_timestamp)
                # edge_df[v] = anomaly

        # 以延迟为基准，供后续与它的指标进行对比
        baseline_df[svc] = latency_df[anomaly]
        # baseline_df[edge[0]] = latency_df[anomaly] * alpha
        edge_df[svc] = anomaly
        # 将instance节点也加入到子图需要处理的node中
        for u, v, data in DG.out_edges(svc, data=True):
            if u in v:
                nodes.append(v)
                baseline_df[v] = getInstanceBaseline(u, v, baseline_df, faults_name, begin_timestamp, begin_timestamp)
                edge_df[v] = anomaly

    # 异常指标基准
    baseline_df = baseline_df.fillna(0)
#    print('edge df:', edge_df)

    nodes = set(nodes)
    # 修饰异常节点svc、edge名称
    nodes = cutSvcNameForAnomalyNodes(nodes)
#    print(nodes)

    # 绘制异常子图
    anomaly_graph = nx.DiGraph()
    for node in nodes:
        # 若为实例节点，则直接跳过
        if DG.nodes[node]['type'] == 'instance':
            continue
        #        print(node)
        # 设置入边权值
        # v是异常节点
        for u, v, data in DG.in_edges(node, data=True):
            edge = (u, v)
#            print(edge)
            # 如果是异常边，直接赋值alpha
            if edge in edges:
                data = alpha
            # 如果是实例边，先跳过，由它的svc赋值时同步赋值
            elif "-" in node:
                continue
            else:
                normal_edge = u + '_' + v + '&p50'
                data = abs(baseline_df[v].corr(latency_df[normal_edge]))
                # 给instance到svc的边赋值相同
                # for node in nodes:
                #     if v in node[:len(node) - 4]:
                #         anomaly_graph.add_edge(v, node, weight=data)
                #         anomaly_graph.nodes[v]['type'] = 'instance'
            data = 0 if np.isnan(data) else data
            data = round(data, 3)
            anomaly_graph.add_edge(u, v, weight=data)
            anomaly_graph.nodes[u]['type'] = DG.nodes[u]['type']
            anomaly_graph.nodes[v]['type'] = DG.nodes[v]['type']

        # 设置出边权值
        # 使用容器资源属性设置个性化数组
        # u是异常节点
        for u, v, data in DG.out_edges(node, data=True):
            edge = (u, v)
            if edge in edges:
                data = alpha
            else:
                # 如果出点是宿主机，则根据指标赋值
                if DG.nodes[v]['type'] == 'host':
                    # 1. 找出该服务的实例的个数
                    instance_number = len(svc_instances_map[u])
                    total = 0
                    for instance in svc_instances_map[u]:
                        # 2. 计算每个实例和该node的相关性，多node需要只考虑当前node上的实例
                        piece, col = node_weight(u, anomaly_graph, baseline_df, faults_name, instance, begin_timestamp, end_timestamp)
                        pieceData = round(piece, 3)
                        if instance_svc_map[instance] in anomaly_source:
                            anomaly_graph.add_edge(instance, v, weight=pieceData)
                        else:
                            anomaly_graph.add_edge(v, instance, weight=pieceData)
                        anomaly_graph.nodes[instance]['type'] = DG.nodes[instance]['type']
                        total += piece
                    data = total / instance_number
                # instance类型节点需要添加到异常子图中去，并赋予权值
                elif DG.nodes[v]['type'] == 'instance' :
                    # 根据指标相似度赋权值
                    data = corrSvcAndInstances(u, v, baseline_df, faults_name, begin_timestamp, end_timestamp)
                    if v in anomalie_instances:
                        data += bias
                    anomaly_graph.add_edge(v, 'node', weight=corrNodeAndInstances(v, faults_name, begin_timestamp, end_timestamp))
                    anomaly_graph.nodes['node']['type'] = 'host'
                else:
                    if 'redis' in v:
                        continue
                    normal_edge = u + '_' + v
                    # 计算该节点的延时与异常节点的关联度
                    data = abs(baseline_df[u].corr(latency_df[normal_edge+"&p50"]))
            data = 0 if np.isnan(data) else data
            data = round(data, 3)
            anomaly_graph.add_edge(u, v, weight=data)
            anomaly_graph.nodes[u]['type'] = DG.nodes[u]['type']
            anomaly_graph.nodes[v]['type'] = DG.nodes[v]['type']

    print("输出异常子图edge及权重")
    for u, v, data in anomaly_graph.edges(data=True):
        print(u + "," + v + ":" + str(data['weight']))

    # 画图并输出成图片文件
    draw(anomaly_graph, "sub_network" + "-" + root_cause)

    # anomaly_graph = anomaly_graph.reverse(copy=True)
#
    # edges = list(anomaly_graph.edges(data=True))
    # train word2vec model
    # model = Word2Vec(window = 4, sg = 1, hs = 0,
    #              negative = 10, # for negative sampling
    #              alpha=0.03, min_alpha=0.0007,
    #              seed = 14)

    # for edge in call_set:
    #     if edge not in edge_walk:
    #         edge_walk.append(edge)
    # # 结合trace数据，作为random_walk基准——>word
    # model.build_vocab(edge_walk, progress_per=2)    
    # model.train(edge_walk, total_examples = model.corpus_count, epochs=20, report_delay=1)
    # for edge in edge_walk:
    #     if edge[0] == 'unknown' or edge[1] == 'unknown': continue
    #     print(edge)
    #     print(model.wv.most_similar(edge[0], topn=10))
    #     print(model.wv.most_similar(edge[1], topn=10))

    # for node in anomaly_graph.nodes:
    #     print(node + anomaly_graph.nodes[node]['type'])

    for u, v, d in edges:
        if anomaly_graph.nodes[v]['type'] == 'host' and anomaly_graph.nodes[u]['type'] != 'instance':
            anomaly_graph.remove_edge(u, v)
            # if anomaly_graph.nodes[u]['type'] == 'instance' and instance_svc_map[u] not in anomaly_source:
            #     anomaly_graph.add_edge(v, u, weight=d['weight'])
            # elif anomaly_graph.nodes[u]['type'] == 'instance' and instance_svc_map[u] in anomaly_source:
            #     anomaly_graph.add_edge(u, v, weight=d['weight'])

    print("输出PageRank异常子图edge及权重")
    for u, v, data in anomaly_graph.edges(data=True):
        print(u + "," + v + ":" + str(data['weight']))

    personalization = {}
    for node in DG.nodes():
        if node in nodes:
            personalization[node] = 0

    svc_personalization_map = {}
    svc_personalization_count = {}
    total_count = 0
    total_point = 0
    # 给个性化数组赋权值
    nodes.append('node')
    print("输出个性化数组值：")
    for node in nodes:
        # if DG.nodes[node]['type'] == 'service':
        #     personalization[node] = 0
        #     continue
        # elif DG.nodes[node]['type'] == 'host': 
        if DG.nodes[node]['type'] == 'instance':
            svc = instance_svc_map[node]
            svc_personalization_map.setdefault(svc, 0)
            svc_personalization_count.setdefault(svc, 0)
            max_corr, col = instance_personalization(
            svc, anomaly_graph, baseline_df, faults_name, node, begin_timestamp, end_timestamp)
            personalization[node] = max_corr / anomaly_graph.degree(node)
            svc_personalization_map[svc] = svc_personalization_map[svc] + personalization[node]
            svc_personalization_count[svc] = svc_personalization_count[svc] + 1
            total_count += 1
            total_point += personalization[node]

    for node in nodes:
        if DG.nodes[node]['type'] == 'service' and 'unknown' not in node:
            personalization[node] = svc_personalization_map[svc] / svc_personalization_count[svc]
        elif DG.nodes[node]['type'] == 'host':
            personalization[node] = total_point / total_count
    
    for node in nodes:
        if root_cause_level == 'pod':
            if DG.nodes[node]['type'] == 'service' or DG.nodes[node]['type'] == 'host':
                personalization[node] = 0
        if root_cause_level == 'service': 
            if DG.nodes[node]['type'] == 'instance' or DG.nodes[node]['type'] == 'host':
                personalization[node] = 0

    for node in personalization.keys():
        if np.isnan(personalization[node]):
            personalization[node] = 0

    for node in nodes:
        print(node, personalization[node])

    # PageRank算法
    try:
        anomaly_score = nx.pagerank(
            anomaly_graph, alpha=0.85, personalization=personalization, max_iter=1000)
        # anomaly_score = pageRank(
        #     anomaly_graph, personalization)
    except:
        anomaly_score = nx.pagerank(
            anomaly_graph, alpha=0.85, personalization=personalization, max_iter=1000, tol=1.0e-1)
        # anomaly_score = pageRank(
        #     anomaly_graph, personalization)

    anomaly_score = sorted(anomaly_score.items(),
                           key=lambda x: x[1], reverse=True)

    # remove_host_score(anomaly_score, anomaly_graph)

    return anomaly_score


def remove_host_score(anomaly_score, anomaly_graph):
    for node in anomaly_graph.nodes():
        if anomaly_graph.nodes[node]['type'] == 'host':
            for score in anomaly_score:
                if score[0] == node:
                    anomaly_score.remove(score)


def count_rank(anomaly_score, target, target_svc):
    num = 0
    svc_num = 0
    for idx, anomaly_target in enumerate(anomaly_score):
        if target in anomaly_target[0]:
            # num = idx + 1
            # 单node
            num = idx
            break
    for idx, anomaly_target in enumerate(anomaly_score):
        if target_svc in anomaly_target[0]:
            # svc_num = idx + 1
            # 单node
            svc_num = idx
            break
    print(target, ' Top K: ', num)
    return num, min(svc_num, num)

# def _node2vec(graph):
#     node2vec = Node2Vec(graph, dimensions=200, walk_length=30, num_walks=200, workers=1)
#     model = node2vec.fit(window=10, min_count=1, batch_words=4) 
#     # model.save('node2vec.model')
#     for node in graph.nodes():
#         print('===========' + node + '===========') 
#         print(model.wv.most_similar(node))

def print_pr(nums):
    pr1 = 0
    pr3 = 0
    pr5 = 0
    fill_nums = []
    for num in nums:
        if num != 0 and num < 10:
            fill_nums.append(num)
    for num in fill_nums:
        # if num <= 5:
        #     print(1)
        # else:
        #     print(0)
        # if num <= 3:
        #     print(1)
        # else:
        #     print(0) 
        # if num == 1:
        #     print(1)
        # else:
        #     print(0)
        if num <= 5:
            pr5 += 1
            if num <= 3:
                pr3 += 1
                if num == 1:
                    pr1 += 1
    print('PR@1:' + str(round(pr1 / len(fill_nums), 3)))
    print('PR@3:' + str(round(pr3 / len(fill_nums), 3)))
    print('PR@5:' + str(round(pr5 / len(fill_nums), 3)))


# def MicroRCA(folder, fault_type, target):
#     alpha = 0.55
#     ad_threshold = 0.045
#     faults_name = folder + '/' + fault_type + '/' + target
#     # 响应延时的服务对
#     latency_df = rt_invocations(faults_name)

#     # 进行异常检测 输出n_v对
#     # anomaly detection on response time of service invocation
#     anomalies = birch_ad_with_smoothing(latency_df, ad_threshold)

#     anomaly_nodes = []
#     for anomaly in anomalies:
#         edge = anomaly.split('_')
#         # 基于响应时间target是根因的假设
#         anomaly_nodes.append(edge[1])

#     anomaly_nodes = set(anomaly_nodes)
#     DG = attributed_graph(faults_name)

#     print(fault_type, target)
#     # 异常子图绘制及PageRank算法
#     anomaly_score = anomaly_subgraph(
#         DG, anomalies, latency_df, faults_name, alpha)
#     results = [score[0] for score in anomaly_score]
#     results = [result.replace('service', '') for result in results]
#     return results

def getInstancesName(folder):
    # metric_exact_folder = folder + '/' + data_type + '/' + concurrency + '/' + metric_folder
    # metric_file_name = metric_exact_folder + '/' + 'metric.csv'
    success_rate_file_name = folder +'/' + 'success_rate.csv'
    success_rate_source_data = pd.read_csv(success_rate_file_name)
    headers = success_rate_source_data.columns
    instances = []
    for header in headers:
        if 'timestamp' in header: continue
        instances.append(header)
    instancesSet = set(instances)
    # print(instancesSet)
    return instancesSet

# 异常检测是有p50 p90 p99等级别的数据，但是调用图中只包含svc的名称，所以需要进行修饰
def cutSvcNameForAnomalyNodes(anomaly_nodes):
    anomaly_nodes_cut = []
    for node in anomaly_nodes:
        if "&p50" in node:
            node = node[:-4]
        anomaly_nodes_cut.append(node)
    return anomaly_nodes_cut

def getRootCauseSvc(root_cause):
    if '-' not in root_cause: return root_cause
    return root_cause[:root_cause.find('-')]

if __name__ == '__main__':

    folder = './20220723'

    # params
    anomaly_count = 20
    gap = 5
    minute = 5
    alpha = 0.8
    bias = 0.2
    instance_tolerant = 0.01
    service_tolerant = 0.03

    # read root_causes
    label_file_name = folder + '/' + 'label-20220723.csv'
    label_data = pd.read_csv(label_file_name, encoding='utf-8')
    root_causes = label_data['cmdb_id']

    time_list = []

    for row in label_data.itertuples():
        root_cause = row[3]
        root_cause_level = row[2]
        real_time = '2022-7-23 ' + row[1]
        real_timestamp = int(time.mktime(time.strptime(real_time, "%Y-%m-%d %H:%M:%S")))
        begin_timestamp = real_timestamp - 60 * minute;
        end_timestamp = real_timestamp + 60 * minute;
        t = Time(begin_timestamp, end_timestamp)
        time_list.append(t)


    # build svc call
    call_file_name = folder + '/' + 'call.csv'
    call_data = pd.read_csv(call_file_name)
    call_set = []
    for head in call_data.columns:
        if 'timestamp' in head: continue
        call_set.append(head[:head.find('&')])

    nums = []
    svc_nums = []

    metric_source_data = pd.read_csv(folder + '/' + 'metric.csv')
    headers = metric_source_data.columns
    data = metric_source_data.iloc[:,2:]
    time_data = metric_source_data.iloc[:,0:1]
    for i, column in data.items():
        x = np.array(column)
        x = np.where(np.isnan(x), 0, x)
        normalized_x = preprocessing.normalize([x])

        X = normalized_x.reshape(-1, 1)
        data[i] = X

    # 暂时用时间作为节点数
    # run_RCA(int(anomaly_count * minute * 2 * 60 / gap), 147, data, time_data, time_list)
    run_RCA(2190, 147, data, time_data, time_list)
