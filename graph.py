import networkx as nx
import pandas as pd
from matplotlib import pyplot as plt


'''
    构造依赖图
'''
def draw_dependency(path):
    # 读取mpg.csv文件
    df = pd.read_csv(path)
    # 构造有向图
    DG = nx.DiGraph()
    for index, row in df.iterrows():
        source = row['source']
        destination = row['destination']
        if 'rabbitmq' not in source and 'rabbitmq' not in destination and 'db' not in destination and 'db' not in source:
            DG.add_edge(source, destination)

    for node in DG.nodes():
        if 'ubuntu' in node:
            DG.nodes[node]['type'] = 'host'
        else:
            DG.nodes[node]['type'] = 'service'
    color_map = []
    for node in DG.nodes:
        if DG.nodes[node]['type'] == 'host':
            color_map.append("blue")
        else:
            color_map.append("green")

    return DG, color_map


DG, color_map = draw_dependency('./data/sock-shop/network_latency/carts-db/mpg.csv')

nx.draw(DG, node_color=color_map, with_labels=True)
plt.show()
