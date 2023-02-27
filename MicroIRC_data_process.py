#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: zhuyuhan2333
"""

from os import error
import pandas as pd

def parse_data(folder):
    metric_file_name = folder + '/' + 'metric.csv'
    metric_source_data = pd.read_csv(metric_file_name)
    # metric_source_data = metric_source_data.set_index('timestamp')

    headers = metric_source_data.columns
    data_prefix = metric_source_data.iloc[:,[0]]
    # data_suffix = metric_source_data.iloc[:,[-3, -2, -1]]

    # node data
    node_file_name = folder + '/' + 'node.csv'
    node_source_data = pd.read_csv(node_file_name)
    for head in node_source_data.columns:
        if 'node' not in head:
            node_source_data = node_source_data.drop([head], axis=1)

    data_map = {}
    
    for head in headers:
        if 'timestamp' in head: continue
        temp_name = head[0:head.find('&')]
        try:
            data = metric_source_data[head].to_frame()
            try:
                df = data_map[temp_name]
                data_map[temp_name] = df.join(data)
            except:
                data_map[temp_name] = data
        except:
            break
    for key, value in data_map.items():
        temp_prefix = data_prefix
        data = data_prefix.join(value).join(node_source_data)
        # data['timestamp'] = data['timestamp'].astype('int')
        # data = data.join(node_source_data, on='timestamp', how='left', lsuffix='', rsuffix='_right', sort=False)
        output_file_name = folder + '/' + key
        data.to_csv(output_file_name + '.csv')

if __name__ == '__main__':
    folder = './20220722'
    parse_data(folder)
    print('yes')

