#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: zhuyuhan2333
"""

from sklearn import base, preprocessing
import pandas as pd


def formalize(data):
    normalized_x = preprocessing.normalize([data])
    X = normalized_x.reshape(-1, 1)
    return X


def formalizeDataFrame(data):
    normalized_x = preprocessing.normalize([data.values.squeeze()])
    X = normalized_x.reshape(-1, 1)
    return X


def normalize_dataframe(data):
    # 获取 DataFrame 的列名
    data_without_time = data.drop(['timestamp'], axis=1)

    # 对每一列进行归一化操作
    normalized_data = preprocessing.normalize(data_without_time.values, axis=0)

    # 创建新的 DataFrame，使用原始列名
    normalized_df = pd.DataFrame(normalized_data, columns=data_without_time.columns)
    normalized_df['timestamp'] = data['timestamp']

    return normalized_df


def normalize_series(data):
    # scaler = StandardScaler()
    # normalized_data = scaler.fit_transform(data.values.reshape(-1, 1))
    normalized_data = preprocessing.normalize(data.fillna(0).values.reshape(-1, 1), axis=0)
    normalized_series = pd.Series(normalized_data.flatten())
    return normalized_series


def df_time_limit(df, begin_timestamp, end_timestamp):
    begin_index = 0
    end_index = 1

    max_timestamp = df['timestamp'][df.shape[0] - 1]
    for index, row in df.iterrows():
        if row['timestamp'] >= int(begin_timestamp):
            begin_index = index
            break
    for index, row in df.iterrows():
        if index > begin_index and row['timestamp'] >= int(end_timestamp):
            end_index = index
            break
    if max_timestamp < int(end_timestamp):
        end_index = df.shape[0] + 1
    if df.loc[end_index]['timestamp'] == int(end_timestamp):
        end_index += 1
    df = df.loc[begin_index:end_index - 1]
    df = df.reset_index(drop=True)
    return df


def df_time_limit_normalization(df, begin_timestamp, end_timestamp):
    return normalize_dataframe(df_time_limit(df, begin_timestamp, end_timestamp).fillna(0))
