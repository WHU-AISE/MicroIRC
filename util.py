#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: zhuyuhan2333
"""

from sklearn import base, preprocessing

def formalize(data):
    normalized_x = preprocessing.normalize([data])
    X = normalized_x.reshape(-1, 1)
    return X

def formalizeDataFrame(data):
    normalized_x = preprocessing.normalize([data.values.squeeze()])
    X = normalized_x.reshape(-1, 1)
    return X