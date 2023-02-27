import numpy as np


# 初始化最初的pr值
def init_first_pr(length):
    pr = np.zeros((length,1),dtype=float)# 构造一个存放pr值的矩阵
    for i in range(length):
        pr[i] = float(1) / length
    return pr


# 计算PageRank值
def compute_pagerankX(p,m,v):
    i = 1
    initV = v.copy()
    while(True):
        v = p * np.dot(v,m) + (1 - p) * initV
        i = i + 1
        # 迭代100次
        if i >= 100:
            break
    return v


def pageRank(A, p):
    pr = init_first_pr(A.shape[0])
    return compute_pagerankX(p, A, pr.T)
