import numpy as np

def init_first_pr(length):
    pr = np.zeros((length,1),dtype=float)
    for i in range(length):
        pr[i] = float(1) / length
    return pr

def compute_pagerankX(p,m,v):
    i = 1
    initV = v.copy()
    while(True):
        v = p * np.dot(v,m) + (1 - p) * initV
        i = i + 1
        if i >= 100:
            break
    return v


def pageRank(A, p):
    pr = init_first_pr(A.shape[0])
    return compute_pagerankX(p, A, pr.T)
