from datetime import date
import itertools
from itertools import combinations, chain
from scipy.stats import norm, pearsonr
import pandas as pd
import numpy as np
import math

from utils import dfs, plot

def subset(iterable):
    """
    求子集: subset([1,2,3]) --> () (1,) (2,) (3,) (1,2) (1,3) (2,3) (1,2,3)
    """
    xs = list(iterable)
    # 返回 iterator 而不是 list
    return chain.from_iterable(combinations(xs, n) for n in range(len(xs) + 1))

def skeleton(suffStat, indepTest, alpha, labels, m_max):
    sepset = [[[] for i in range(len(labels))] for i in range(len(labels))]

    # 完全无向图
    G = [[True for i in range(len(labels))] for i in range(len(labels))]

    for i in range(len(labels)):
        G[i][i] = False  # 不需要检验 i -- i

    done = False  # done flag

    ord = 0
    n_edgetests = {0: 0}
    while done != True and any(G) and ord <= m_max:
        ord1 = ord + 1
        n_edgetests[ord1] = 0

        done = True

        # 相邻点对
        ind = []
        for i in range(len(G)):
            for j in range(len(G[i])):
                if G[i][j] == True:
                    ind.append((i, j))

        G1 = G.copy()

        for x, y in ind:
            if G[x][y] == True:
                neighborsBool = [row[x] for row in G1]
                neighborsBool[y] = False

                # adj(C,x) \ {y}
                neighbors = [i for i in range(len(neighborsBool)) if neighborsBool[i] == True]

                if len(neighbors) >= ord:

                    # |adj(C, x) \ {y}| > ord
                    if len(neighbors) > ord:
                        done = False

                    # |adj(C, x) \ {y}| = ord
                    for neighbors_S in set(itertools.combinations(neighbors, ord)):
                        n_edgetests[ord1] = n_edgetests[ord1] + 1

                        # 节点 x, y 是否被 neighbors_S d-seperation
                        # 条件独立性检验，返回 p-value
                        pval = indepTest(suffStat, x, y, list(neighbors_S))

                        # 条件独立
                        if pval >= alpha:
                            G[x][y] = G[y][x] = False

                            # 把 neighbors_S 加入分离集
                            sepset[x][y] = list(neighbors_S)
                            break

        ord += 1

    return {'sk': np.array(G), 'sepset': sepset}

def extend_cpdag(graph):
    def rule1(pdag, solve_conf=False, unfVect=None):
        """Rule 1: 如果存在链 a -> b - c，且 a, c 不相邻，把 b - c 变为 b -> c"""
        search_pdag = pdag.copy()
        ind = []
        for i in range(len(pdag)):
            for j in range(len(pdag)):
                if pdag[i][j] == 1 and pdag[j][i] == 0:
                    ind.append((i, j))

        #
        for a, b in sorted(ind, key=lambda x:(x[1], x[0])):
            isC = []

            for i in range(len(search_pdag)):
                if (search_pdag[b][i] == 1 and search_pdag[i][b] == 1) and (search_pdag[a][i] == 0 and search_pdag[i][a] == 0):
                    isC.append(i)

            if len(isC) > 0:
                for c in isC:
                    if 'unfTriples' in graph.keys() and ((a, b, c) in graph['unfTriples'] or (c, b, a) in graph['unfTriples']):
                        # if unfaithful, skip
                        continue
                    if pdag[b][c] == 1 and pdag[c][b] == 1:
                        pdag[b][c] = 1
                        pdag[c][b] = 0
                    elif pdag[b][c] == 0 and pdag[c][b] == 1:
                        pdag[b][c] = pdag[c][b] = 2

        return pdag

    def rule2(pdag, solve_conf=False):
        """Rule 2: 如果存在链 a -> c -> b，把 a - b 变为 a -> b"""
        search_pdag = pdag.copy()
        ind = []

        for i in range(len(pdag)):
            for j in range(len(pdag)):
                if pdag[i][j] == 1 and pdag[j][i] == 1:
                    ind.append((i, j))

        #
        for a, b in sorted(ind, key=lambda x:(x[1], x[0])):
            isC = []
            for i in range(len(search_pdag)):
                if (search_pdag[a][i] == 1 and search_pdag[i][a] == 0) and (search_pdag[i][b] == 1 and search_pdag[b][i] == 0):
                    isC.append(i)
            if len(isC) > 0:
                if pdag[a][b] == 1 and pdag[b][a] == 1:
                    pdag[a][b] = 1
                    pdag[b][a] = 0
                elif pdag[a][b] == 0 and pdag[b][a] == 1:
                    pdag[a][b] = pdag[b][a] = 2

        return pdag

    def rule3(pdag, solve_conf=False, unfVect=None):
        """Rule 3: 如果存在 a - c1 -> b 和 a - c2 -> b，且 c1, c2 不相邻，把 a - b 变为 a -> b"""
        search_pdag = pdag.copy()
        ind = []
        for i in range(len(pdag)):
            for j in range(len(pdag)):
                if pdag[i][j] == 1 and pdag[j][i] == 1:
                    ind.append((i, j))

        #
        for a, b in sorted(ind, key=lambda x:(x[1], x[0])):
            isC = []

            for i in range(len(search_pdag)):
                if (search_pdag[a][i] == 1 and search_pdag[i][a] == 1) and (search_pdag[i][b] == 1 and search_pdag[b][i] == 0):
                    isC.append(i)

            if len(isC) >= 2:
                for c1, c2 in combinations(isC, 2):
                    if search_pdag[c1][c2] == 0 and search_pdag[c2][c1] == 0:
                        # unfaithful
                        if 'unfTriples' in graph.keys() and ((c1, a, c2) in graph['unfTriples'] or (c2, a, c1) in graph['unfTriples']):
                            continue
                        if search_pdag[a][b] == 1 and search_pdag[b][a] == 1:
                            pdag[a][b] = 1
                            pdag[b][a] = 0
                            break
                        elif search_pdag[a][b] == 0 and search_pdag[b][a] == 1:
                            pdag[a][b] = pdag[b][a] = 2
                            break

        return pdag

    # Rule 4: 如果存在链 i - k -> l 和 k -> l -> j，且 k 和 l 不相邻，把 i - j 改为 i -> j
    # 显然，这种情况不可能存在，所以不需要考虑 rule4

    pdag = [[0 if graph['sk'][i][j] == False else 1 for i in range(len(graph['sk']))] for j in range(len(graph['sk']))]

    ind = []
    for i in range(len(pdag)):
        for j in range(len(pdag[i])):
            if pdag[i][j] == 1:
                ind.append((i, j))

    # 把 x - y - z 变为 x -> y <- z
    for x, y in sorted(ind, key=lambda x:(x[1],x[0])):
        allZ = []
        for z in range(len(pdag)):
            if graph['sk'][y][z] == True and z != x:
                allZ.append(z)

        for z in allZ:
            if graph['sk'][x][z] == False \
                and graph['sepset'][x][z] != None \
                and graph['sepset'][z][x] != None \
                and not (y in graph['sepset'][x][z] or y in graph['sepset'][z][x]):
                pdag[x][y] = pdag[z][y] = 1
                pdag[y][x] = pdag[y][z] = 0

    # 应用 rule1 - rule3
    pdag = rule1(pdag)
    pdag = rule2(pdag)
    pdag = rule3(pdag)

    return np.array(pdag)

def pc(suffStat, alpha, labels, indepTest, m_max=float("inf"), verbose=False):
    # 骨架
    graphDict = skeleton(suffStat, indepTest, alpha, labels, m_max)
    # 扩展为 CPDAG
    cpdag = extend_cpdag(graphDict)
    # 输出贝叶斯网络图矩阵
    if verbose:
        print(cpdag)
    return cpdag

def gauss_ci_test(suffstat, x, y, S):
    """条件独立性检验"""
    C = suffstat["C"]
    n = suffstat["n"]

    cut_at = 0.9999999

    # ------ 偏相关系数 ------
    # S中没有点
    if len(S) == 0:
        r = C[x, y]

    # S 中只有一个点，即一阶偏相关系数
    elif len(S) == 1:
        r = (C[x, y] - C[x, S] * C[y, S]) / math.sqrt((1 - math.pow(C[y, S], 2)) * (1 - math.pow(C[x, S], 2)))

    # 其实我没太明白这里是怎么求的，但 R 语言的 pcalg 包就是这样写的
    else:
        m = C[np.ix_([x]+[y] + S, [x] + [y] + S)]
        PM = np.linalg.pinv(m)
        r = -1 * PM[0, 1] / math.sqrt(abs(PM[0, 0] * PM[1, 1]))

    r = min(cut_at, max(-1 * cut_at, r))

    # Fisher’s z-transform
    res = math.sqrt(n - len(S) - 3) * .5 * math.log1p((2 * r) / (1 - r))

    # Φ^{-1}(1-α/2)
    return 2 * (1 - norm.cdf(abs(res)))


if __name__ == '__main__':
    file_path = 'data/metric/'
    image_path = 'data/result.png'

    # data = pd.DataFrame()
    # labels = ['adservice','cartservice','checkoutservice','currencyservice','emailservice','productcatalogservice',
    # 'frontend','paymentservice','recommendationservice','redis','shippingservice']

    # for label in labels:
    #     df = pd.read_csv(file_path + label + '.csv')
    #     data[label] = df['ctn_cpu']
    # data.to_csv(file_path + 'all.csv')
    data = pd.read_csv('data/svc_latency3.csv').fillna(0)
    labels = ['adservice','cartservice','checkoutservice','currencyservice','emailservice','productcatalogservice',
    'frontend','paymentservice','recommendationservice','shippingservice']

    row_count = sum(1 for row in data)
    p = pc(
        suffStat = {"C": data.corr().values, "n": data.values.shape[0]},
        alpha = 0.05,
        labels = [str(i) for i in range(row_count)],
        indepTest = gauss_ci_test,
        verbose = True
    )

    # DFS 因果关系链
    start = 2  # 起始异常节点
    vis = [0 for i in range(row_count)]
    vis[start] = True
    path = []
    path.append(start)
    dfs(p, start, path, vis)

    # 画图
    plot(p, labels, image_path)
