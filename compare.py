import MicroRCA as MicroRCA
import MicroInfer as MicroInfer
import numpy as np
import time

hipster_folder = 'D:/postgraduate/代码/whuMicroservice/MicroInfer/data/hipster'
sockShop_folder = 'D:/postgraduate/代码/whuMicroservice/MicroInfer/data/sock-shop'

# anomaly_dics = {
#     'cpu_hog': ['cart','checkout','currency','email','productcatalogue','recommendation','shipping'],
#     'memory_leak': ['cart','checkout','currency','payment','productcatalogue','shipping'],
#     'network_latency': ['cart','checkout','currency','productcatalogue','recommendation','shipping']
# }

anomaly_dics = {
    # 'network_latency': ['carts', 'carts-db', 'orders-db', 'catalogue-db', 'user-db', 'catalogue', 'orders', 'payment', 'shipping']
    # 'network_latency': ['carts-db', 'orders-db', 'catalogue-db', 'user-db']
    'network_latency': ['carts', 'catalogue', 'orders', 'payment', 'shipping']

}

# 计算是否命中


def hit(k, ranks, roots):
    sum_hit = 0
    hit_db = False
    for i in range(k):
        if i >= len(ranks) or hit_db:
            break
        for root in roots:
            if 'db' in root:
                if 'db' in ranks[i]:
                    sum_hit += 1
                    hit_db = True
                    break
            elif root == ranks[i]:
                sum_hit += 1
                break
    return sum_hit


def MAP(pr_1, pr_3, pr_5):
    return np.mean([pr_1, pr_3, pr_5])

def precision(k, ranks, roots):
    # precision = TP / (TP + FP)
    # False Positive(假正, FP)：将负类预测为正类数 → 误报.
    TP = 0
    hit_db = False
    for i in range(k):
        if i >= len(ranks) or hit_db:
            break
        for root in roots:
            if 'db' in root:
                if 'db' in ranks[i]:
                    TP += 1
                    hit_db = True
                    break
            elif root == ranks[i]:
                TP += 1
    FP = len(ranks) - TP
    return TP / (TP + FP)

def recall(k, ranks, roots):
    # recall = TP / (TP + FN)
    #  False Negative(假负 , FN)：将正类预测为负类数 → 漏报.
    TP = 0
    hit_db = False
    for i in range(k):
        if i >= len(ranks) or hit_db:
            break
        for root in roots:
            if 'db' in root:
                if 'db' in ranks[i]:
                    TP += 1
                    hit_db = True
                    break
            elif root == ranks[i]:
                TP += 1
    FN = len(roots) - TP
    return TP / (TP + FN)

def f1_score(precision, recall):
    # f1-score = 2 * (precision * recall) / (precision + recall)
    if(precision + recall == 0): return 0
    return 2 * (precision * recall) / (precision + recall)


def execute(dataset):
    pr_hit_1 = 0
    pr_hit_3 = 0
    pr_hit_5 = 0
    precision_list = []
    recall_list = []
    f1Score_list = []
    anomaly_type_num = 0

    for fault_type in anomaly_dics.keys():
        targets = anomaly_dics[fault_type]
        anomaly_type_num += len(targets)
        for target in targets:
            results = MicroInfer.MicroInfer(dataset, fault_type, target)
            # results = MicroInfer.MicroInfer(dataset, fault_type, target)
            roots = [target]
            pr_hit_1 += hit(1, results, roots)
            pr_hit_3 += hit(3, results, roots)
            pr_hit_5 += hit(5, results, roots)

            p = precision(5, results, roots)
            r = recall(5, results, roots)
            precision_list.append(p)
            recall_list.append(r)
            f1Score_list.append(f1_score(p, r))

    pr_1 = (pr_hit_1 / min(1, 1)) / anomaly_type_num
    pr_3 = (pr_hit_3 / min(1, 3)) / anomaly_type_num
    pr_5 = (pr_hit_5 / min(1, 5)) / anomaly_type_num

    print('PR@1: ', pr_1)
    print('PR@3: ', pr_3)
    print('PR@5: ', pr_5)
    print('hit1: ', pr_hit_1)
    print('hit3: ', pr_hit_3)
    print('hit5: ', pr_hit_5)
    print('MAP: ', MAP(pr_1, pr_3, pr_5))
    print('precision_5: ', precision_list)
    print('recall_5: ', recall_list)
    print('f1-score', f1Score_list)


print('==============================hipster==================================================')

# execute(hipster_folder)

# MicroRCA.MicroRCA(hipster_folder, 'network_latency','productcatalogue')


print('==============================sock-shop==================================================')
execute(sockShop_folder)