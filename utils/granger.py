from statsmodels.tsa.stattools import grangercausalitytests
import pandas as pd
df = pd.read_csv('./data/svc_latency.csv').fillna(0)
labels = ['adservice','cartservice','checkoutservice','currencyservice','emailservice','productcatalogservice',
    'frontend','paymentservice','recommendationservice','shippingservice']

results = []
for Y in labels:
    for X in labels:
        if X == Y:
            continue
        else:
            flag = True
            dic = grangercausalitytests(df[[Y, X]], maxlag=2)
            for key in dic.keys():
                t = dic[key][0]
                for item in t:
                    if t[item][1] > 0.05:
                        flag = False
                        break
            if flag:
                results.append(X + '->'+ Y)

for result in results:
    print(result)
            



# 每个检验的p值都小于5%,所以可以说月份对澳大利亚药物销售的预测有用，或者说药物的销售可能存在季节性。