from cProfile import label
import matplotlib.pyplot as plt
import numpy as np

def paint_sample():
    # sample 5 10 20
    X_S = ['PR@1', 'PR@3', 'PR@5']
    y5_data = [0.468, 0.787, 0.957]
    y10_data = [0.593, 0.817, 0.919]
    y20_data = [0.527, 0.797, 0.84]
    fig,ax= plt.subplots()
    ax.plot(X_S,y5_data,"ob:",label='Sample5')            # 绘制曲线图
    ax.plot(X_S,y10_data,"sg-.",label='Sample10')
    ax.plot(X_S,y20_data,"pr--",label='Sample20')
    ax.set_xlabel("PR@K")     # X轴标签
    ax.set_ylabel("PR@K value")        # Y轴坐标标签
    ax.set_title("Root cause result with sample 5 10 20") 
    plt.rcParams.update({'font.size': 10, 'font.weight': 'bold'})
    plt.legend(loc="upper left")
    plt.savefig('sample' + ".svg",format='svg',dpi=300)

def paint_candidate():
    # candidate 5 10 20 准确率
    X_S = ['PR@1', 'PR@3', 'PR@5']
    c5_data = [0.564, 0.852, 0.927]
    c10_data = [0.593, 0.817, 0.919]
    c20_data = [0.535, 0.802, 0.911]
    fig,ax= plt.subplots()
    ax.plot(X_S,c5_data,"sb:",label='Candidate5')            # 绘制曲线图
    ax.plot(X_S,c10_data,"og-.",label='Candidate10')
    ax.plot(X_S,c20_data,"^r--",label='Candidate20')
    ax.set_xlabel("PR@K")     # X轴标签
    ax.set_ylabel("PR@K value")        # Y轴坐标标签
    ax.set_title("Root cause result with candidate 5 10 20") 
    plt.rcParams.update({'font.size': 10, 'font.weight': 'bold'})
    plt.legend(loc="upper left")
    plt.savefig('candidate' + ".svg",format='svg',dpi=300)

def paint_candidate_recall():
    # candidate 5 10 20 召回率
    X_S = ['data1', 'data2', 'data3']
    r5_data = [0.70, 0.70, 0.65]
    r10_data = [0.85, 0.90, 0.95]
    r20_data = [0.90, 0.95, 0.90]
    fig,ax= plt.subplots()
    ax.plot(X_S,r5_data,"ob:",label='Candidate5')            # 绘制曲线图
    ax.plot(X_S,r10_data,"<g-.",label='Candidate10')
    ax.plot(X_S,r20_data,">r--",label='Candidate20')
    ax.set_xlabel("Dataset")     # X轴标签
    ax.set_ylabel("Recall")        # Y轴坐标标签
    ax.set_title("Root cause recall with candidate 5 10 20") 
    plt.rcParams.update({'font.size': 6, 'font.weight': 'bold'})
    plt.legend(loc="upper left")
    plt.savefig('candidate-recall' + ".svg",format='svg',dpi=300)

def paint_bias():
    X1=[0,0.1,0.2,0.3,0.4,0.5]  # X轴坐标数据
    Y1=[0.5,0.688,0.75,0.562,0.688,0.562]                   # Y轴坐标数据
    # # plt.plot(X,Y,lable="$sin(X)$",color="red",linewidth=2)

    X2=[0,0.1,0.2,0.3,0.4,0.5]  # X轴坐标数据
    Y2=[0.312,0.438,0.625,0.375,0.562,0.562]  

    fig,ax= plt.subplots()
    ax.plot(X1,Y1,"^b:",label='Svc')            # 绘制曲线图
    ax.plot(X2,Y2,"og-.",label='Instance')

    ax.set_xlabel("Bias")     # X轴标签
    ax.set_ylabel("PR@K")        # Y轴坐标标签
    ax.set_title("Root cause result:PR@1 influenced by bias") 

    plt.rcParams.update({'font.size': 6, 'font.weight': 'bold'})
    plt.legend(loc="upper left")
    plt.savefig('bias' + ".svg",format='svg',dpi=300)

    # X3=[0,5,10,15,20]  # X轴坐标数据
    # Y3=[0,0.529,0.75,0.562,0.529]                 # Y轴坐标数据

    # X4=[0,5,10,15,20]  # X轴坐标数据
    # Y4=[0,0.353,0.625,0.5,0.438] 

def paint_instance():
    # X5=[0.01,0.02,0.03,0.04,0.05]  # X轴坐标数据
    # Y5=[0.471,0.353,0.75,0.6,0.6]                 # Y轴坐标数据

    # X6=[0.01,0.02,0.03,0.04,0.05]  # X轴坐标数据
    # Y6=[0.235,0.294,0.625,0.4,0.467] 

    # plt.xlabel("bias")     # X轴标签
    # plt.ylabel("PR@K")        # Y轴坐标标签
    # plt.title("root cause result:PR@1")      #  曲线图的标题

    # ax.plot(X1,Y1,"ob:",label='svc')            # 绘制曲线图
    # ax.plot(X2,Y2,"og:",label='instance')

    # ax.plot(X3,Y3,"ob:",label='svc')            # 绘制曲线图
    # ax.plot(X4,Y4,"og:",label='instance')

    # ax.plot(X5,Y5,"ob:",label='svc')            # 绘制曲线图
    # ax.plot(X6,Y6,"og:",label='instance')

    fig,ax= plt.subplots()
    a = ['data1','data2']
    ax.set_xlabel("Data set")
    ax.set_ylabel("PR@K")
    ax.set_title("MicroIRC instance root cause result")
    ax.set_ylim(0,1)
    # 画图，plt.bar()可以画柱状图

    bar_width = 0.25
    x_data = [1, 3]
    x1_data = [i + bar_width for i in x_data]
    x2_data = [i + bar_width for i in x1_data]
    x3_data = [i + bar_width for i in x2_data]
    x4_data = [i + bar_width for i in x3_data]
    x5_data = [i + bar_width for i in x4_data]

    # # IRC instance
    y1s_data = [0.882,0.938]
    y2s_data = [0.765,0.875]
    y3s_data = [0.529,0.625]

    ax.bar(x_data, y1s_data, width=2 * bar_width, label='MicroIRC instance PR@5')
    ax.bar(x2_data, y2s_data, width=2 * bar_width, label='MicroIRC instance PR@3')
    ax.bar(x4_data, y3s_data, width=2 * bar_width, label='MicroIRC instance PR@1')
    plt.rcParams.update({'font.size': 6, 'font.weight': 'bold'})
    plt.legend(loc="upper right")
    plt.xticks([i for i in x2_data], a, size=10)
    # for a,b in zip(x_data, y1s_data):
    #     plt.text(a - bar_width,b,b,fontsize = 8)
    # for a,b in zip(x2_data, y2s_data):
    #     plt.text(a - bar_width,b,b,fontsize = 8)
    # for a,b in zip(x4_data, y3s_data):
    #     plt.text(a - bar_width,b,b,fontsize = 8)
    plt.savefig('instance' + ".svg",format='svg',dpi=300)

def paint_time_window():
    X3=[0,5,10,15,20]  # X轴坐标数据
    Y3=[0,0.529,0.75,0.562,0.529]                 # Y轴坐标数据

    X4=[0,5,10,15,20]  # X轴坐标数据
    Y4=[0,0.353,0.625,0.5,0.438] 

    fig,ax= plt.subplots()
    ax.plot(X3,Y3,"^b:",label='svc')            # 绘制曲线图
    ax.plot(X4,Y4,"og-.",label='instance')

    ax.set_xlabel("Time window(minute)")     # X轴标签
    ax.set_ylabel("PR@K value")        # Y轴坐标标签
    ax.set_title("Root cause result:PR@1 influenced by time window") 

    plt.rcParams.update({'font.size': 6, 'font.weight': 'bold'})
    plt.legend(loc="upper left")
    plt.savefig('time_window' + ".svg",format='svg',dpi=300)

def paint_fault_tolerant():
    X5=[0.01,0.02,0.03,0.04,0.05]  # X轴坐标数据
    Y5=[0.471,0.353,0.75,0.6,0.6]                 # Y轴坐标数据

    X6=[0.01,0.02,0.03,0.04,0.05]  # X轴坐标数据
    Y6=[0.235,0.294,0.625,0.4,0.467]
    fig,ax= plt.subplots()
    ax.plot(X5,Y5,"^b:",label='Svc')            # 绘制曲线图
    ax.plot(X6,Y6,"og-.",label='Instance')

    ax.set_xlabel("Fault tolerant")     # X轴标签
    ax.set_ylabel("PR@K value")        # Y轴坐标标签
    ax.set_title("Root cause result:PR@1 influenced by fault tolerant") 

    plt.rcParams.update({'font.size': 6, 'font.weight': 'bold'})
    plt.legend(loc="upper left")
    plt.savefig('fault_tolerant' + ".svg",format='svg',dpi=300)

def paint_level():
    X7=['PR@1', 'PR@3', 'PR@5', 'PR@10']  # X轴坐标数据
    Y7=[0.557,0.75,0.833,1]                 # Y轴坐标数据

    Y8=[0.417,0.917,1,1]
    fig,ax= plt.subplots()
    ax.plot(X7,Y8,"^b:",label='Svc')            # 绘制曲线图
    ax.plot(X7,Y7,"og-.",label='Instance')

    ax.set_xlabel("PR@K")     # X轴标签
    ax.set_ylabel("PR@K value")        # Y轴坐标标签
    ax.set_title("Root cause result:PR@K with different failure level") 

    plt.rcParams.update({'font.size': 6, 'font.weight': 'bold'})
    plt.legend(loc="upper left")
    plt.savefig('level' + ".svg",format='svg',dpi=300)

def paint_failure_type():
    X9=['PR@1', 'PR@3', 'PR@5', 'PR@10']  # X轴坐标数据

    Y12=[1,1,1,1]
    Y13=[0.333,1,1,1]
    Y14=[1,1,1,1]
    Y15=[0.375,0.875,1,1]
    Y16=[0,1,1,1]
    fig,ax= plt.subplots()
    ax.plot(X9,Y12,"^b:",label='Failure type2')            # 绘制曲线图
    ax.plot(X9,Y13,"og-.",label='Failure type3')
    ax.plot(X9,Y14,"*r--",label='Failure type4')
    ax.plot(X9,Y15,"hy-",label='Failure type5')
    ax.plot(X9,Y16,"sm--",label='Failure type6')

    ax.set_xlabel("PR@K")     # X轴标签
    ax.set_ylabel("PR@K value")        # Y轴坐标标签
    ax.set_title("Root cause result:PR@K with different failure type") 

    plt.rcParams.update({'font.size': 6, 'font.weight': 'bold'})
    plt.legend(loc="lower right")
    plt.savefig('failure_type' + ".svg",format='svg',dpi=300)

def paint_old():
    # svc picture
    fig,ax= plt.subplots()
    a = ['data1','data2','data3']
    ax.set_xlabel("data set")
    ax.set_ylabel("PR@K")
    ax.set_title("MicroIRC & MicroRCA svc root cause result")
    ax.set_ylim(0,1)
    bar_width = 0.2
    x_data = [1, 3, 5]
    x1_data = [i + bar_width for i in x_data]
    x2_data = [i + bar_width for i in x1_data]
    x3_data = [i + bar_width for i in x2_data]
    x4_data = [i + bar_width for i in x3_data]
    x5_data = [i + bar_width for i in x4_data]

    # IRC svc
    y1_data = [1.0,0.912,0.875]
    y2_data = [0.882,0.75,0.812]
    y3_data = [0.6,0.6,0.5]

    # RCA svc
    y11_data = [0.8,0.9,0.875]
    y22_data = [0.75,0.8,0.7]
    y33_data = [0.6,0.5,0.375]

    ax.bar(x_data, y1_data, width=bar_width, label='MicroIRC svc PR@5', color = '#11EE96')
    ax.bar(x2_data, y2_data, width=bar_width, label='MicroIRC svc PR@3', color = '#2BD54D')
    ax.bar(x4_data, y3_data, width=bar_width, label='MicroIRC svc PR@1', color = '#4DB376')
    ax.bar(x1_data, y11_data, width=bar_width, label='MicroRCA PR@5', color = '#EE6911')
    ax.bar(x3_data, y22_data, width=bar_width, label='MicroRCA PR@3', color = '#E61A1A')
    ax.bar(x5_data, y33_data, width=bar_width, label='MicroRCA PR@1', color = '#C43C3C')
    plt.rcParams.update({'font.size': 4, 'font.weight': 'bold'})
    plt.legend(loc="upper right")
    plt.xticks([i for i in x2_data], a, size=10)

    #在ipython的交互环境中需要这句话才能显示出来
    # plt.show()
    for a,b in zip(x_data, y1_data):
        plt.text(a - bar_width / 1.5,b,b,fontsize = 6)
    for a,b in zip(x2_data, y2_data):
        plt.text(a - bar_width / 2,b,b,fontsize = 6)
    for a,b in zip(x4_data, y3_data):
        plt.text(a - bar_width / 2,b,b,fontsize = 6)
    for a,b in zip(x1_data, y11_data):
        if b == 0.875: continue
        plt.text(a - bar_width / 2,b,b,fontsize = 6)
    for a,b in zip(x3_data, y22_data):
        plt.text(a - bar_width / 2,b,b,fontsize = 6)
    for a,b in zip(x5_data, y33_data):
        plt.text(a - bar_width / 2,b,b,fontsize = 6)
    plt.savefig('svc' + ".svg",format='svg',dpi=300)

def paint_compare():
    fig,ax= plt.subplots()
    a = ['data1','data2']
    ax.set_xlabel("Data set")
    ax.set_ylabel("Acc")
    ax.set_title("MicroIRC & other methods' root cause Acc result")
    ax.set_ylim(0,1)
    bar_width = 0.2
    x_data = [1, 3]
    x1_data = [i + bar_width for i in x_data]
    x2_data = [i + bar_width for i in x1_data]
    x3_data = [i + bar_width for i in x2_data]
    x4_data = [i + bar_width for i in x3_data]
    x5_data = [i + bar_width for i in x4_data]
    x6_data = [i + bar_width for i in x5_data]

    # y data
    tbac_data = [0.5454545454545455,0.4357142857142856]
    cloud_data = [0.4090909090909091,0.47857142857142854]
    dycause_data = [0.3727272727272727,0.2785714285714285]
    monitor_rank_data = [0.49090909090909096,0.5071428571428571]
    netmedic_data = [0.6090909090909091,0.5285714285714286]
    microRCA_data = [0.7241925233000748,0.6282928882958035]
    microIRC_data = [0.7881598793363499,0.7396028991893654]

    ax.bar(x_data, microIRC_data, width=bar_width, label='MicroIRC', color = '#33CC52')
    ax.bar(x1_data, microRCA_data, width=bar_width, label='MicroRCA', color = '#7B6699')
    ax.bar(x2_data, netmedic_data, width=bar_width, label='NetMedic', color = '#B34D8A')
    ax.bar(x3_data, monitor_rank_data, width=bar_width, label='MonitorRank', color = '#1AE6E6')
    ax.bar(x4_data, tbac_data, width=bar_width, label='TBAC', color = '#E66B1A')
    ax.bar(x5_data, cloud_data, width=bar_width, label='CloudRanger', color = '#226DDD')
    ax.bar(x6_data, dycause_data, width=bar_width, label='Dycause', color = '#B38A4D')
    plt.rcParams.update({'font.size': 10, 'font.weight': 'bold'})
    plt.legend(loc="upper right")
    plt.xticks([i + bar_width / 2 for i in x2_data], a, size=10)

    #在ipython的交互环境中需要这句话才能显示出来
    # plt.show()
    # for a,b in zip(x_data, y1_data):
    #     plt.text(a - bar_width / 1.5,b,b,fontsize = 6)
    # for a,b in zip(x2_data, y2_data):
    #     plt.text(a - bar_width / 2,b,b,fontsize = 6)
    # for a,b in zip(x4_data, y3_data):
    #     plt.text(a - bar_width / 2,b,b,fontsize = 6)
    # for a,b in zip(x1_data, y11_data):
    #     if b == 0.875: continue
    #     plt.text(a - bar_width / 2,b,b,fontsize = 6)
    # for a,b in zip(x3_data, y22_data):
    #     plt.text(a - bar_width / 2,b,b,fontsize = 6)
    # for a,b in zip(x5_data, y33_data):
    #     plt.text(a - bar_width / 2,b,b,fontsize = 6)
    plt.savefig('对比实验结果' + ".svg",format='svg',dpi=300)

def paint_ablation():
    fig,ax= plt.subplots()
    a = ['data1','data2']
    ax.set_xlabel("Data set")
    ax.set_ylabel("PR@Avg")
    ax.set_title("MicroIRC ablation experiment result")
    ax.set_ylim(0,1)
    bar_width = 0.2
    x_data = [1, 3]
    x1_data = [i + bar_width for i in x_data]
    x2_data = [i + bar_width for i in x1_data]
    x3_data = [i + bar_width for i in x2_data]
    x4_data = [i + bar_width for i in x3_data]
    x5_data = [i + bar_width for i in x4_data]
    x6_data = [i + bar_width for i in x5_data]
    x7_data = [i + bar_width for i in x6_data]

    # instance_avg 1 3 5 10 svc_avg 1 3 5 10
    # data1
    # 0.643	0.834	0.9	0.95	0.625	0.75	0.825	0.906
    # 0.2	0.4	0.547	0.76	0	0.437	0.638	0.819
    # 0.443	0.434	0.353	0.19	0.625	0.313	0.187	0.087

    # data2
    # 0.455	0.606	0.691	0.818	0.4	0.511	0.574	0.727
    # 0.273	0.424	0.564	0.745	0	0.311	0.48	0.7
    # 0.182	0.182	0.127	0.073	0.4	0.2	0.094	0.027
    # y data
    data_i_1 = [0.643,0.455]
    data_i_3 = [0.834,0.606]
    data_i_5 = [0.9,0.691]
    data_i_10 = [0.95,0.818]
    data_s_1 = [0.625,0.4]
    data_s_3 = [0.75,0.511]
    data_s_5 = [0.825,0.574]
    data_s_10 = [0.906,0.727]

    data_i_1_a = [0.2,0.273]
    data_i_3_a = [0.4,0.424]
    data_i_5_a = [0.547,0.564]
    data_i_10_a = [0.76,0.745]
    data_s_1_a = [0.2,0.2]
    data_s_3_a = [0.437,0.311]
    data_s_5_a = [0.638,0.48]
    data_s_10_a = [0.819,0.7]

    ax.bar(x_data, data_i_1, width=bar_width, label='Instance_PR@Avg', color = '#11EE96')
    ax.bar(x_data, data_i_1_a, width=bar_width, label='Instance_PR@Avg_ablation', color = '#11C2EE')
    ax.bar(x1_data, data_i_3, width=bar_width, color = '#11EE96')
    ax.bar(x1_data, data_i_3_a, width=bar_width, color = '#11C2EE')
    ax.bar(x2_data, data_i_5, width=bar_width, color = '#11EE96')
    ax.bar(x2_data, data_i_5_a, width=bar_width, color = '#11C2EE')
    ax.bar(x3_data, data_i_10, width=bar_width, color = '#11EE96')
    ax.bar(x3_data, data_i_10_a, width=bar_width, color = '#11C2EE')
    ax.bar(x4_data, data_s_1, width=bar_width, label='Svc_PR@Avg', color = '#F7C709')
    ax.bar(x4_data, data_s_1_a, width=bar_width, label='Svc_PR@Avg_ablation', color = '#DD6D22')
    ax.bar(x5_data, data_s_3, width=bar_width, color = '#F7C709')
    ax.bar(x5_data, data_s_3_a, width=bar_width, color = '#DD6D22')
    ax.bar(x6_data, data_s_5, width=bar_width, color = '#F7C709')
    ax.bar(x6_data, data_s_5_a, width=bar_width, color = '#DD6D22')
    ax.bar(x7_data, data_s_10, width=bar_width, color = '#F7C709')
    ax.bar(x7_data, data_s_10_a, width=bar_width, color = '#DD6D22')
    plt.rcParams.update({'font.size': 6, 'font.weight': 'bold'})
    plt.legend(loc="upper right")
    plt.xticks([i + bar_width / 2 for i in x3_data], a, size=10)
    plt.savefig('消融实验结果' + ".svg",format='svg',dpi=300)

if __name__ == '__main__':
    paint_compare()
    paint_ablation()
    paint_instance()
    paint_candidate()
    paint_bias()
    paint_candidate_recall()
    paint_sample()
    paint_fault_tolerant()
    paint_time_window()
    paint_level()
    paint_failure_type()
    print('yes')