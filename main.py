import csv
import json
import math
import numpy as np
import pandas as pd
import scipy.stats as stats
from scipy.stats import pearsonr

# region network of northern Italy
netDict = {
    1: [2, 3 ,4, 6, 11,12 ,14 ,15],
    2: [1, 4, 5, 11],
    3: [1 ,7, 8, 9 ,10 ,11 ,15],
    4: [1, 2, 5 ,6 ,12],
    5: [2, 4 ,6, 7 ,11 ,13],
    6: [1, 4, 5, 10, 12 ,13 ,14],
    7: [3 ,5 ,10 ,11 ,13],
    8: [3, 9 ,15, 16],
    9: [3, 8, 10, 16],
    10: [3, 6 ,7 ,9, 13],
    11: [1,2 ,3 ,5 ,7],
    12: [1, 4 ,6],
    13: [5, 6, 7, 10],
    14: [1, 6],
    15: [1, 3, 8, 16],
    16: [9,11, 15]
}


# 计算时间窗口
nodes = len(netDict)
degree = netDict.values()
s = 0
for i in degree:
    a = len(i)
    s = s + a
M = round(s / nodes + 1)
d = []
with open("上海.csv", 'r', encoding='utf-8') as cf:
    reader = csv.reader(cf)
    count = 0
    for i in reader:
        count += 1
        if count == 1:
            continue
        i = i[1:]
        temp_l = [int(item) for item in i]
        d.append(temp_l)

dArr = np.array(d)

# save daily_cases
dArr1 = dArr[:, 7:]
y2 = np.sum(dArr1, 0)
data_df2 = pd.DataFrame(y2)
data_df2.to_csv('daily_cases.csv')

def normalization(data):
    _range = np.max(data) - np.min(data)
    return (data - np.min(data)) / _range

dArr = normalization(dArr)

dArr = dArr[:, :]
edgeL = []
for key in netDict:
    for i in range(0, len(netDict[key])):
        if [key, netDict[key][i]] in edgeL or [netDict[key][i], key] in edgeL:
            continue
        else:
            edgeL.append([key, netDict[key][i]])
edgeL = [edge if edge[1] > edge[0] else [edge[1], edge[0]] for edge in edgeL]
edgeL.sort(key=lambda x: (x[0], x[1]))



dist = []
graphL = []
# Calculate the weight of edges in the network
for week in range(M, dArr.shape[1]):
    week1 = dArr[:, week - M:week]
    week2 = dArr[:, week - M + 1:week + 1]
    deltaL = []
    graphL_t = []
    dist_t = []
    for edge in edgeL:

        if np.all(week1[edge[0] - 1, :] == 0) == 1 or np.all(week1[edge[1] - 1, :] == 0) == 1:
            pcc1 = 0
        else:
            pcc1 = abs(np.nan_to_num(pearsonr(week1[edge[0] - 1, :], week1[edge[1] - 1, :])[0]))
        if np.all(week1[edge[0] - 1, :] == 0) == 1:
            sdr11 = 0
        else:
            sdr11 = np.std(week1[edge[0] - 1, :])
        if np.all(week1[edge[1] - 1, :] == 0) == 1:
            sdr12 = 0
        else:
            sdr12 = np.std(week1[edge[1] - 1, :])
        if np.all(week2[edge[0] - 1, :] == 0) == 1 or np.all(week2[edge[1] - 1, :] == 0) == 1:
            pcc2 = 0
        else:
            pcc2 = abs(np.nan_to_num(pearsonr(week2[edge[0] - 1, :], week2[edge[1] - 1, :])[0]))
        if np.all(week2[edge[0] - 1, :] == 0) == 1:
            sdr21 = 0
        else:
            sdr21 = np.std(week2[edge[0] - 1, :])
        if np.all(week2[edge[1] - 1, :] == 0) == 1:
            sdr22 = 0
        else:
            sdr22 = np.std(week2[edge[1] - 1, :])
        # 计算距离
        distij = np.linalg.norm(week2[edge[0] - 1, :] - week2[edge[1] - 1, :])
        if distij == 0:
            distij = 1
        dist_t.append([edge[0], edge[1], 1000/distij])
        sd1 = sdr11 + sdr12
        sd2 = sdr21 + sdr22
        pcc = abs(pcc2 - pcc1)
        sd = abs(sd2 - sd1)
        delta = abs(pcc1 - pcc2) * abs(sd1 - sd2) * 1000 + 2
        graphL_t.append([edge[0], edge[1], delta])
    graphL.append(graphL_t)
    dist.append(dist_t)
jsonStr = {"0": graphL}
jsonStr = {"0": dist}
# Write the file in JSON format
with open("weight.json", "w") as fp:
    fp.write(json.dumps(jsonStr, indent=4))
with open("dist.json", "w") as fp:
    fp.write(json.dumps(jsonStr, indent=4))

graphArr = np.array(graphL)
distArr = np.array(dist)

# 计算t时刻节点i概率
List1 = []
List2 = []
p_1 = []
p_2 = []
p_3 = []
I_p1 = []
I_p2 = []
I_p3 = []
entropy1 = []
entropy2 = []
entropy_ = []
x = dArr[:, dArr.shape[1] - len(graphL):dArr.shape[1]]
for t in range(0, len(graphL)):
    xt = x[:, t]
    List1 = dist[t]
    List2 = graphL[t]
    HK1 = []
    HK2 = []
    HK_ = []
    for k in netDict:
        xtk = xt[k - 1]
        pro1 = []
        pro2 = []
        L = len(netDict[k])
        for m in range(0, len(List1)):
            for n in range(0, len(List1[m])):
                Listr = List1[m]
                if n < 2:
                    if Listr[n] == k:
                            pro1.append(List1[m][2])
                            pro2.append(List2[m][2])
        t1 = sum(pro1)
        t2 = sum(pro2)
        P1 = []
        P2 = []
        P3 = []
        Ip1 = []
        Ip2 = []
        Ip3 = []
        for i in netDict[k]:
            it = xt[i - 1]
            for m in range(0, len(List1)):
                if k == List1[m][0]:
                    if List1[m][1] == i:
                        if t1 == 0:
                            p1 = 0
                            ip1 = abs(it * p1)
                        else:
                            p1 = List1[m][2] / t1
                            ip1 = abs(it * p1)
                        P1.append(p1)
                        Ip1.append(ip1)
                        if t2 == 0:
                            p2 = 0
                            ip2 = abs(it * p2)
                            p3 = abs(p2 * p1)
                            ip3 = abs(it * p3)
                        else:
                            p2 = List2[m][2] / t2
                            ip2 = abs(it * p2)
                            p3 = abs(p2 * p1)
                            ip3 = abs(it * p3)
                        P2.append(p2)
                        Ip2.append(ip2)
                        P3.append(p3)
                        Ip3.append(ip3)

                if k == List1[m][1]:
                    if List1[m][0] == i:
                        if t1 == 0:
                            p1 = 0
                        else:
                            p1 = List1[m][2] / t1
                            ip1 = abs(it * p1)
                        P1.append(p1)
                        Ip1.append(ip1)
                        if t2 == 0:
                            p2 = 0
                        else:
                            p2 = List2[m][2] / t2
                            ip2 = abs(it * p2)
                            p3 = abs(p2 * p1)
                            ip3 = abs(it * p3)
                        P2.append(p2)
                        Ip2.append(ip2)
                        P3.append(p3)
                        Ip3.append(ip3)
        p_1.append(P1)  # t时刻i的一阶邻居的概率
        p_2.append(P2)
        p_3.append(P3)
        I_p1.append(Ip1)
        I_p2.append(Ip2)
        I_p3.append(Ip3)
        H1 = 0
        H2 = 0
        for ele in range(0, len(Ip1)):
            if Ip1[ele] == 0:
                Ip1[ele] = 2
            if Ip2[ele] == 0:
                Ip2[ele] = 2
            if Ip3[ele] == 0:
                Ip3[ele] = 2

            H1 = H1 - (xt[ele] - xt[ele - 1])*P1[ele] * math.log(P1[ele]) / math.log(L + 1)
            H2 = H2 - (xt[ele] - xt[ele - 1])*P3[ele] * math.log(P2[ele]) / math.log(L + 1)
            H_ = (H2 - H1)
            if np.isnan(H_) == 1:
                H_ = 0
        HK1.append(H1)
        HK2.append(H2)
        HK_.append(H_)
    entropy1.append(HK1)
    entropy2.append(HK2)
    entropy_.append(HK_)

# save
data_df = pd.DataFrame(entropy_)
data_df.to_csv('entropy.csv', header=None, index=None, na_rep=0)
data_df2 = pd.DataFrame(entropy2)
data_df2.to_csv('entropy2.csv', header=None, index=None, na_rep=0)
data_df3 = pd.DataFrame(entropy_)
data_df3.to_csv('entropy_.csv', header=None, index=None, na_rep=0)

# 检验是否显著
Ht = []
Ht1 = []
Ht2 = []
Ht3 = []
PV = []
for i in range(M+1, len(entropy_)):
    Ht = entropy_[i]
    Ht2 = entropy_[:i]
    Ht1 = np.mean(Ht2, 0)
    stat_val, p_val = stats.ttest_ind(Ht, Ht1)
    if p_val < 0.01:
        PV.append(1/(0.01+p_val))
    else:
        PV.append(1 / p_val)
_p = pd.DataFrame(PV)
_p.to_csv('pv.csv')

# 检验是否显著
Ht_ = []
Ht1_ = []
Ht2_ = []
Ht3_ = []

_p_ = []
P_ =[]
entropy_=np.array(entropy_)
for i in range(M+1, len(entropy_)):
    Ht_ = entropy_[i-M-1:i-1]
    Ht3_ = np.mean(Ht_, 0)
    Ht2_ = entropy_[i-M:i]
    Ht1_ = np.mean(Ht2_, 0)
    PV_ = []
    for p in range(0,len(Ht1_)):

        stat_val, p_val = stats.ttest_1samp(Ht_[:,p], Ht1_[p])
        if p_val < 0.01:
            PV_.append(1/(0.01+p_val))
        else:
            PV_.append(1 / p_val)
    _p_.append(PV_)
P = pd.DataFrame(_p_)
P.to_csv('pv1.csv')