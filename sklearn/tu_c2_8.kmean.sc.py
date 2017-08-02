#!/usr/bin/python
# -*- coding: UTF-8 -*-

# Python机器学习及实践
# 如果评估的数据没有所属类别，习惯上使用轮廓系数来度量聚类结果的数量

import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt

# 分割出3*2*1=6个图 在1号子图作图
plt.subplot(3,2,1)

# 模拟出两个维度作为x轴和y轴的数据
x1=np.array([1,2,3,1,5,6,5,5,6,7,8,9,7,9])
x2=np.array([1,3,2,2,8,6,7,6,7,1,2,1,1,3])
X=np.array(zip(x1,x2)).reshape(len(x1),2)

# 设置x和y的范围
plt.xlim([0,10])
plt.ylim([0,10])
plt.title('Instance')
plt.scatter(x1,x2) # 展示所有点

colors=['b','g','r','c','m','y','k','b'] # 每类颜色
markers=['o','s','D','v','^','p','*','+'] #每类图标

clusters=[2,3,4,5,8] # k每次取值
subplot_counter=1 # 计数器
sc_socres=[] # 保存评分

for t in clusters:
    subplot_counter += 1
    plt.subplot(3,2,subplot_counter)
    kmeans_model=KMeans(n_clusters=t).fit(X)
    # 遍历类型
    for i,l in enumerate(kmeans_model.labels_):
    	plt.plot(x1[i],x2[i],color=colors[l],marker=markers[l],ls='None')
    plt.xlim([0,10])
    plt.ylim([0,10])
    sc_socre=silhouette_score(X,kmeans_model.labels_,metric='euclidean') # 计算轮廓系数sc
    sc_socres.append(sc_socre)

    plt.title('K=%s,silhouette coefficent=%0.03f' % (t,sc_socre))
plt.figure()
plt.plot(clusters,sc_socres,'*-')
plt.xlabel('Number of Cluster')
plt.ylabel('Number of Cluster')
plt.show()
