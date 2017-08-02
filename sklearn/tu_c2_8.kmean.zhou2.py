#!/usr/bin/python
# -*- coding: UTF-8 -*-

# Python机器学习及实践的第二章中的例子，
# 聚类是选择k的大小，肘部观察法，上个例子k=3，根据3进行聚类

import numpy as np
from sklearn.cluster import KMeans
from scipy.spatial.distance import cdist
import matplotlib.pyplot as plt

cluster1=np.random.uniform(0.5,1.5,(2,10))
cluster2=np.random.uniform(5.5,6.5,(2,10))
cluster3=np.random.uniform(3.0,4.0,(2,10))
X=np.hstack((cluster1,cluster2,cluster3)).T


# 绘制30个数据样本的分布图像
plt.scatter(X[:,0],X[:,1])
plt.xlabel('x1')
plt.ylabel('x2')
plt.show()


colors=['b','g','r']
markers=['o','s','D']

# 使用3进行训练
kmeans=KMeans(n_clusters=3)
kmeans_model=kmeans.fit(X)
# 绘制结果
for i,l in enumerate(kmeans_model.labels_):
    # print 'i=',i,' l=',l,' X=',X[0][i],X[1][i]
    plt.plot(X[i][0],X[i][1],color=colors[l],marker=markers[l],ls='None')
plt.xlabel('k')
plt.ylabel('Average Dispersion')
plt.show()