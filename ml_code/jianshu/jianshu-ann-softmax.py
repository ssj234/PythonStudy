#!/usr/bin/python
# -*- coding: UTF-8 -*-

# 简书教程之4.0-Softmax分类函数
# http://www.jianshu.com/p/8eb17fa41164

import numpy as np 
import matplotlib.pyplot as plt 
from matplotlib.colors import colorConverter, ListedColormap 
from mpl_toolkits.mplot3d import Axes3D 
from matplotlib import cm

# 定义softmax函数
def softmax(z):
	return np.exp(z) / np.sum(np.exp(z))

# 构造200个点
nb_of_zs = 200
zs = np.linspace(-10,10,num=nb_of_zs)
zs_1, zs_2 = np.meshgrid(zs, zs) 
y = np.zeros((nb_of_zs,nb_of_zs,2))

# 计算每个二维向量的的softmax值
for i in range(nb_of_zs):
	for j in range(nb_of_zs):
		y[i,j,:] = softmax(np.asarray([zs_1[i,j],zs_2[i,j]]))

# 绘图
fig = plt.figure()
ax = fig.gca(projection='3d')
surf = ax.plot_surface(zs_1,zs_2,y[:,:,0],linewidth=0,cmap=cm.coolwarm)
ax.view_init(elev=30,azim=70)
cbar = fig.colorbar(surf)
ax.set_xlabel('$z_1$',fontsize=15)
ax.set_ylabel('$z_1$',fontsize=15)
ax.set_zlabel('$z_1$',fontsize=15)
ax.set_title('$P(t=1|\mathbf{z})$')
cbar.ax.set_ylabel('$P(t=1|\mathbf{z})$',fontsize=15)
plt.grid()
plt.show()