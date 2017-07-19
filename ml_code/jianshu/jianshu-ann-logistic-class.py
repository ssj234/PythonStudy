#!/usr/bin/python
# -*- coding: UTF-8 -*-

# 简书教程之2-Logistic分类函数
# http://www.jianshu.com/p/abc2acf092a3

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import colorConverter, ListedColormap
from matplotlib import cm

# 当t=1 用蓝色表示  t=0 使用红色表示
# 程序中输入参数x是一个N*2的矩阵，目标分类是N*1的向量

nb_of_samples_per_class = 20
red_mean = [-1,0]
blue_mean = [1,0]
std_dev = 1.2 #偏移

x_red = np.random.randn(nb_of_samples_per_class,2) * std_dev + red_mean
x_blue = np.random.randn(nb_of_samples_per_class,2) * std_dev + blue_mean

X = np.vstack((x_red,x_blue))
t = np.vstack((np.zeros((nb_of_samples_per_class,1)),np.ones((nb_of_samples_per_class,1)))) #red=0

# 在第一个子图中绘制生成的点
plt.subplot(2,2,1)
plt.plot(x_red[:,0],x_red[:,1],'ro',label='class red')
plt.plot(x_blue[:,0],x_blue[:,1],'bo',label='class blue')
plt.grid()
plt.legend(loc=2)
plt.xlabel('$x_1$',fontsize=15)
plt.ylabel('$x_2$',fontsize=15)
plt.axis([-4,4,-4,4])



# 我们的目的是根据输入x取预测分类t，假设 输入x=[x1,x2] 权重w=[w1,w2] 预测目标t=1
# P(t=1 |x,w)是神经网输出的y，即 y=o(x*wT)，o表示为logistic函数

# logistic函数
def logistic(z):
	return 1 / (1 + np.exp(-z))

# 神经网络加权求和
def nn(x,w):
	return logistic(x.dot(w.T))

# 返回0或1 预测
def nn_predict(x,w):
	return np.around(nn(x,w))

# 交叉熵误差函数作为损失函数
def cost(y,t):
	return -np.sum(np.multiply(t,np.log(y)) + np.multiply((1-t),np.log(1-y)))

nb_of_ws =100 #
ws1 = np.linspace(-5,5,num = nb_of_ws)
ws2 = np.linspace(-5,5,num = nb_of_ws)

ws_x,ws_y = np.meshgrid(ws1,ws2)
cost_ws = np.zeros((nb_of_ws,nb_of_ws))

# 从(-5,5),(-5,5)各种组合为不同的权重生成不同的损失
for i in range(nb_of_ws):
	for j in range(nb_of_ws):
		cost_ws[i,j] = cost(nn(X,np.asmatrix([ws_x[i,j],ws_y[i,j]])),t)

# 在第二个子图绘制损失
plt.subplot(2,2,2)
plt.contourf(ws_x,ws_y,cost_ws,20,cmap=cm.pink)
cbar = plt.colorbar()
cbar.ax.set_ylabel('$\\xi$',fontsize=15)
plt.xlabel('$w_1$',fontsize=15)
plt.ylabel('$w_2$',fontsize=15)



# 计算梯度
def gradient(w,x,t):
	return (nn(x,w) - t).T * x

# 计算delta
def delta_w(w_k,x,t,learning_rate):
	return learning_rate * gradient(w_k,x,t)

# 初始化权重参数和学习率
w = np.asmatrix([-4,-2])
learning_rate = 0.05

# 训练次数
nb_of_train = 10
w_train =[w] # 保存权重

# 这里是训练
for i in range(nb_of_train):
	dw = delta_w(w,X,t,learning_rate)
	w = w - dw
	w_train.append(w)


# 展示权重变化路径
for i in range(1,4):
	w1 = w_train[i-1]
	w2 = w_train[i]
	plt.plot(w1[0,0],w1[0,1],'bo')
	plt.plot([w1[0,0],w2[0,0]],[w1[0,1],w2[0,1]],'b-')
	plt.text(w1[0,0]-0.2,w1[0,1]+0.4,'$w({})'.format(i),color='b')

w1 = w_train[3]

plt.plot(w1[0,0],w1[0,1],'bo')
plt.text(w1[0,0]-0.2,w1[0,1]+0.4,'$w({})'.format(4),color='b')

plt.xlabel('$w_1$',fontsize=15)
plt.ylabel('$w_2$',fontsize=15)
plt.grid()


# 展示训练后的预测结果，生成200个点，分别进行预测，之后绘制出分类平面
nb_of_xs =200
xs1 = np.linspace(-4,4,num=nb_of_xs)
xs2 = np.linspace(-4,4,num=nb_of_xs)
xx,yy = np.meshgrid(xs1,xs2)

classfication_plane = np.zeros((nb_of_xs,nb_of_xs))
for i in range(nb_of_xs):
	for j in range(nb_of_xs):
		classfication_plane[i,j] = nn_predict(np.asmatrix([xx[i,j],yy[i,j]]),w)

cmap = ListedColormap([
        colorConverter.to_rgba('r', alpha=0.30),
        colorConverter.to_rgba('b', alpha=0.30)])
plt.subplot(2,2,3)
plt.contourf(xx, yy, classfication_plane, cmap=cmap) 
plt.plot(x_red[:,0], x_red[:,1], 'ro', label='target red') 
plt.plot(x_blue[:,0], x_blue[:,1], 'bo', label='target blue') 
plt.legend(loc=2) 
plt.xlabel('$x_1$', fontsize=15) 
plt.ylabel('$x_2$', fontsize=15) 
plt.title('red vs. blue classification boundary') 
plt.grid() 
plt.show()
