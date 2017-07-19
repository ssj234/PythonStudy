#!/usr/bin/python
# -*- coding: UTF-8 -*-

# 简书教程之1-神经网络入门之线性回归
# http://www.jianshu.com/p/0da9eb3fd06b

# 一个非常简单的神经网络
# 目标函数，损失函数
# 梯度下降

#假设要学习到的最后的方程为y=x*2 

#构造数据
import numpy as np
import matplotlib.pyplot as plt

# 产生0-1中均匀分布的样本值，生成20个，结果是1行20列
x = np.random.uniform(0,1,20)

# 目标函数
def f(x):return x*2

# 加入噪声
noise_variance = 0.2
noise = np.random.randn(x.shape[0]) * noise_variance
t = f(x) + noise

# 绘制

plt.plot(x, t, 'o', label='t')
plt.plot([0, 1], [f(0), f(1)], 'b-', label='f(x)')
plt.xlabel('$x$', fontsize=15) 
plt.ylabel('$t$', fontsize=15) 
plt.ylim([0,2]) 
plt.title('inputs (x) vs targets (t)') 

# plt.show()


# 以上是创建的测试数据，下面开始拟合


# 定义神经网络的函数
def nn(x,w): return x*w

# 定义损失函数
def cost(y,t):return ((t-y) ** 2).sum()


# 这是
def gradient(w,x,t):
	return 2 * x * (nn(x,w) - t)

# 返回的是每次减去的delta(w)，w_k是第k步的权重，x是输入，t是目标值，由于参数是ndarray，即多笔测试数据，需要sum1
def delta_w(w_k,x,t,learning_rate):
	return learning_rate * gradient(w_k,x,t).sum()

# 初始化权重为0.1
w = 0.1

# 初始化学习率为0.1
learning_rate = 0.1

nb_of_iterations = 100 # 梯度下降更新数量
w_cost = [(w,cost(nn(x,w),t))] # 存储权重，损失函数结果

for i in range(nb_of_iterations):
	dw = delta_w(w,x,t,learning_rate) # 计算梯度
	w = w -dw #更新权重
	w_cost.append((w,cost(nn(x,w),t))) #保存权重和loss

for i in range(0,len(w_cost)):
	print 'w %f cost: %f' % (w_cost[i][0],w_cost[i][1])


# 可视化
plt.plot([0,1],[w*0,w*1],'r-',label='fitted line')
plt.grid() 
plt.legend(loc=3) 
plt.show()