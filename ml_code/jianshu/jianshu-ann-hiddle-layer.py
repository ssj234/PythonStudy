#!/usr/bin/python
# -*- coding: UTF-8 -*-

# 简书教程之3-隐藏层设计
# http://www.jianshu.com/p/8e1e6c8f6d52

import numpy as np
import matplotlib.pyplot as plt

from matplotlib.colors import colorConverter, ListedColormap 
from mpl_toolkits.mplot3d import Axes3D 
from matplotlib import cm

# [-1-]
# 二个类别，用蓝色表示t=1 红色表示t=0
# 数据是一维的，但并非线性分割的

nb_of_samples_per_class =20
blue_mean = [0]
red_left_mean = [-2]
red_right_mean = [2]

# 生成分类，蓝色被两边的红色包围
std_dev = 0.5
x_blue = np.random.randn(nb_of_samples_per_class,1) * std_dev + blue_mean # 20 * 1
x_red_left = np.random.randn(nb_of_samples_per_class/2,1) * std_dev + red_left_mean
x_red_right = np.random.randn(nb_of_samples_per_class/2,1)* std_dev + red_right_mean

x = np.vstack((x_blue,x_red_left,x_red_right))
t = np.vstack((np.ones((x_blue.shape[0],1)),
				np.zeros((x_red_left.shape[0],1)),
				np.zeros((x_red_right.shape[0],1))))

plt.subplot(2,2,1)
plt.figure(figsize=(8,0.5))
plt.xlim(-3,3)
plt.ylim(-1,1)

# zeros_like 根据参数的shape生成一个全为0的数组
plt.plot(x_blue,np.zeros_like(x_blue),'b|',ms = 30)
plt.plot(x_red_left,np.zeros_like(x_red_left),'r|',ms = 30)
plt.plot(x_red_right,np.zeros_like(x_red_right),'r|',ms = 30) 
plt.gca().axes.get_yaxis().set_visible(False)
plt.title('Input samples from the blue and red class')
plt.xlabel('$x$',fontsize=15)
plt.show()


# [-2-]
# 绘制RBF函数的图像
def rbf(z):
	return np.exp(-z**2)

z = np.linspace(-6,6,100)
plt.subplot(2,2,2)
plt.plot(z,rbf(z),'b-')
plt.xlabel('$z$',fontsize=15)
plt.ylabel('$e^{-z^2}$',fontsize=15)
plt.title('RBF function')
plt.grid()
plt.show()




# logistic函数
def logistic(z):
	return 1 / (1 + np.exp(-z))

# 隐藏层的激活函数
def hidden_activations(x,wh):
	return rbf(x * wh)

# 输出层的激活函数
def output_activations(h,wo):
	return logistic(h * wo - 1)

# 神经网络计算
def nn(x, wh ,wo):
	return output_activations(hidden_activations(x,wh),wo)

# 神经网络预测
def nn_predict(x, wh , wo):
	return np.around(nn(x,wh,wo))

# 损失函数
def cost(y,t):
	return -np.sum(np.multiply(t,np.log(y)) + np.multiply((1-t),np.log(1-t)))

# 保证损失函数 先计算神经网络的输出，再计算损失
def cost_for_param(x,wh,wo,t):
	return cost(nn(x,wh,wo),t)


# [-3-]
#

nb_of_ws = 200
wsh = np.linspace(-10,10,num=nb_of_ws)
wso = np.linspace(-10,10,num=nb_of_ws)

ws_x,ws_y = np.meshgrid(wsh,wso)
cost_ws = np.zeros((nb_of_ws,nb_of_ws))

for i in range(nb_of_ws):
	for j in range(nb_of_ws):
		cost_ws[i,j] = cost(nn(x,ws_x[i,j],ws_y[i,j]),t)
fig = plt.figure()
ax = Axes3D(fig)
surf = ax.plot_surface(ws_x,ws_y,cost_ws,linewidth=1,cmap=cm.pink)
ax.view_init(elev=60,azim=-30)
cbar = fig.colorbar(surf)
ax.set_xlabel('$w_h$',fontsize=15)
ax.set_ylabel('$w_o$',fontsize=15)
ax.set_zlabel('$\\xi$',fontsize=15)
cbar.ax.set_ylabel('$\\xi$',fontsize=15)
plt.title('Cost function surface')
plt.grid()
plt.show()


#

# y-t 误差
def gradient_output(y,t):
	return y - t

def gradient_weight_out(h,grad_output):
	return h * grad_output

#
def gradient_hidden(wo,grad_output):
	return wo * grad_output

def gradient_weight_hidden(x, zh,h,grad_output):
	return -2 * x * zh * h *grad_output

def backprop_update(x,t,wh,wo,learning_rate):
	# 隐藏层
	zh = x * wh
	h = rbf(zh)
	# 输出层
	y=output_activations(h,wo)
	# 计算差值
	grad_output = gradient_output(y,t)
	
	# 计算输出层的梯度
	d_wo = learning_rate * gradient_weight_out(h,grad_output)
	# 计算隐藏层的差值
	grad_hidden = gradient_hidden(wo,grad_output)
	# 计算隐藏层的梯度
	d_wh = learning_rate * gradient_weight_hidden(x,zh,h,grad_hidden)
	return (wh-d_wh.sum(),wo-d_wo.sum())

# []
# 开始训练

# 初始化权重和学习率
wh = 2
wo = -5
learning_rate = 0.2

nb_of_train = 50
lr_update = learning_rate /nb_of_train # ?

w_cost_iter =[(wh,wo,cost_for_param(x,wh,wo,t))]
for i in range(nb_of_train):
	learning_rate -= lr_update # 减少学习率 why 越来越小
	wh,wo = backprop_update(x,t,wh,wo,learning_rate)
	w_cost_iter.append((wh,wo,cost_for_param(x,wh,wo,t)))
# final cost is nan , wh is 1.136488 wo is 5.508287
print 'final cost is %f , wh is %f wo is %f' % (cost_for_param(x, wh, wo, t), wh, wo)


# [--]
# 分类结果可视化
nb_of_xs = 100

xs = np.linspace(-3,3,num=nb_of_xs)
ys = np.linspace(-1,1,num=nb_of_xs)
xx,yy = np.meshgrid(xs,ys)
classification_plane = np.zeros((nb_of_xs, nb_of_xs))
for i in range(nb_of_xs):
	for j in range(nb_of_xs):
		classification_plane[i,j] = nn_predict(xx[i,j], wh, wo)
cmap = ListedColormap([
        colorConverter.to_rgba('r', alpha=0.25),
        colorConverter.to_rgba('b', alpha=0.25)])

plt.figure(figsize=(8,0.5))
plt.contourf(xx, yy, classification_plane, cmap=cmap)
plt.xlim(-3,3)
plt.ylim(-1,1)
plt.plot(x_blue, np.zeros_like(x_blue), 'b|', ms = 30) 
plt.plot(x_red_left, np.zeros_like(x_red_left), 'r|', ms = 30) 
plt.plot(x_red_right, np.zeros_like(x_red_right), 'r|', ms = 30) 
plt.gca().axes.get_yaxis().set_visible(False) 
plt.title('Input samples and their classification') 
plt.xlabel('x')
plt.show()


# final cost is nan , wh is 1.136488 wo is 5.508287
# 神经网络模型能够利用Logistic进行分类是因为
# 隐藏层的RBF转换函数可以将靠近原点的样本（蓝色）的输出值大于0，

x_data_show = np.vstack((x_blue,x_red_left,x_red_right))
plt.plot(x_blue,rbf(x_blue*1.136488),'bo')
plt.plot(x_red_left,rbf(x_red_left*1.136488),'ro')
plt.plot(x_red_right,rbf(x_red_right*1.136488),'yo')
plt.xlabel('$z$',fontsize=15)
plt.ylabel('$e^{-z^2}$',fontsize=15)
plt.title('RBF function result')
plt.grid()
plt.show()
