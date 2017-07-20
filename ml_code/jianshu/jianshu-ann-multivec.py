#!/usr/bin/python
# -*- coding: UTF-8 -*-

# 简书教程之4.0-矢量化
# http://www.jianshu.com/p/1fe8ab3da28c

import numpy as np 
import sklearn.datasets
import matplotlib.pyplot as plt 
from matplotlib.colors import colorConverter, ListedColormap 
from mpl_toolkits.mplot3d import Axes3D 
from matplotlib import cm

# [-¶-]
# 生成样本
X,t = sklearn.datasets.make_circles(n_samples=100,shuffle=False,factor=0.3,noise=0.1)
# 根据t标记分类结果
T = np.zeros((100,2))
T[t==1,1] = 1
T[t==0,0] = 1

x_red = X[t==0]
x_blue = X[t==1]

print 'shape of X:', X.shape
print 'shape of T:', T.shape

# 绘制样本

plt.plot(x_red[:,0],x_red[:,1],'ro',label='class red')
plt.plot(x_blue[:,0],x_blue[:,1],'bo',label='class blue')
plt.grid()
plt.legend(loc=1)
plt.xlabel('$x_1$',fontsize=15)
plt.xlabel('$x_2$',fontsize=15)
plt.axis([-1.5,1.5,-1.5,1.5])
plt.show()



#
def logistic(z):
	return 1 / ( 1 + np.exp(-z))

def softmax(z):
	return np.exp(z)/np.sum(np.exp(z),axis=1,keepdims=True)

# 隐藏层激活函数
def hidden_activations(X,Wh,bh):
	return logistic(X.dot(Wh) + bh)

# 输出层激活函数
def output_activations(H,Wo,bo):
	return softmax(H.dot(Wo) + bo)

# 神经网络
def nn(X,Wh,bh,Wo,bo):
	return output_activations(hidden_activations(X,Wh,bh),Wo,bo)

# 预测
def nn_predict(X,Wh,bh,Wo,bo):
	return np.around(nn(X,Wh,bh,Wo,bo))



# [-¶-]
# 更新输出层的权重和偏置项

def cost(Y,T):
	return - np.multiply(T,np.log(Y)).sum()

# 输出结果之差
def error_output(Y,T):
	return Y - T

# 输出层的权重
def gradient_weight_output(H,Eo):
	return H.T.dot(Eo)

# 输出层的偏置更新
def gradient_bias_output(Eo):
	return np.sum(Eo, axis=0,keepdims=True)

# 隐藏层的[误差|Zh]
def error_hidden(H,Wo,Eo):
	return np.multiply(np.multiply(H,(1-H)),Eo.dot(Wo.T))

# 隐藏层的权重梯度
def gradient_weight_hidden(X,Eh):
	return X.T.dot(Eh)

# 隐藏层的偏置梯度
def gradient_bias_hidden(Eh):
	return np.sum(Eh,axis=0,keepdims=True)


# [-¶-]
# 初始化参数

init_var = 1
bh = np.random.randn(1,3) * init_var
Wh = np.random.randn(2,3) * init_var
bo = np.random.randn(1,2) * init_var
Wo = np.random.randn(3,2) * init_var

# 计算隐藏层输出
H = hidden_activations(X,Wh,bh)
# 计算输出层输出
Y = output_activations(H,Wo,bo)
# 计算输出层差值 Eo=[loss|zo]
Eo = error_output(Y,T)
JWo = gradient_weight_output(H,Eo)
Jbo = gradient_bias_output(Eo)
# 计算隐藏层差值 Eh=[loss|zh]
Eh = error_hidden(H,Wo,Eo)
JWh = gradient_weight_hidden(X,Eh)
Jbh = gradient_bias_hidden(Eh)


# [-¶-]
# 梯度检查
params = [Wh,bh,Wo,bo]
grad_params = [JWh,Jbh,JWo,Jbo]
eps = 0.0001 #计算梯度

for p_idx in range(len(params)):
	#每次检查参数
	for row in range(params[p_idx].shape[0]):
		for col in range(params[p_idx].shape[1]):
			# 修改参数，分别加eps和减eps
			p_matrix_min = params[p_idx].copy()
			p_matrix_min[row,col] -= eps
			p_matrix_plus = params[p_idx].copy()
			p_matrix_plus[row,col] += eps

			# 将参数更新回去
			params_min = params[:]
			params_min[p_idx] = p_matrix_min
			params_plus = params[:]
			params_plus[p_idx] = p_matrix_plus

			# 计算两者的损失值之差，与原来的梯度进行比对，若差值不在容忍范围内会抛出异常
			#  numpy.isclose(a, b, rtol=1e-05, atol=1e-08, equal_nan=False)[source]
			grad_num = (cost(nn(X,*params_plus),T)-cost(nn(X,*params_min),T))/(2*eps)
			if not np.isclose(grad_num,grad_params[p_idx][row,col]):
				raise ValueError('Numerical gradient of {:.6f} is not close to the backpropagation gradient of {:.6f}!'.format(float(grad_num), float(grad_params[p_idx][row,col])))
print 'No gradient errors found'


# 计算梯度
def backprop_gradients(X,T,Wh,bh,Wo,bo):
	
	# 计算隐藏层输出
	H = hidden_activations(X,Wh,bh)
	# 计算输出层输出
	Y = output_activations(H,Wo,bo)
	# 计算输出层差值 Eo=[loss|zo]
	Eo = error_output(Y,T)
	JWo = gradient_weight_output(H,Eo)
	Jbo = gradient_bias_output(Eo)
	# 计算隐藏层差值 Eh=[loss|zh]
	Eh = error_hidden(H,Wo,Eo)
	JWh = gradient_weight_hidden(X,Eh)
	Jbh = gradient_bias_hidden(Eh)
	return [JWh,Jbh,JWo,Jbo]

# 更新速度：参数分别为：X:输入样本 T:样本分类 []:四个参数 momentum_term:速度根据阻力减小的值
def update_velocity(X,T,ls_of_params,Vs,momentum_term,learning_rate):
	# 计算梯度
	Js = backprop_gradients(X,T,*ls_of_params)
	# zip按列组合
	return [momentum_term * V - learning_rate * J for V,J in zip(Vs,Js)]
# 更新参数
def update_params(ls_of_params,Vs):
	return [P + V for P,V in zip(ls_of_params,Vs)]


# [--]
# 训练并展现cost下降

learning_rate = 0.02
momentum_term = 0.9

# 初始速度
Vs = [np.zeros_like(M) for M in [Wh,bh,Wo,bo]]

nb_of_train = 300
lr_update = learning_rate / nb_of_train # 学习率每次更新
ls_costs = [cost(nn(X,Wh,bh,Wo,bo),T)] # 保存损失函数的值
for i in range(nb_of_train):
	# 参数分别为：X:输入样本 T:样本分类 []:四个参数 Vs:当前速度 momentum_term:速度根据阻力减小的值
	Vs = update_velocity(X,T,[Wh,bh,Wo,bo],Vs,momentum_term,learning_rate)
	Wh,bh,Wo,bo = update_params([Wh,bh,Wo,bo],Vs)
	ls_costs.append(cost(nn(X,Wh,bh,Wo,bo),T))

plt.plot(ls_costs,'b-')
plt.xlabel('train')
plt.ylabel('$\\xi$',fontsize=15)
plt.title('Decrease of cost over backprop iteration')
plt.grid()
plt.show()



# 可视化结果
nb_of_xs =200
xs1 = np.linspace(-2,2,num=nb_of_xs)
xs2 = np.linspace(-2,2,num=nb_of_xs)

xx,yy = np.meshgrid(xs1,xs2)
classification_plane = np.zeros((nb_of_xs,nb_of_xs))

for i in range(nb_of_xs):
	for j in range(nb_of_xs):
		pred = nn_predict(np.asarray([xx[i,j], yy[i,j]]), Wh, bh, Wo, bo)
		classification_plane[i,j] = pred[0,0]
cmap = ListedColormap([
        colorConverter.to_rgba('b', alpha=0.30),
        colorConverter.to_rgba('r', alpha=0.30)])

plt.contourf(xx, yy, classification_plane, cmap=cmap) 
plt.plot(x_red[:,0], x_red[:,1], 'ro', label='class red') 
plt.plot(x_blue[:,0], x_blue[:,1], 'bo', label='class blue') 
plt.grid() 
plt.legend(loc=1) 
plt.xlabel('$x_1$', fontsize=15) 
plt.ylabel('$x_2$', fontsize=15) 
plt.axis([-1.5, 1.5, -1.5, 1.5]) 
plt.title('red vs blue classification boundary') 
plt.show()


# 隐藏层的数据分类

H_blue = hidden_activations(x_blue,Wh,bh)
H_red = hidden_activations(x_red,Wh,bh)

fig = plt.figure()
ax =Axes3D(fig)
ax.plot(np.ravel(H_blue[:,0]),np.ravel(H_blue[:,1]),np.ravel(H_blue[:,2]),'bo')
ax.plot(np.ravel(H_red[:,0]),np.ravel(H_red[:,1]),np.ravel(H_red[:,2]),'ro')
ax.set_xlabel('$h_1$',fontsize=15)
ax.set_ylabel('$h_2$',fontsize=15)
ax.set_zlabel('$h_3$',fontsize=15)
ax.view_init(elev=10,azim=40)
plt.grid()
plt.show()