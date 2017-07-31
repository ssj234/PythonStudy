#!/usr/bin/python
# -*- coding: UTF-8 -*-

# 简书教程之5-构建多层网络
# http://www.jianshu.com/p/cb6d0d5d777b
# http://www.cnblogs.com/maybe2030/p/5089753.html

import numpy as np 
from sklearn import datasets, cross_validation, metrics
import matplotlib.pyplot as plt 
from matplotlib.colors import colorConverter, ListedColormap 

import itertools
import collections

# [-1-]
# 构造样本相关数据
digits = datasets.load_digits()

# 结果处理，digits.target时一个一维数组 1798个数据，将其转为向量方式
T = np.zeros((digits.target.shape[0],10))
T[np.arange(len(T)),digits.target] += 1

# 拆分为60%训练集 40%测试集
X_train, X_test, T_train, T_test = cross_validation.train_test_split(
    digits.data, T, test_size=0.4)
# 对测试集进行拆分，一半用于校验模型，一半用于最终测试
X_validation, X_test, T_validation, T_test = cross_validation.train_test_split(
    X_test, T_test, test_size=0.5)

# 展现图像
# fig = plt.figure(figsize=(10, 1), dpi=100) 
# for i in range(10): 
# 	ax = fig.add_subplot(1,10,i+1) 
# 	ax.matshow(digits.images[i], cmap='binary') 
# 	ax.axis('off') 
#plt.show()


def logistic(z):
	return 1 / (1 + np.exp(-z))

def logistic_deriv(y):
	return np.multiply(y,(1-y))

def softmax(z):
	return np.exp(z) / np.sum(np.exp(z),axis=1,keepdims=True)


# 定义层
class Layer(object):

	# 返回该层的参数(LinearLayer)
	def get_params_iter(self):
		return []

	# 返回梯度
	def get_params_grad(self,X,output_grad):
		return []

	# 返回输出
	def get_output(self,X):
		pass

	# 
	def get_input_grad(self, Y, output_grad=None, T=None):
		pass

# 线性层
class LinearLayer(Layer):

	def __init__(self,n_in,n_out):

		self.W = np.random.randn(n_in,n_out) * 0.1
		self.b = np.zeros(n_out)

	def get_params_iter(self):
		return itertools.chain(np.nditer(self.W,op_flags=['readwrite']),
			np.nditer(self.b,op_flags=['readwrite']))

	def get_output(self,X):
		return X.dot(self.W) + self.b

	# 计算梯度
	def get_params_grad(self, X, output_grad):
		JW = X.T.dot(output_grad)
		Jb = np.sum(output_grad,axis=0)
		return [g for g in itertools.chain(np.nditer(JW),np.nditer(Jb))]

	# 参见隐藏层设计一节，求 H|Wh = Wo * Z (Z = fun(Wh))
	def get_input_grad(self, Y, output_grad):
		return output_grad.dot(self.W.T)

# Logistic
class LogisticLayer(Layer):

	def get_output(self,X):
		return logistic(X)

	# logistic求导 [->Y|Z = ((y)(1 - y))]
	def get_input_grad(self,Y,output_grad):
		return np.multiply(logistic_deriv(Y), output_grad)

# SoftmaxOutputLayer
class SoftmaxOutputLayer(Layer):

	def  get_output(self,X):
		return softmax(X)

	# 交叉熵损失函数求导 [->|z = (y - t)]
	def get_input_grad(self,Y,T):
		return (Y - T) / Y.shape[0]

	def get_cost(self, Y, T):
		return - np.multiply(T,np.log(Y)).sum() / Y.shape[0]

# 构造神经网络
hidden_neurons_1 = 20
hidden_neurons_2 = 20
layers = []
# 第一层
layers.append(LinearLayer(X_train.shape[1],hidden_neurons_1))
layers.append(LogisticLayer())
# 第二层
layers.append(LinearLayer(hidden_neurons_1,hidden_neurons_2))
layers.append(LogisticLayer())
# 第三层
layers.append(LinearLayer(hidden_neurons_2,T_train.shape[1]))
layers.append(SoftmaxOutputLayer())


# 向前
def forward_step(input_samples,layers):
	activations = [input_samples]
	X = input_samples
	for layer in layers:
		Y = layer.get_output(X)
		activations.append(Y)
		X = activations[-1]
	return activations

# 向后
def backward_step(activations,targets,layers):
	param_grads = collections.deque()
	output_grad = None
	for layer in reversed(layers):
		Y = activations.pop()

		# 根据误差倒推各层的梯度
		if output_grad is None:
			input_grad = layer.get_input_grad(Y,targets)
		else:
			input_grad = layer.get_input_grad(Y,output_grad)

		# get_params_grad只有Linear层有，计算第一层的梯度
		X = activations[-1] # 由于倒序，-1为输入
		grads = layer.get_params_grad(X,output_grad)
		param_grads.appendleft(grads)
		output_grad = input_grad
	return list(param_grads)


# 梯度检查
nb_samples_gradientcheck = 10 
X_temp = X_train[0:nb_samples_gradientcheck,:]
T_temp = T_train[0:nb_samples_gradientcheck,:]

activations = forward_step(X_temp, layers) # 返回每次输出的列表
param_grads = backward_step(activations, T_temp, layers)
eps = 0.0001
for idx in range(len(layers)):
    layer = layers[idx]
    layer_backprop_grads = param_grads[idx]
    
    for p_idx, param in enumerate(layer.get_params_iter()):
        grad_backprop = layer_backprop_grads[p_idx]
        # + eps
        param += eps
        plus_cost = layers[-1].get_cost(forward_step(X_temp, layers)[-1], T_temp)
        # - eps
        param -= 2 * eps
        min_cost = layers[-1].get_cost(forward_step(X_temp, layers)[-1], T_temp)
        # reset param value
        param += eps
        # calculate numerical gradient
        grad_num = (plus_cost - min_cost)/(2*eps)
        # Raise error if the numerical grade is not close to the backprop gradient
        if not np.isclose(grad_num, grad_backprop):
            raise ValueError('Numerical gradient of {:.6f} is not close to the backpropagation gradient of {:.6f}!'.format(float(grad_num), float(grad_backprop)))
print('No gradient errors found')


# 根据批量大小进行拆分，每次训练一小批次
batch_size = 25
nb_of_batches = X_train.shape[0] / batch_size # 需要训练的次数

XT_batches = zip(
	np.array_split(X_train,nb_of_batches,axis=0),
	np.array_split(T_train,nb_of_batches,axis=0)
	)

# 遍历每层，更新参数
def update_params(layers,param_grads,learning_rate):
	for layer,layer_backprop_grads in zip(layers, param_grads):
		for param,grad in itertools.izip(layer.get_params_iter(),layer_backprop_grads):
			param -= learning_rate * grad


minibatch_costs = []
training_costs = []
validation_costs = []

max_nb_of_iterations = 300
learning_rate = 0.1

for iteration in range(max_nb_of_iterations):
	for X,T in XT_batches:
		activations = forward_step(X,layers) # 向前
		minibatch_cost = layers[-1].get_cost(activations[-1],T) # 计算cost
		minibatch_costs.append(minibatch_cost) # 保存cost
		param_grads = backward_step(activations,T,layers) # 向后
		update_params(layers,param_grads,learning_rate) # 更新参数
	activations = forward_step(X_train,layers)
	train_cost = layers[-1].get_cost(activations[-1],T_train)
	training_costs.append(train_cost)

	activations = forward_step(X_validation,layers)
	validation_cost = layers[-1].get_cost(activations[-1], T_validation)
	validation_costs.append(validation_cost)
	if len(validation_costs) > 3:
		# 3次
		if validation_costs[-1] >= validation_costs[2] >= validation_costs[-3]:
			break

nb_of_iterations = iteration + 1

minibatch_x_inds = np.linspace(0, nb_of_iterations, num=nb_of_iterations*nb_of_batches)
iteration_x_inds = np.linspace(1, nb_of_iterations, num=nb_of_iterations)
# Plot the cost over the iterations
plt.plot(minibatch_x_inds, minibatch_costs, 'k-', linewidth=0.5, label='cost minibatches')
plt.plot(iteration_x_inds, training_costs, 'r-', linewidth=2, label='cost full training set')
plt.plot(iteration_x_inds, validation_costs, 'b-', linewidth=3, label='cost validation set')
# Add labels to the plot
plt.xlabel('iteration')
plt.ylabel('$\\xi$', fontsize=15)
plt.title('Decrease of cost over backprop iteration')
plt.legend()
x1,x2,y1,y2 = plt.axis()
plt.axis((0,nb_of_iterations,0,2.5))
plt.grid()
plt.show()



# 计算性能

y_true = np.argmax(T_test,axis=1)
activations = forward_step(X_test,layers)
y_pred = np.argmax(activations[-1],axis=1)
test_accuracy = metrics.accuracy_score(y_true,y_pred)
print 'test is ', test_accuracy

