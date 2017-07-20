#!/usr/bin/python
# -*- coding: UTF-8 -*-

# 简书教程之5-构建多层网络
# http://www.jianshu.com/p/cb6d0d5d777b

import numpy as np 
from sklearn import datasets, cross_validation, metrics
import matplotlib.pyplot as plt 
from matplotlib.colors import colorConverter, ListedColormap 

import itertools
import collections


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

	def get_params_iter(self):
		return []

	def get_params_grad(self,X,output_grad):
		return []

	def get_output(self,X):
		pass
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

	def get_params_grad(self, Y, output_grad):
		JW = X.T.dot(output_grad)
		Jb = np.sum(output_grad,axis=0)
		return [g for g in itertools.chain(np.nditer(JW),np.nditer(Jb))]

	def get_input_grad(self, Y, output_grad):
		return output_grad.dot(self.W.T)





