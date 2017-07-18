#!/usr/bin/python
# -*- coding: UTF-8 -*-

#假设要学习到的最后的方程为y=x^2-0.5

#构造数据
import numpy as np

# 构造300个点，分布在-1到1区间
# linspace函数通过指定开始值、终值和元素个数来创建一维数组，等差数列
x_data=np.linspace(-1,1,100)[:,np.newaxis]  # x是n
noise=np.random.normal(0,0.05,x_data.shape)
y_data = np.square(x_data)-0.5 +noise

# 输出数据
# print x_data.shape,y_data.shape

import matplotlib.pyplot as plt

# 分割出3行2列子图，在1号作图
plt.subplot(1,1,1)
plt.title('y=x^2-0.5')
plt1,=plt.plot(x_data,y_data,'o',color='red',label='train data(noise)')   
# plt1,=plt.plot(x_data,y_data,color='red',label='train data(noise)')


class Layer(object):

	# 输入为n 输出为m（该层的神经元个数），每个神经元有一个n为的weight
	# 每个单元有一个截距
	def __init__(self, input,output,active='log',hidden=True):
		super(Layer, self).__init__()
		self.input = input
		self.output = output
		self.active = active
		index =0
		self.weight=[]
		self.intercept =[] #截距
		self.name='I'+str(input)+'O'+str(output)
		self.weight =np.random.random((input,output))
		self.intercept = np.random.random((1,1))
		self.outputData = [] #每层的单元的输出结果
		self.inputData = [] #每层的单元的输出结果
		self.hidden = hidden
		print 'weight is ',self.weight
		print 'intercept is ',self.intercept


	# 输入是前一层的输出，输出是output的个数
	def calcResult(self,trainData):
		self.inputData = trainData
		self.outputData = np.dot(trainData,self.weight)
		# print 'w*i',self.outputData,np.asarray(self.outputData),self.intercept
		self.outputData = self.outputData + self.intercept
		# print 'change',self.outputData
		if self.hidden == True:
			self.outputData = self.log(self.outputData)
		return self.outputData

	def changeWeight(self,delta,rate):
		if self.hidden == True: # 隐藏层
			self.weight = self.weight - (rate * delta * self.weight * self.outputData*(1- self.outputData))
			self.intercept =  self.intercept   - (rate) * delta
			return delta
		else:  #输出层
			self.weight = self.weight -(rate) * delta
			self.intercept =  self.intercept   - (rate) * delta
			return delta


	def log(self,ndarray):
		return 1/(1+np.exp(-ndarray))

	def doubles(self,ndarray):
		return 2/(1+np.exp(-ndarray))-1

	def max(self,ndarray):
		return np.where(ndarray<0,0,ndarray)

	def predict1(self,x_data):
		print 'myweight',self.weight,x_data
		x_data = np.array(x_data).dot(self.weight)
		print 'cheng', x_data,self.intercept
		x_data = x_data + self.intercept
		if self.hidden == True:
			x_data = self.log(x_data)
		return x_data

class ANN(object):

	def __init__(self,):
		super(ANN, self).__init__()
		self.layers = []

	def logistic(i):
		return 1/(1+np.e**i)

	def addLayer(self,layer):
		self.layers.append(layer)

	def fit(self,x_data,y_data,size=20,step=1000,learningRate=0.01):
		countIndex = 0 #第n次训练
		dataIndex = 0 #数据序列
		length = len(x_data)
		while countIndex < step:
			count = 0 	#count为0
			print '[0-Fit]begin count is %d/%d' % (countIndex,step)
			while count < length:
				trainInputData = trainData = x_data[count] #当前训练的数据
				trainOutputData = y_data[count] #当前训练的数据的结果
				#print '[0-Fit]begin count is',count,'/',size, trainData,'result is', trainResult
				# 计算每层的输出
				for layer in self.layers:
					trainData = layer.calcResult(trainData)
				print '>>>[last-calcOutput]:',trainData,trainOutputData

				# 开始计算loss
				print 'loss is ',(trainOutputData-trainData)**2
				# if (trainOutputData-trainData)**2 < 0.00001:
					# return
				delta = trainData-trainOutputData
				begin = len(self.layers)
				while begin >0:
					curLayer = self.layers[begin-1]
					delta = curLayer.changeWeight(delta,learningRate)
					begin = begin - 1

				count = count +1 #开始训练下一个数据
			countIndex = countIndex +1


	def predict(self,x_data):
		for layer in self.layers:
			x_data = layer.predict1(x_data)
		return x_data

ann = ANN()
layer1 = Layer(1,20)
layer2 = Layer(20,20)
layer3 = Layer(20,1,hidden=False)
ann.addLayer(layer1)
ann.addLayer(layer2)
ann.addLayer(layer3)
ann.fit(x_data[:100],y_data)
print ann.predict([[0.2],[0.8]])


x_test=np.linspace(-1,1,100)[:,np.newaxis]
rs_real=np.square(x_test)-0.5

ann_test=ann.predict(x_test.reshape(-1,1).tolist())

print 'ann_test',ann_test,type(ann_test)


plt2,=plt.plot(x_test,rs_real,color='blue',label='real')
plt3,=plt.plot(x_test,ann_test.reshape(-1,1),color='yellow',label='ANN')
plt.legend(handles=[plt1,plt2,plt3])
plt.show()
