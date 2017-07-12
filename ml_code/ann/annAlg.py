#!/usr/bin/python
# -*- coding: UTF-8 -*-

#假设要学习到的最后的方程为y=x^2-0.5 

#构造数据
import numpy as np

# 构造300个点，分布在-1到1区间 
# linspace函数通过指定开始值、终值和元素个数来创建一维数组，等差数列
x_data=np.linspace(-1,1,100)[:,np.newaxis]
noise=np.random.normal(0,0.05,x_data.shape)
y_data = np.square(x_data)-0.5 +noise

# 输出数据
# print x_data.shape,y_data.shape

# import matplotlib.pyplot as plt

# 分割出3行2列子图，在1号作图
# plt.subplot(1,1,1)
# plt.title('y=x^2-0.5')
# plt1,=plt.plot(x_data,y_data,color='red',label='train data(noise)')   


class Layer(object):
	
	# 输入为n 输出为m（该层的神经元个数），每个神经元有一个n为的weight
	# 每个单元有一个截距
	def __init__(self, input,output,active='log'):
		super(Layer, self).__init__()
		self.input = input
		self.output = output
		self.active = active
		index =0
		self.weight=[]
		self.intercept =[] #截距
		self.name='I'+str(input)+'O'+str(output)
		#self.outputData=[] #输出数据
		while index < output:
			self.weight.append(np.array(np.random.normal(0,0.5,input))) #[0.4 for x in range(0,input)]
			self.intercept.append(0.5)
			index = index + 1
		self.outputData = [] #每层的单元的输出结果
		self.errorData = [] #每层每个单元的ERR
		print 'weight is ',self.weight
	
	def getWeight(index):
		return self.weight[index]

	# 输入是前一层的输出，输出是output的个数
	def calcWeight(self,trainData):
		index = 0
		self.outputData = []
		while index < self.output:
			print '=========Layer=======cell',index
			# 乘以系数矩阵 加截距
			tmp = np.matrix(self.weight[index])  * np.matrix(trainData)  + self.intercept[index]
			print trainData,self.weight[index],self.intercept[index]
			self.outputData.append(tmp[0])
			# print tmp,type(tmp)
			index = index + 1
		return self.outputData

	def calcErrorLast(self,trainResult):
		index = 0
		self.errorData = []
		# 遍历单元 
		outputData = self.outputData#.tolist()
		while index < self.output:
			curOpt = outputData[index]
			err =  curOpt*(1-curOpt)*(trainResult[index]-curOpt)
			print '[][err]',type(err),err,err[0,:1],err.shape
			self.errorData.append(err[0][0])
			index = index + 1
		# print type(self.errorData)
		return self.errorData

	def calcError(self,errWeightSum):
		index = 0
		self.errorData = []
		# 遍历单元 
		while index < self.output:
			curOpt = self.outputData[index]
			err =  curOpt*(1-curOpt)*(errWeightSum[index])
			self.errorData.append(err)
			index = index + 1
		return self.errorData

	# 本层错误加权求和,使用输出的[每列为输出对应输入的权重  ipt*opt]*[O1 O2](输出列向量，opt*1)= ipt*1
	# 结果为每个输入的Err加权求和
	# ipt * 1
	def getErrWeightSum(self):
		# self.weight 是每个细胞元的
		print '>>>weight ',np.matrix(self.weight).transpose()
		print '>>>self.errorData ',self.errorData,type(self.errorData),self.errorData[0]
		print  '>>>err',np.matrix(self.errorData)
		a = np.matrix(self.weight).transpose()* np.matrix(self.errorData)
		print 'a is ' ,a 
		return a

class ANN(object):
	
	def __init__(self,):
		super(ANN, self).__init__()
		self.layers = []

	def logistic(i):
		return 1/(1+np.e**i)

	def addLayer(self,layer):
		self.layers.append(layer)

	def fit(self,x_data,y_data,size=1):
		print "fit data:" ,x_data
		# x_data，取前10个，每个
		dataIndex = 0 #数据序列
		length = len(x_data)
		while dataIndex < length:
			count = 0 	#count为0
			trainData = x_data[dataIndex] #当前训练的数据
			trainResult = y_data[dataIndex] #当前训练的数据的结果
			while count < size:
				# 计算每层的输出
				for layer in self.layers:
					trainData = layer.calcWeight(trainData)
					trainData = np.array(trainData).reshape(-1,1)
					print layer.name,' after layer:',trainData
				# 开始计算Err
				print '==============[Err]========='
				begin = len(self.layers)
				lastErr = []
				lastWeight = []
				while begin >0:
					curLayer = self.layers[begin-1]
					if(begin == len(self.layers)):
						# 输出层,计算输出单元的err 
						lastErr = curLayer.calcErrorLast(trainResult)
						#print '>>>>',curLayer.errorData ,type(curLayer.errorData)
						# 计算上一次的加权求和
						lastWeight = curLayer.getErrWeightSum()
					else:
						lastErr = curLayer.calcError(lastWeight)
						# 计算上一次的加权求和
						lastWeight = curLayer.getErrWeightSum()
					print 'last error' ,lastErr
					print 'errWeightSum',lastWeight
					begin = begin - 1

				count = count +1
			dataIndex = dataIndex +1
		

	def predict(self,x_data):
		pass


ann = ANN()
layer1 = Layer(1,4)
layer2 = Layer(4,4)
layer3 = Layer(4,1)
ann.addLayer(layer1)
ann.addLayer(layer2)
ann.addLayer(layer3)
ann.fit(x_data[:1],y_data)
ann.predict([[0.2]])