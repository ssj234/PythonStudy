#!/usr/bin/python
# -*- coding: UTF-8 -*-

#假设要学习到的最后的方程为y=x*2 

#构造数据
import numpy as np

# 构造100个点，分布在-1到1区间 
# linspace函数通过指定开始值、终值和元素个数来创建一维数组，等差数列
x_data=np.linspace(-1,1,100)[:,np.newaxis]
noise=np.random.normal(0,0.05,x_data.shape)
y_data = 2*(x_data)+noise

# 输出数据
# print x_data.shape,y_data.shape

import matplotlib.pyplot as plt

# 分割出1行1列子图，在1号作图
plt.subplot(1,1,1)
plt.title('y=x*2')
plt1,=plt.plot(x_data,y_data,'o',color='red',label='train data(noise)')   


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
		#self.outputData=[] #输出数据
		while index < output:
			self.weight.append(np.array([0.4*(x%2==0) for x in range(0,input)])) #[0.4 for x in range(0,input)] #np.random.normal(0,0.5,input)
			self.intercept.append(0)
			index = index + 1
		self.weight = np.array(self.weight)
		self.intercept = np.array(self.intercept).reshape(-1,1)
		self.outputData = [] #每层的单元的输出结果
		self.errorData = [] #每层每个单元的ERR
		self.hidden = hidden
		print 'weight is ',self.weight
		print 'intercept is ',self.intercept
	
	def getWeight(index):
		return self.weight[index]

	# 输入是前一层的输出，输出是output的个数
	def calcWeight(self,trainData):
		# index = 0
		#print '==============='
		# print 'weight',np.matrix(self.weight)
		# print 'input',np.matrix(trainData)
		self.outputData = []
		self.outputData = np.matrix(self.weight) * np.matrix(trainData)
		# print 'w*i',self.outputData,np.asarray(self.outputData),self.intercept
		self.outputData = np.asarray(self.outputData) + self.intercept
		# print 'change',self.outputData
		if self.hidden == True:
				self.outputData = self.doubles(self.outputData)
		return self.outputData

	def calcErrorLast(self,trainResult):
		index = 0
		self.errorData = []
		# print type(self.errorData)
		outdata = np.array(self.outputData)
		self.errorData = (outdata-np.array(trainResult))
		return self.errorData

	def calcError(self,errWeightSum):
		index = 0
		self.errorData = []
		outdata = np.array(self.outputData)
		self.errorData = outdata*(1-outdata)*(errWeightSum[index])
		return self.errorData

	# 本层错误加权求和,使用输出的[每列为输出对应输入的权重  ipt*opt]*[O1 O2](输出列向量，opt*1)= ipt*1
	# 结果为每个输入的Err加权求和
	# ipt * 1
	def getErrWeightSum(self):
		# self.weight 是每个细胞元的
		# print '==[get weight then add]=='
		# print '>>>EWS:weight ',np.matrix(self.weight).transpose()
		# print '>>>EWS:errorData ',self.errorData,type(self.errorData),self.errorData[0]
		# print '>>>EWS:err-matrix',np.matrix(self.errorData)
		a = np.matrix(self.weight).transpose()* np.matrix(self.errorData)
		# print '>>>EWS:rs is ' ,a 
		return a

	def log(self,ndarray):
		return 1/(1+np.exp(-ndarray))

	def doubles(self,ndarray):
		return 2/(1+np.exp(-ndarray))-1

	def predict1(self,x_data):
		print 'myweight',self.weight,x_data
		x_data =np.matrix(self.weight) * np.matrix(x_data)
		print 'cheng', x_data,self.intercept
		x_data = x_data + self.intercept
		if self.hidden == True:
			x_data = self.doubles(x_data)
		return x_data

class ANN(object):
	
	def __init__(self,):
		super(ANN, self).__init__()
		self.layers = []

	def logistic(i):
		return 1/(1+np.e**i)

	def addLayer(self,layer):
		self.layers.append(layer)

	def fit(self,x_data,y_data,size=20,learningRate=0.9):
		#print "fit data:" ,x_data
		# x_data，取前10个，每个
		dataIndex = 0 #数据序列
		length = len(x_data)
		while dataIndex < length:
			count = 0 	#count为0
			endflag = False
			trainData = x_data[dataIndex] #当前训练的数据,每次训练一个数据 应该使用批次进行训练
			trainDataBegin = x_data[dataIndex] #当前训练的数据
			trainResult = y_data[dataIndex] #当前训练的数据的结果
			print '[0-Fit]begin trainData is', trainData,'result is', trainResult
			while count < size:
				trainData = x_data[dataIndex]
				print '[0-Fit]begin count is',count,'/',size, trainData,'result is', trainResult
				# 计算每层的输出
				for layer in self.layers:
					trainData = layer.calcWeight(trainData)
					# trainData = np.array(trainData).reshape(-1,1)
					print '[1-calcOutput]:',layer.name,trainData
				print '>>>[1-calcOutput]:',trainData,trainResult
				# 开始计算Err
				# print '==============[Err]========='
				begin = len(self.layers)
				lastErr = []
				lastWeight = []
				while begin >0:
					curLayer = self.layers[begin-1]
					if(begin == len(self.layers)):
						# 输出层,计算输出单元的err 
						lastErr = curLayer.calcErrorLast(trainResult)
						# print '[lastErr]',curLayer.errorData ,type(curLayer.errorData)
						# 计算上一次的加权求和
						lastWeight = curLayer.getErrWeightSum()
					else:
						lastErr = curLayer.calcError(lastWeight)
						# print '[lastErr]',curLayer.errorData ,type(curLayer.errorData)
						# 计算上一次的加权求和
						lastWeight = curLayer.getErrWeightSum()
					print '[2-calcError]last error' ,lastErr
					# print '[calce]errWeightSum',lastWeight
					begin = begin - 1
				
				# 调整权重
				for layer in self.layers:
					if (endflag==False) and (abs(layer.errorData[0,0]) < 0.000001):
						endflag = True
						break;
					# print '-------------------------------------'
					# print '->>',layer.name,'err is:',type(layer.errorData),layer.errorData
					# print '->>',layer.name,'weight is:',type(layer.weight),layer.weight,(layer.weight).shape
					# print '->>',layer.name,'outdata is:',type(layer.outputData)
					# print '->>',layer.name,'outdata is 2:',np.array(layer.outputData)
					tmp = (np.asarray(layer.errorData)*np.array(trainDataBegin)*2*learningRate)
					# print '->>',layer.name, tmp ,type(tmp),tmp.shape
					layer.weight = layer.weight - tmp
					# layer.intercept = layer.intercept +  (np.asarray(layer.errorData)*learningRate)
					print '[3-rechange]',layer.name, layer.weight ,layer.intercept

				count = count +1 #同一个数据开始下一次训练
				if endflag==True:
					break
			dataIndex = dataIndex +1
		

	def predict(self,x_data):
		for layer in self.layers:
			x_data = layer.predict1(x_data)
		return x_data

ann = ANN()
# layer1 = Layer(1,4)
# layer2 = Layer(4,4)
layer3 = Layer(1,1,hidden=False)
# ann.addLayer(layer1)
# ann.addLayer(layer2)
ann.addLayer(layer3)
ann.fit(x_data,y_data,size=50,learningRate=0.01)
print ann.predict([[0.2]])


x_test=np.linspace(-1,1,100)[:,np.newaxis]
rs_real=x_test*2

ann_test=ann.predict(x_test.reshape(1,-1).tolist())

print 'ann_test',ann_test,type(ann_test)


plt2,=plt.plot(x_test,rs_real,color='blue',label='real')   
plt3,=plt.plot(x_test,ann_test.reshape(-1,1),color='yellow',label='ANN')   
# plt4,=plt.plot(x_test,lrPloy,color='yellow',label='ploy')   
plt.legend(handles=[plt1,plt2,plt3])
plt.show()