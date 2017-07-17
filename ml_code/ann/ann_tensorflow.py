#!/usr/bin/python
# -*- coding: UTF-8 -*-

import tensorflow as tf
import numpy as np

#构造满足一元二次方程的函数，采用np生成等差数列的方法，并将结果为300个点的一维数组，转换成300*1的二维数组
x_data = np.linspace(-1,1,300)[:,np.newaxis]
noise =  np.random.normal(0,0.05,x_data.shape) #加入噪声，均值为0 方差为0.05
y_data = np.square(x_data) - 0.5 +noise  #  y = x^2 - 0.5 + noise

#定义x和y的占位符作为将要输入神经网络的变量：
xs = tf.placeholder(tf.float32,[None,1])
ys = tf.placeholder(tf.float32,[None,1])

#weights =[]
#biases = []
#weights = tf.Variable(tf.random_normal([in_size,out_size]))
#biases = tf.Variable(tf.zeros([1,out_size]) + 0.1)
def add_layer(inputs,in_size,out_size,activation_function=None):
    #构建权重 in_size * out_size 大小的矩阵
    weights = tf.Variable(tf.random_normal([in_size,out_size]))
    #构建偏置
    biases = tf.Variable(tf.zeros([1,out_size]) + 0.1)
    #矩阵相乘
    Wx_plus_b = tf.matmul(inputs,weights) + biases
    if activation_function is None:
        outputs = Wx_plus_b
    else:
        outputs = activation_function(Wx_plus_b)
    return outputs,weights,biases #得到输出数据

# 构建隐藏层
h1,w1,b1 = add_layer(xs,1,20,activation_function=tf.nn.relu)

# 构建输出层
prediction,w2,b2 = add_layer(h1,20,1)

# 构建损失函数：计算输出层与真实值间的误差，对二者差的平方求和再取平均，得到损失函数
loss = tf.reduce_mean(tf.reduce_sum(tf.square(ys-prediction),reduction_indices=[1]))

train_step = tf.train.GradientDescentOptimizer(0.1).minimize(loss)

# 训练1000次，每50次输出损失值
init = tf.global_variables_initializer() #初始化所有变量
sess = tf.Session()
sess.run(init)

for i in range(10000):
    sess.run(train_step,feed_dict={xs:x_data,ys:y_data})
    if i % 500 ==0:
        print sess.run(loss,feed_dict={xs:x_data,ys:y_data})
        l1 = np.array([[0.5],[0.1],[0.3]]).dot(sess.run(w1))+sess.run(b1)
        l1 = sess.run(tf.nn.relu(l1))
        l2 = l1.dot(sess.run(w2))+sess.run(b2)
        print '[[0.5]]',l2
        #print sess.run(w1),sess.run(b1)
        #print np.array([[0.1]]).dot(sess.run(w1))+
