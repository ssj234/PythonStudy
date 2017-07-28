#!/usr/bin/python
# -*- coding: UTF-8 -*-

import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
from sklearn.cross_validation import train_test_split
#
mnist = input_data.read_data_sets('MNIST_data',one_hot=True)

# hyper-param
learning_rate = 0.001
training_iters =  100000
batch_size = 128
display_step = 10

n_inputs = 28 # 输入数据是28*28 这个是列
n_steps = 28  # 28行
n_hidden_units = 128 # 神经元 hidden layer 
n_classes = 10 # 分为10类 0-9

# define placeholder for input to ann
x = tf.placeholder(tf.float32,[None,n_steps,n_inputs]) # 784 = 28 * 28
y = tf.placeholder(tf.float32,[None,n_classes])

# define weights

weights = {
  # (28*128)
 'in':tf.Variable(tf.random_normal([n_inputs,n_hidden_units])),
  # (128*10)
 'out':tf.Variable(tf.random_normal([n_hidden_units,n_classes]))
}
biases = {
  # 128
 'in':tf.Variable(tf.constant(0.1,shape=[n_hidden_units,])),
  # 10
 'out':tf.Variable(tf.constant(0.1,shape=[n_classes,]))
}

def RNN(X,weights,biases):
    # hidden layer for input to cell
    # X n_sample * 28 * 28  => 128*28  28  因为要1行1行的输入 1行有28列
    X = tf.reshape(X,[-1,n_inputs])    
    # n_sample*28 * 28  28*128 => n_sample*28 * 128 
    X_in = tf.matmul(X,weights['in'] + biases['in'])
    # n_sample*28 * 128  =>  n_sample * 28 * 128
    X_in = X_in.reshape(X_in,[-1,n_steps,n_hidden_units])
    # cell  param: 隐藏层unit,forget-gate的bias,每一步计算的结果是一个state，是否保存为元组
    lstm_cell = tf.nn.rnn_cell.BasicLSTMCell(n_hidden_units,forget_bias=1.0,state_is_tuple=True)
    _init_state = lstm_cell.zero_state(batch_size,dtype=tf.float32)
    
    outputs,states = tf.nn.dynamic_rnn(lstm_cell,X_in,initial_state=_init_state,time_major=False)
    results = tf.matmul(states[1],weights['out'] + biases['out'])

    #outputs = tf.unpack(tf.transpose(outputs,[1,0,2]))
    #results = tf.matmul(outputs[-1],weights['out'] + biases['out'])
    # hidden layer for output as the final result
    return results


pred = RNN(x,weights,biases)
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(pred,y))
train_step = tf.train.AdamOptimizer(learning_rate).minimize(cost)

correct_pred = tf.equal(tf.argmax(pred,1),tf.argmax(y,1))
accuracy = tf.reduce_mean(tf.cast(correct_pred,tf.float32))


sess = tf.Session()

sess.run(tf.initialize_all_variables())

train_x,_,train_y,_= train_test_split(mnist.train.images,mnist.train.labels,test_size=0.85,random_state=33)
test_x,_,test_y,_= train_test_split(mnist.test.images,mnist.test.labels,test_size=0.75,random_state=33)

def next_batch(i,count=100):
    size = train_x.shape[0]
    maxcount = size / count
    i = i % maxcount
    if i == 0: 
        b = 0
        e = count
    else:
        b = i * count
        e = (i+1) * count
    if e > size:
        e = size - 1
    return train_x[b:e],train_y[b:e]

for i in range(1000):
    batch_xs,batch_ys =  next_batch(i) 
    sess.run(train_step,feed_dict={x:batch_xs,y:batch_ys,keep_prob:0.5})
    #print 'step'
    if i % 50 == 0:
        print 'accuracy:',sess.run(accuracy,feed_dict={x:batch_xs,y:batch_ys,keep_prob:0.5})
