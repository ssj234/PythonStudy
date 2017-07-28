#!/usr/bin/python
# -*- coding: UTF-8 -*-

import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
from sklearn.cross_validation import train_test_split
#
mnist = input_data.read_data_sets('MNIST_data',one_hot=True)


def weight_variable(shape):
    initial = tf.truncated_normal(shape,stddev=0.1) # generate random variable
    return tf.Variable(initial)

def bias_variable(shape):
    initial = tf.constant(0.1,shape=shape) 
    return tf.Variable(initial)

def conv2d(x,W):
    # stride [1,x_mov,y_mov,1]
    # padding VALID SAME
    return tf.nn.conv2d(x,W,strides=[1,1,1,1],padding='SAME') # 2-dim

def max_pool_2x2(x):
    return tf.nn.max_pool(x,ksize=[1,2,2,1],strides=[1,2,2,1],padding='SAME')

def compute_accuracy(v_xs,v_ys):
    global prediction
    y_pre = sess.run(prediction,feed_dict={xs:v_xs,keep_prob:0.5})
    correct_prediction = tf.equal(tf.argmax(y_pre,1),tf.argmax(v_ys,1)) # argmax return the index of the largest value
    accuracy = tf.reduce_mean(tf.cast(correct_prediction,tf.float32)) # cast: cast data type, reduce_mean:calc the average along given reduction_indices
    result = sess.run(accuracy,feed_dict={xs:v_xs,ys:v_ys,keep_prob:1}) 
    return result
    

# define placeholder for input to ann
xs = tf.placeholder(tf.float32,[None,784]) # 784 = 28 * 28
ys = tf.placeholder(tf.float32,[None,10])
keep_prob = tf.placeholder(tf.float32)
x_image = tf.reshape(xs,[-1,28,28,1]) # 黑白照片-1层 RGB-3层

# conv1 layer
W_conv1 = weight_variable([5,5,1,32]) # 5*5的卷积核 称为patch 1：厚度  32:32个卷积核
b_conv1 = bias_variable([32]) # 32个卷积核 32个biases
h_conv1 = tf.nn.relu(conv2d(x_image,W_conv1) + b_conv1) # optsize:28*28*32
h_pool1 = max_pool_2x2(h_conv1) # 池化层 14*14*32

# conv2 layer
W_conv2 = weight_variable([5,5,32,64]) # 5*5的卷积核 称为patch 32：厚度  64:卷积核
b_conv2 = bias_variable([64]) # 64个卷积核 64个biases
h_conv2 = tf.nn.relu(conv2d(h_pool1,W_conv2) + b_conv2) # optsize:14*14*64
h_pool2 = max_pool_2x2(h_conv2) # 池化层 7*7*64

# func1 layer
W_fc1 = weight_variable([7*7*64,512])
b_fc1 = bias_variable([512])
h_pool2_flat = tf.reshape(h_pool2,[-1,7*7*64]) # [n_sampe,7,7,64] -> [n_sample,7*7*64]
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat,W_fc1) + b_fc1)
h_fc1_drop = tf.nn.dropout(h_fc1,keep_prob)

# func2 layer
W_fc2 = weight_variable([512,10])
b_fc2 = bias_variable([10])
prediction = tf.nn.softmax(tf.matmul(h_fc1_drop,W_fc2) + b_fc2)

# loss
cross_entropy = tf.reduce_mean(-tf.reduce_sum(ys * tf.log(prediction),reduction_indices=[1]))

train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)

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
    sess.run(train_step,feed_dict={xs:batch_xs,ys:batch_ys,keep_prob:0.5})
    #print 'step'
    if i % 50 == 0:
        print 'accuracy:',compute_accuracy(test_x,test_y)
