#!/usr/bin/python
# -*- coding: UTF-8 -*-
# why we need activation?
# resolve the problem those we cannot use linear
# add layer1 and layer2 ,each layer have weight and biases
# tensorboard --logdir='logs/'
# tf.histogram_summary(layer_name+'/weights',Weights)
# tf.histogram_summary(layer_name+'/biases',biases)
# tf.scalar_summary('loss',loss)
# merged = tf.merge_all_summaries() # write befor filewriter
# writer.add_summary(result,i) # in train loop

import tensorflow as tf
import numpy as np

def add_layer(inputs,in_size,out_size,activation_function=None):
    layer_name = 'Layer %d-%d' % (in_size,out_size)
    with tf.name_scope('layer'):
        with tf.name_scope('weights'):  
    #layer_name = 'Layer %d-%d' % (in_size,out_size)
    #print layer_name
    Weights = tf.Variable(tf.random_normal([in_size,out_size]),name='weights')  # matrix so Upper
  with tf.name_scope('biases'):
     biases = tf.Variable(tf.zeros([1,out_size]) + 0.1,name='biases')  # a list so Lower not zero so add 0.1
  with tf.name_scope('Wx_plus_b'):
    Wx_plus_b = tf.matmul(inputs,Weights) + biases 
    if activation_function is None:
        outputs = Wx_plus_b
    else:
        outputs = activation_function(Wx_plus_b)
    return outputs

# define the train data
x_data = np.linspace(-1,1,300)[:,np.newaxis]
noise = np.random.normal(0,0.05,x_data.shape)
y_data = np.square(x_data) - 0.5 + noise

#
with tf.name_scope('inputs'):
    xs = tf.placeholder(tf.float32,[None,1],name='x_input') #None:多少行都可以
    ys = tf.placeholder(tf.float32,[None,1],name='y_input') 

l1 = add_layer(xs,1,10,activation_function=tf.nn.relu)
prediction = add_layer(l1,10,1,activation_function=None)

loss = tf.reduce_mean(tf.reduce_sum(tf.square(ys - prediction),reduction_indices=[1]))

train_step = tf.train.GradientDescentOptimizer(0.1).minimize(loss) # learning_rate less than 1

init = tf.initialize_all_variables()
sess = tf.Session()
#writer = tf.train.SummaryWriter('logs/',sess.graph)
writer = tf.summary.FileWriter('logs/',sess.graph)
sess.run(init)

for i in range(1000):
    sess.run(train_step,feed_dict={xs:x_data,ys:y_data}) # feed_dict是为了方便小批量训练
    if i % 50 :
        print sess.run(loss,feed_dict={xs:x_data,ys:y_data})
        print sess.run(prediction,feed_dict={xs:np.array([[0.34]])})
        #prinn sess.run()
















