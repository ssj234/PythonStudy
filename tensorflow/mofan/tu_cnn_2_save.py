#!/usr/bin/python
# -*- coding: UTF-8 -*-

import tensorflow as tf

# Save to file
# 记得需要定义相同的dtype，否则恢复的时候会出错
W = tf.Variable([[1,2,3],[3,4,5]],dtype=tf.float32,name='weights')
b = tf.Variable([[1,2,3]],dtype=tf.float32,name='biases')

init = tf.initialize_all_variables()

saver = tf.train.Saver() # create saver

sess = tf.Session()
sess.run(init)

# save
save_path = saver.save(sess,'myNet/save_net.ckpt')  # save
print 'Save to path:',save_path
