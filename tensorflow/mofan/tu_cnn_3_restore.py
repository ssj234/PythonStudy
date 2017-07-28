#!/usr/bin/python
# -*- coding: UTF-8 -*-

import tensorflow as tf
import numpy as np

# 定义相同的shape和dtype

W = tf.Variable(np.arange(6).reshape((2,3)),dtype=tf.float32,name='weights')
b = tf.Variable(np.arange(3).reshape((1,3)),dtype=tf.float32,name='biases')


# 不需要init步骤
saver = tf.train.Saver()
with tf.Session() as sess:
    saver.restore(sess,'myNet/save_net.ckpt')
    print 'weights:',sess.run(W)
    print 'biases:',sess.run(b)

