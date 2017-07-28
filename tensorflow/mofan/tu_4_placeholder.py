# use placeholder
# placeholder can be replaced by numer or matrix
import tensorflow as tf
import numpy as np

p1 = tf.placeholder(tf.int32)
p2 = tf.placeholder(tf.int32)

add = tf.add(p1,p2)

with tf.Session() as sess:
    # opt: 55
    print sess.run(add,feed_dict={p1:23,p2:32})
    # opt :[11,12,13]
    print sess.run(add,feed_dict={p1:[1,2,3],p2:[10,10,10]})

