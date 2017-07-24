# show the use of session

import tensorflow as tf
import numpy as np

matrix1 = tf.constant([[1,2]])
matrix2 = tf.constant([[3],[4]])

product = tf.matmul(matrix1,matrix2)


# method 1
# sess = tf.Session()

# method 2
with tf.Session() as sess:
    result = sess.run(product)
    print result
