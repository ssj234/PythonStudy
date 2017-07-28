
#
import tensorflow as tf

sess = tf.Session()
print 'test constant'
a = tf.constant(2)
b = tf.constant(3)

c = a + b
d = a * b

#sess = tf.Session()
print sess.run(c)
print sess.run(d)


print 'test placeholder'
a = tf.placeholder(tf.int32)
b = tf.placeholder(tf.int32)

add = tf.add(a,b)
mul = tf.multiply(a,b)
print sess.run(add,feed_dict={a:2,b:3})
print sess.run(mul,feed_dict={a:2,b:3})


print 'test matrix'
matrix1 = tf.constant([[3.,3.]])
matrix2 = tf.constant([[2.],[2.]])
product = tf.matmul(matrix2,matrix1)
print sess.run(product)

print 'test matrix by Variable'

mat1 = tf.Variable(tf.random_normal([3,2]))
mat2 = tf.Variable(tf.random_normal([2,3]))
product = tf.matmul(mat1,mat2)

m1 = [[1,3],[2,1],[0,5]]
m2 = [[3,2,1],[1,2,3]]

print sess.run(product,feed_dict={mat1:m1,mat2:m2})
