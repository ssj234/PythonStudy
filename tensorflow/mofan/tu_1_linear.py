# the first tensorflow demo
# linear
import tensorflow as tf
import numpy as np

# 0.create data we want weight is 0.1 and biases is 0.3
x_data = np.random.rand(100).astype(np.float32)
y_data = x_data * 0.1 + 0.3

### create tensorflow structure start ###
# 1.define weights init random generated between -1.0 and 1.0,one dim
Weights = tf.Variable(tf.random_uniform([1],-1.0,1.0))
biases = tf.Variable(tf.zeros([1])) # init with 0 , one dim

# 2.define how to calc the Y
y = Weights * x_data + biases

# 3.define the loss function
loss = tf.reduce_mean(tf.square(y-y_data))

# 4.define the optimizer
optimizer = tf.train.GradientDescentOptimizer(0.5)
train = optimizer.minimize(loss)

### create tensorflow structure end ###

# 5.init all variables
init = tf.initialize_all_variables()

# 6.create session and run
sess = tf.Session()
sess.run(init)

for step in range(201):
    sess.run(train)
    if step % 20 == 0:
        print step,sess.run(Weights),sess.run(biases)
