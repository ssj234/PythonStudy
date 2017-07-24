# how to use variable

# 
import tensorflow as tf
import numpy as np

# define a variable named 'counter'
# ouput:counter:0
state = tf.Variable(0,name='counter')
print state.name

# define constant
one = tf.constant(1)
# new-value is an instruction:state+one
new_value = tf.add(state , one)
# update is an inst: set new_value to state
update = tf.assign(state,new_value)

# if you use tf.Variable,must write next line to init all variables
init = tf.initialize_all_variables()

with tf.Session() as sess:
    sess.run(init)
    print sess.run(new_value) # just run state + one opt:1
    print sess.run(new_value) # opt:1
    for _ in range(3):
        sess.run(update) # update is an instruction not variable
        print sess.run(state)
