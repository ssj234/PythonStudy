#!/usr/bin/python
# -*- coding: UTF-8 -*-

import numpy as np

# Input
X = np.array([[0,0,1],
			  [0,1,1],
			  [1,0,1],
			  [1,1,1]])
# Output
y = np.array([0,1,1,0]).reshape(-1,1)

# Weight
np.random.seed(1)

W0 = 2*np.random.random((3,4))-1
W1 = 2*np.random.random((4,1))-1

# Nonlinear function
def sigmoid(X,derive=False):
	if not derive:
		return 1/(1 + np.exp(-X))
	else:
		return X * (1 - X) # 这是sigmoid的导数

# Training
training_times = 100
for i in range(training_times):
	# Layer0
	A0 = np.dot(X,W0)
	Z0 = sigmoid(A0)

	# Layer1
	A1 = np.dot(Z0,W1)
	_y = Z1 = sigmoid(A1)

	cost = (_y-y)  # cost = (y-_y)**2/2

	print 'Cost ',np.mean(np.abs(cost))

	# Calc delta
	delta_A1 = cost * sigmoid(Z1,derive=True)
	delta_W1 = np.dot(Z0.T,delta_A1)
	delta_A0 = np.dot(delta_A1,W1.T) * sigmoid(Z0,derive=True)
	delta_W0 = np.dot(X.T, delta_A0)

	# Update
	rate = 0.1
	W1 = rate * delta_W1
	W0 = rate * delta_W0
else:
	print 'Output:',_y