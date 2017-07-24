import numpy as np
weight=[]
incept =[]
output =3
index = 0
input =2
while index < output:
        tmp = [0.5 for x in range(0,input)]
	weight.append(np.array(tmp))
	incept.append(np.array(tmp))
	index = index +1
print weight
a=np.matrix(weight[0])
b=np.matrix([[1.1],[2.2]])
c=0.5
print a
print ' =\n ',b
print ' =\n ',c
print a*b+c
