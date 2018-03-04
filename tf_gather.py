# -*- coding: utf-8 -*-
"""
Created on Thu Mar  1 17:23:36 2018

@author: lankuohsing
"""
# In[]
import tensorflow as tf

temp = tf.constant ([[[1,2,3,4],
                      [5,6,7,8],
                      [9,10,11,12]],
                      [[-1,-2,-3,-4],
                      [-5,-6,-7,-8],
                      [-9,-10,-11,-12]]])

temp2 = tf.gather(temp,1)

with tf.Session() as sess:

    print( sess.run(temp))
    print( sess.run(temp2))
# In[]
import numpy as np
input_size=1

seq = [np.array(raw_seq[i * input_size: (i + 1) * input_size]) for i in range(len(raw_seq) // input_size)]
# In[]
seq1 = [seq[0] / seq[0][0] - 1.0] + [
              curr / seq[i][-1] - 1.0 for i, curr in enumerate(seq[1:])]
# In[]
X= np.array([seq[i: i + num_steps] for i in range(len(seq) - num_steps)])
y = np.array([seq[i + num_steps] for i in range(len(seq) - num_steps)])
# In[]
a=[]
for i in range(10):
    a.append([1,2,3])
# In[]
b=[a[i][0] for i in range(10)]
print(b)