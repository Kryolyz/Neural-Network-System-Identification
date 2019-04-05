# -*- coding: utf-8 -*-
"""
Created on Tue Jan  8 09:20:04 2019

@author: Kryolyz
"""

#Neural Network System Identification

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx

num_ins = 3
num_outs = 2
num_hid = 2
initials = [[1,]]
inputs = tf.placeholder(shape=[None,num_ins], dtype = tf.float32)
labels = tf.placeholder(shape=[None,num_outs], dtype = tf.float32)
w1 = tf.Variable(tf.ones([num_ins,num_hid], tf.float32), name='Weights1')
w2 = tf.Variable(tf.ones([num_hid,num_outs], tf.float32), name='Weights2')
b1 = tf.Variable(tf.zeros([num_hid]), tf.float32, name='Biases1')
b2 = tf.Variable(tf.zeros([num_outs]), tf.float32, name='Biases2')
layer1 = tf.add(tf.matmul(inputs,w1), b1)
outputs = tf.add(tf.matmul(inputs,w1),b1)
loss = tf.losses.mean_squared_error(labels, outputs)
optimizer = tf.train.AdamOptimizer(learning_rate = 0.05)
update = optimizer.minimize(loss)

#Konstanten
K = 0.01
R = 1
J = 0.01
L = 0.5
steps = 1001
epochs = 1000
state = np.zeros([steps,3])
t = np.arange(steps-1)
t1 = np.arange(epochs)

#State Space Matrizen
x = np.array([[0.0],[0.0]])
A = np.array([[0,K/J],[-K/L,-R/L]])
B = np.array([[0],[1/L]])

def step(x,u):
    xd = np.matmul(A, x) + B * u
    return xd

for i in range(steps):
    u = np.random.uniform(-10,10)
    state[i,0:2] = x.reshape(2)
    xd = step(x,u)
    x += xd
    state[i,2] = u

init = tf.global_variables_initializer()
l = np.zeros(epochs)
with tf.Session() as sess:
    sess.run(init)
    feed_dict = {inputs:state[0:1000,0:num_ins], labels:state[1:1001,0:num_outs]}
    for i in range(epochs):
        _,l[i],outs = sess.run([update,loss,outputs], feed_dict=feed_dict)
    weights1,weights2,biases1,biases2 = sess.run([w1,w2,b1,b2])
    print("Gewichte der ersten Schicht: ")
    print(weights1)
#    print("Gewichte der zweiten Schicht: ")
#    print(weights2)
    print("Biases der ersten Schicht: ")
    print(biases1)
#    print("Biases der zweiten Schicht: ")
#    print(biases2)

plt.figure(1)
plt.plot(t,outs - state[1:1001,0:2])
plt.figure(2)
plt.plot(t1,l)

posi = {'N(t)':[2,3],'I(t)':[2,2],'U':[2,1],'N(t+1)':[3,3],'I(t+1)':[3,1]}
posil = {'N(t)':[2,3.15],'I(t)':[2,2.15],'U':[2,1.15],'N(t+1)':[3,3.15],'I(t+1)':[3,1.15]}

plt.figure(3)
AdjM = np.zeros([5,5])
weights1 = np.around(weights1,2)
AdjM[0:3,3:5] = weights1
gm = nx.from_numpy_matrix(AdjM, create_using = nx.DiGraph)
mapping = {0: 'N(t)', 1: 'I(t)', 2: 'U', 3: 'N(t+1)', 4: 'I(t+1)'}
nx.relabel_nodes(gm, mapping=mapping, copy=False)
nx.draw(gm, pos=posi)
nx.draw_networkx_labels(gm, posil)
nx.draw_networkx_edge_labels(gm, posi)













