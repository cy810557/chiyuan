# 这个可以看做是一个纯visualize脚本，作为脚本三的预备
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

def add_layer(input, output_dim, activation = None):
    with tf.name_scope('Layer'):
        input_dim = input.get_shape().as_list()[1]
        with tf.name_scope('Varibles'):
            W = tf.Variable(tf.random_normal([input_dim, output_dim]),name='Weights')

            # b = tf.zeros(output_dim) + 0.05 错误写法
            b = tf.Variable(tf.zeros(output_dim) + 0.5,name='Biases')
        if activation is None:
            output = tf.matmul(input, W)+b
        else:
            output = activation(tf.matmul(input, W)+b)
        return output

x_data = np.linspace(-1,1,3000)[:,np.newaxis] #每个x_data有一个特征
noise = np.random.normal(0, 0.05, x_data.shape)
y_data = 5*np.sin(2*x_data) - 0.5 + noise*10

output_dim = y_data.shape[1]

with tf.name_scope('Inputs'):
    Xs = tf.placeholder(tf.float32, [None,1],name='X')
    ys = tf.placeholder(tf.float32, [None,1],name = 'y')
l1 = add_layer(Xs, 10, activation=tf.nn.relu)
prediction = add_layer(l1, output_dim)
with tf.name_scope('Loss'):
    loss = tf.reduce_mean(tf.reduce_sum(tf.square(prediction - ys),axis=1))
with tf.name_scope('Train'):  #重要！！！
    train_step = tf.train.GradientDescentOptimizer(0.1).minimize(loss)
init = tf.global_variables_initializer()
sess = tf.Session(config=tf.ConfigProto(device_count={'cpu':0})) #config=tf.ConfigProto(device_count={'cpu':0})
writer = tf.summary.FileWriter('tf_visualizer/',sess.graph)
sess.run(init)



