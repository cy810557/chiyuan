#练习：使用tf搭建一个简单的两层网络
#问题：如果使用mini-batch如何得到整体拟合曲线？
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
def add_layer(input, output_dim, activation = None):
    # input_dim = input.shape[1]
    # input_dim = tf.shape(input)
    input_dim = input.get_shape().as_list()[1]
    W = tf.Variable(tf.random_normal([input_dim, output_dim]))
    # b = tf.zeros(output_dim) + 0.05
    b = tf.Variable(tf.zeros(output_dim) + 0.5)
    if activation is None:
        output = tf.matmul(input, W)+b
    else:
        output = activation(tf.matmul(input, W)+b)
    return output

x_data = np.linspace(-1,1,3000)[:,np.newaxis] #每个x_data有一个特征
noise = np.random.normal(0, 0.05, x_data.shape)
y_data = 5*np.sin(2*x_data) - 0.5 + noise*10

fig = plt.figure(figsize=(10,5))
ax = fig.add_subplot(1,1,1)
ax.scatter(x_data, y_data)
plt.ion() #打开交互模式，show之后程序不暂停
plt.show() #注意：如果只用plt.show函数，会使程序暂停
output_dim = y_data.shape[1]


Xs = tf.placeholder(tf.float32, [None,1])
ys = tf.placeholder(tf.float32, [None,1])
l1 = add_layer(Xs, 10, activation=tf.nn.relu)
prediction = add_layer(l1, output_dim)

loss = tf.reduce_mean(tf.reduce_sum(tf.square(prediction - ys),axis=1))
train_step = tf.train.GradientDescentOptimizer(0.1).minimize(loss)
init = tf.global_variables_initializer()

sess = tf.Session(config=tf.ConfigProto(device_count={'cpu':0})) #config=tf.ConfigProto(device_count={'cpu':0})
sess.run(init)
for i in range(2000): #注意：这里如果不用try-catch 结构的话画不出红色拟合线
    sess.run(train_step, feed_dict={Xs:x_data, ys:y_data})
    if i%100 ==0:
        try:
            ax.lines.remove(lines[0])
        except:
            pass
        prediction_value = sess.run(prediction, feed_dict={Xs: x_data})
        lines = ax.plot(x_data, prediction_value,'r-',lw = 5)
        plt.pause(0.05)

