#练习：使用tf搭建一个简单的两层网络
import numpy as np
import tensorflow as tf
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

# x_data = np.linspace(-1,1,3000)[:,np.newaxis] #每个x_data有一个特征
# noise = np.random.normal(0, 0.05, x_data.shape)
# y_data = np.square(x_data) - 0.5 + noise
# output_dim = y_data.shape[1]
#
# Xs = tf.placeholder(tf.float32, [None,1])
# ys = tf.placeholder(tf.float32, [None,1])
# l1 = add_layer(Xs, 10, activation=tf.nn.relu)
# prediction = add_layer(l1, output_dim)
#
# loss = tf.reduce_mean(tf.reduce_sum(tf.square(prediction - ys),axis=1))
# train_step = tf.train.GradientDescentOptimizer(0.1).minimize(loss)
# init = tf.global_variables_initializer()
#
# sess = tf.Session()
# sess.run(init)
# for iter in range(4000):
#     X_batch = []
#     y_batch = []
#     indices = np.random.choice(x_data.shape[0], 200, replace = True)
#     X_batch = x_data[indices]
#     y_batch = y_data[indices]
#     sess.run(train_step, feed_dict={Xs: X_batch, ys: y_batch})
#     if iter%200 == 0:
#         print('iteration: {0}/4000  loss = {1} '.format(iter, sess.run(loss, feed_dict={Xs: X_batch, ys: y_batch})))

 ############################################ 将上面的程序扩展 ##########################
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
digits = load_digits()
digits_x = digits.data
digits_y = digits.target
batch_size = 200
train_X, test_X, train_y, test_y = train_test_split(digits_x, digits_y)
clean_img = train_X
noise = np.random.normal(0, 0.05, size=train_X.shape)
train_X = noise + clean_img

def inference(X, output_dim, activation_list):
    l1 = add_layer(X, 100, activation=activation_list[0])
    residual = add_layer(l1, output_dim=input_dim, activation= activation_list[1])
    return X - residual

input_dim = train_X.shape[1]
X = tf.placeholder(dtype=tf.float32, shape=[None, input_dim],name='input_tensor')
y = tf.placeholder(dtype=tf.float32, shape=[None, input_dim],name='label_tensor')
activation_list = [None, tf.nn.relu]
prediction = inference(X,output_dim=input_dim, activation_list=activation_list)
loss = 1/ batch_size * tf.reduce_sum(tf.square(prediction - y) )
train_step = tf.train.GradientDescentOptimizer(0.01).minimize(loss)
init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)
for iter in range(4000):
    X_batch = []
    y_batch = []
    indices = np.array(np.random.choice(clean_img.shape[0], batch_size, replace = True))
    X_batch = train_X[indices]
    y_batch = clean_img[indices]
    sess.run(train_step, feed_dict={X: X_batch, y: y_batch})
    if iter%200 == 0:
        print('iteration: {0}/4000  loss = {1} '.format(iter, sess.run(loss, feed_dict={X: X_batch, y: y_batch})))


# def inference():


