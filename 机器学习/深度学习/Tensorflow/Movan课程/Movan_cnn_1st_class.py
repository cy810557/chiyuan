# 这节课尝试使用tf搭建一个两层的cnn
LEARNING_RATE = 1e-4
ITERATION = 3000
BATCH_SIZE = 100
HIDDEN_SIZE = 1024
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets('MNIST_data',one_hot=True)

def weight_variable(shape, name):
    initial = tf.truncated_normal(shape,stddev=0.1, name = name)
    return tf.Variable(initial)

def bias_variable(shape, name):
    initial = tf.constant(0.1, shape=shape, name = name)
    return tf.Variable(initial)

def conv2d(x, W, name):
    return tf.nn.conv2d(x, W, strides=[1,1,1,1],padding='SAME', name = name)


def max_pool_2x2(x, name):
    return tf.nn.max_pool(x,ksize = [1,2,2,1],strides=[1,2,2,1],padding='SAME', name = name)

# 定义placeholder
with tf.name_scope('Input'):
    xs = tf.placeholder(tf.float32, shape=[None,784]) #28x28
    ys = tf.placeholder(tf.float32, shape=[None,10])
    keep_prob = tf.placeholder(tf.float32)

# 注意：先将input reshape成4-D 以供卷积层使用
x_image = tf.reshape(xs, shape=[-1,28,28,1])

## conv1 layer ##
with tf.name_scope('Conv_L1'):
    W_conv1 = weight_variable([5,5,1,32], name = 'W_conv1')
    b_conv1 = bias_variable([32], name = 'b_conv1')
    h_conv1 = conv2d(x_image,W_conv1, 'Conv_1')+b_conv1
    h_pool1 = max_pool_2x2(tf.nn.relu(h_conv1), 'pool_1') #[N, 14, 14, 32]
## conv2 layer ##
with tf.name_scope('Conv_L2'):
    W_conv2 = weight_variable([5,5,32,64], name = 'W_conv2')
    b_conv2 = bias_variable([64], name='b_conv2')
    h_conv2 =conv2d(h_pool1, W_conv2, name = 'Conv_2')+b_conv2
    h_pool2 = max_pool_2x2(tf.nn.relu(h_conv2),name = 'pool_2') #[N, 7, 7, 64]

#注意：把卷积层输出flat为二维，以供fc层使用
h_pool2_flat = tf.reshape(h_pool2,[-1, 7*7*64])
## fc1   layer ##
with tf.name_scope('FC_L1'):
    W_fc1 = weight_variable([7*7*64, HIDDEN_SIZE], name='W_fc1')
    b_fc1 = bias_variable([HIDDEN_SIZE], name='bias_fc1')
    h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat,W_fc1)+b_fc1, name='relu_activation')

## fc2   layer ##
with tf.name_scope('FC_L2'):
    W_fc2 = weight_variable([HIDDEN_SIZE, 10], name='W_fc2')
    b_fc2 = bias_variable([10], name='b_fc2')
    prediction = tf.nn.softmax(tf.matmul(h_fc1, W_fc2) + b_fc2, name='softmax')
with tf.name_scope('Loss'):
    cross_entropy = tf.reduce_mean(-tf.reduce_sum(ys * tf.log(prediction),axis=1))
    tf.summary.scalar('loss', cross_entropy)

with tf.name_scope('Train'):
    train_step = tf.train.AdamOptimizer(LEARNING_RATE).minimize(cross_entropy)

correct_prediction = tf.equal(tf.argmax(prediction,axis=1), tf.argmax(ys,axis=1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction,tf.float32))
init = tf.global_variables_initializer()
sess = tf.InteractiveSession()
merged = tf.summary.merge_all()
# writer = tf.summary.FileWriter('tf_visualizer/tempt_cnn', sess.graph)
train_writer = tf.summary.FileWriter('tf_visualizer/tempt_cnn/train', sess.graph)
test_writer = tf.summary.FileWriter('tf_visualizer/tempt_cnn/test', sess.graph)

sess.run(init)
for iter in range(ITERATION):
    batch_x, batch_y = mnist.train.next_batch(BATCH_SIZE)
    sess.run(train_step, feed_dict={xs:batch_x ,ys:batch_y})
    if iter % 200 == 0:
        # train_accuracy = sess.run(merged, feed_dict ={xs: batch_x, ys: batch_y})
        # test_accuracy = sess.run(merged, feed_dict = {xs: mnist.test.images[1:1000], ys: mnist.test.labels[1:1000]})
        train_accuracy = accuracy.eval(feed_dict ={xs: batch_x, ys: batch_y})
        test_accuracy = accuracy.eval(feed_dict = {xs: mnist.test.images[1:1000], ys: mnist.test.labels[1:1000]})
        print('test_accuracy：{0}/{1}: {3:3f}  train_accuracy：{0}/{1}: {2:3f}'.format(iter, ITERATION,train_accuracy,test_accuracy))
        # train_writer.add_summary(train_accuracy, iter)
        # test_writer.add_summary(test_accuracy, iter)
