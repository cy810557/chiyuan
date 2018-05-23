#练习：这一脚本内容较多。请及时复习
#该脚本练习使用tensorboard可视化工具：包括：绘制计算图，loss曲线，参数直方图
# 问题：W是在函数中初始化的，那么W是如何更新的？？？？？？？？？？？？？？？？？？？？？？是否可以对最后训练出的w进行可视化？
# 问题2：accuracy的定义也在一个函数中（注意这个函数是在会话建立之后调用的），为什么accuracy的summary没有被加入到tensorboard中？
# 问题3：tensorflow如何进行cross-validation?????????????????????????????/
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
BATCH_SIZE = 100
LEARNING_RATE = 0.005
def add_layer(input, output_dim,layer_name,  activation = None):
    with tf.name_scope(layer_name):
        input_dim = input.get_shape().as_list()[1]
        with tf.name_scope('Weights'):
            W = tf.Variable(tf.random_normal([input_dim, output_dim]),name='Weights')
            tf.summary.histogram(layer_name + '/Weights', W)
        with tf.name_scope('biases'):
            b = tf.Variable(tf.zeros(output_dim) + 0.5,name='Biases')
            tf.summary.histogram(layer_name +'/biases', b)
        if activation is None:
            output = tf.matmul(input, W)+b
        else:
            output = activation(tf.matmul(input, W)+b)
            tf.summary.histogram(layer_name+'/output', output)
        return output

def compute_accuracy(val_Xs,val_ys):  #注意：这个函数是在会话中调用的，且不需要显示的传入一个参数代表会话sess
    # 问题：这里的accuracy能不能像loss一样在tensoborad scalar部分画出？
    #注意：使用全局变量
    global prediction
    y_pre = sess.run(prediction,feed_dict={Xs:val_Xs, ys:val_ys})
    # tf.argmax返回的是最大值所对应的索引
    correct_prediction = tf.equal(tf.argmax(y_pre,1), val_ys)
    with tf.name_scope('accuracy'):
        accuracy = tf.reduce_mean(tf.cast(correct_prediction,tf.float32))
        tf.summary.scalar('accuracy',accuracy)
    result = sess.run(accuracy, feed_dict={Xs:val_Xs, ys:val_ys})
    return result


digits = load_digits()
data = digits.data
label = digits.target
X_train, X_test, y_train, y_test = train_test_split(data, label)
with tf.name_scope('Input'):
    Xs = tf.placeholder(dtype=tf.float32, shape = [None,64] , name='X')
    ys = tf.placeholder(dtype=tf.int32, shape = [None], name = 'y')
l1 = add_layer(Xs, 100, layer_name='layer_1', activation=tf.nn.tanh) #为什么使用relu效果要差很多？
prediction = add_layer(l1, 10, layer_name='layer_2', activation=tf.nn.softmax)

with tf.name_scope('Loss'):
    #注意下一句的错误①：ys 为N x 1, 而 prediction 为 N X 10,故要转换成one-hot形式
    #错误②：在使用cross_entropy = -tf.reduce_sum(y_*tf.log(y))时，y最小值有可能取到0，导致log出错
    # cross_entropy = tf.reduce_mean(-tf.reduce_sum(tf.one_hot(indices=ys, depth=10)*tf.log(prediction), axis=1))
    cross_entropy = tf.reduce_mean(-tf.reduce_sum(tf.one_hot(indices=ys, depth=10) *
                                                  tf.log(tf.clip_by_value(prediction, 1e-10, 1.0)), axis=1))
    tf.summary.scalar('loss',cross_entropy)  #注意：以逗号分隔
with tf.name_scope('Train'):
    train_step = tf.train.AdamOptimizer(LEARNING_RATE).minimize(cross_entropy)
init = tf.global_variables_initializer()
N = X_train.shape[0]
# with tf.device('/gpu:0'):
sess = tf.Session()
merged = tf.summary.merge_all()
writer = tf.summary.FileWriter('tf_visualizer/c1', sess.graph)

sess.run(init)
for iter in range(2000):
    indices = np.array(np.random.choice(N,BATCH_SIZE,replace=True))
    X_batch = X_train[indices]
    y_batch = y_train[indices]
    sess.run(train_step, feed_dict={Xs:X_batch,ys:y_batch})
    if iter%200 == 0:
        print('iteration: {}/2000: '.format(iter),)
        print(' 准确率为：' + repr(compute_accuracy(X_batch, y_batch)),
              'loss值为：'+repr(sess.run(cross_entropy,feed_dict={Xs:X_batch,ys:y_batch})))
        result = sess.run(merged,feed_dict={Xs:X_batch,ys:y_batch})
        writer.add_summary(result,iter)  #注意不要忘记了








