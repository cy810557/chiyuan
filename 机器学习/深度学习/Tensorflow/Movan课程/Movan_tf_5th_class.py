# 本节学习文件的存储和读取
import tensorflow as tf

# W = tf.Variable([[1,2,3],[4,4,6]],dtype=tf.float32,name='weigths')
# b = tf.Variable([[1,2,3]], dtype=tf.float32,name = 'biases')
#
# saver = tf.train.Saver()
# init = tf.global_variables_initializer()
# with tf.Session() as sess:
#     sess.run(init)
#     save_path = saver.save(sess, 'my_net/save_1_net.ckpt')
#     print('save path to :',save_path)

## 读取变量：注意要保证读取的变量与占位变量有相同shape和dtype
import numpy as np
W = tf.Variable(np.arange(6).reshape((2,3)),dtype=tf.float32)
b = tf.Variable(np.arange(3).reshape(1,3),dtype=tf.float32)
saver = tf.train.Saver()
with tf.Session() as sess:
    saver.restore(sess,'my_net/save_1_net.ckpt')
    print(sess.run(W))
    print()
    print(sess.run(b))