import tensorflow as tf
import numpy as np
# str_001 = np.array(tf.fill([10,3],3)).reshape(3,10)
str_002 = tf.fill([10,3],3)
zz = tf.constant([1,1,2],shape=[3,1]).shape
mm = tf.zeros(shape=[10,5])
nn = tf.ones_like(mm)
sess = tf.compat.v1.Session()
ty = tf.Variable(tf.constant(-2, shape=[10, 10]))

sess.run(str_002)
fill_var = tf.Variable(tf.fill([10,1],-1))
print(sess.run(str_002))
print(zz)
print(sess.run(nn))
print(sess.run(mm))
print(sess.run(ty.initial_value))
print(sess.run(fill_var.initial_value))



run_cnorm = tf.random.truncated_normal([10,10],mean=0,stddev=0.1)
new_run_cnorm = sess.run(run_cnorm)
print(new_run_cnorm)
pp = tf.constant([1,2,3,4,5,6,7,8,9],shape=[9,1])
new_pp = tf.random.shuffle(pp)
print(sess.run(new_pp))


print('Tensorflow version: {}'.format(tf.__version__))