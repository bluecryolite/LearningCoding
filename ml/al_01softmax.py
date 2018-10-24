#!~/anaconda3/bin/python
"""
Softmax, 归一化指数函数。把K维向量转成一个每一个元素都在(0, 1)之间并且其和为1的另一个向量
参见：https://zh.wikipedia.org/wiki/Softmax%E5%87%BD%E6%95%B0
"""
import math
import numpy as np
import tensorflow as tf

print(tf.__version__);
print(tf.__path__);

# 方法一：math
z = [1.0, 2.0, 3.0, 4.0, 1.0, 2.0, 3.0]
z_exp = [math.exp(i) for i in z]
sum_z_exp = sum(z_exp)
softmax = [round(i / sum_z_exp, 4) for i in z_exp]
print("math")
print(softmax)

# 方法二：numpy
z1 = np.asarray(z)
z_exp1 = np.exp(z1)
softmax1 = z_exp1 / sum(z_exp1)
print("numpy")
print(np.round(softmax1, 4))

# 方法三：tensorflow
config = tf.ConfigProto()
config.intra_op_parallelism_threads = 2
config.inter_op_parallelism_threads = 2
sess = tf.InteractiveSession()
softmax2 = tf.nn.softmax(z1)
print("tensorflow")
print(np.round(sess.run(softmax2), 4))
sess.close()
