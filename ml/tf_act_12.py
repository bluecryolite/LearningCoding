# Copyright 2015 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the 'License');
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an 'AS IS' BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ======================================================================================

# 《深度学习之TensorFlow：入门、原理和进阶实战》.
# 9.4.2 使用动态RNN处理变长序列
# ======================================================================================
# 笔记
#
#
# ======================================================================================
import tensorflow as tf
import numpy as np

tf.reset_default_graph()

x = np.random.randn(2, 4, 5)

x[1, 1:] = 0
seq_lengths = [4, 1]

cell = tf.contrib.rnn.BasicLSTMCell(num_units=3, state_is_tuple=True)
gru = tf.contrib.rnn.GRUCell(3)

outputs, last_states = tf.nn.dynamic_rnn(cell, x, seq_lengths, dtype=tf.float64)
gruoutputs, grulast_states = tf.nn.dynamic_rnn(gru, x, seq_lengths, dtype=tf.float64)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    print("x:\n{0}".format(x))
    print("\n\n")

    result, sta, gruout, grusta = sess.run([outputs, last_states, gruoutputs, grulast_states])

    print("全序列:\n{0}".format(result[0]))
    print("短序列:\n{0}".format(result[1]))
    print("LSTM的状态: {0}, \n{1}\n\n{2}".format(len(sta), sta[1], sta[0]))

    print("GRU的短序列:\n{0}\n\n{1}".format(gruout[1], gruout[0]))
    print("GRU的状态: {0}, \n{1}\n\n{2}".format(len(grusta), grusta[1], grusta[0]))

exit(0)
