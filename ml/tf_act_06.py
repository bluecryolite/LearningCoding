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
# 7.2 异或
# ======================================================================================
# 笔记
# 190416:
# 震惊，4组数据就训练个模型出来。
# 书上给出的10,000次迭代不行，需要更多万次迭代后，再反复运行。看loss数据，判断是进入了局部最优。
#
# ======================================================================================
import tensorflow as tf
from absl import app
import numpy as np


class Model(object):

    def __init__(self, learning_rate):
        # 输入层节点个数
        self.input_count = 2
        # 隐藏层节点个数
        self.hidden_count = 2
        # 标签节点个数
        self.label_count = 1

        self.x = tf.placeholder(tf.float32, [None, self.input_count])
        self.y = tf.placeholder(tf.float32, [None, self.label_count])

        self.weights = {
            'h1': tf.Variable(tf.truncated_normal([self.input_count, self.hidden_count], stddev=0.1)),
            'h2': tf.Variable(tf.truncated_normal([self.hidden_count, self.label_count], stddev=0.1))
        }

        self.bias = {
            'h1': tf.Variable(tf.zeros([self.hidden_count])),
            'h2': tf.Variable(tf.zeros([self.label_count]))
        }

        self.layer_1 = tf.nn.relu(tf.add(tf.matmul(self.x, self.weights['h1']), self.bias['h1']))
        self.y_pred = tf.nn.sigmoid(tf.add(tf.matmul(self.layer_1, self.weights['h2']), self.bias['h2']))

        self.loss = tf.reduce_mean((self.y_pred - self.y) ** 2)
        self.train_step = tf.train.AdamOptimizer(learning_rate).minimize(self.loss)

    def training(self, features, labels, epochs):
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())

            for epoch in range(epochs):
                _, loss = sess.run([self.train_step, self.loss], feed_dict={self.x: features, self.y: labels})
                if epoch % 10000 == 0:
                    print({epoch, loss})

            print(sess.run(self.y_pred, feed_dict={self.x: features}))
            print(sess.run(self.layer_1, feed_dict={self.x: features}))


def main(argv):
    """ Entry point for running one selfplay game.
    :param argv
    """
    del argv  # Unused

    x = np.array([[0, 0], [0, 1], [1, 0], [1, 1]]).astype('float32')
    y = np.array([[0], [1], [1], [0]]).astype("int16")
    train_model = Model(0.0001)
    train_model.training(x, y, 400000)


if __name__ == '__main__':
    app.run(main)
