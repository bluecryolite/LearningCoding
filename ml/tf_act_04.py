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
# 7.1 多层神经网络，线性单分类
# ======================================================================================
# 笔记
# 190313:
# 原文中有shuffle，不知道怎么实现的
#
# ======================================================================================
import tensorflow as tf
from absl import app, flags
import os
import numpy as np
import matplotlib.pyplot as plt

default_model_dir = os.path.abspath('./model_tf_act_04/')

flags.DEFINE_integer('verbose', 1, 'How much debug info to print.')
flags.DEFINE_string('model_dir', default_model_dir, 'Path to saving model files.')
flags.DEFINE_float('learning_rate', 0.01, '学习率')
flags.DEFINE_integer('batch_size', 100, '批次大小')
flags.DEFINE_integer('epochs', 50, '迭代次数')
flags.DEFINE_bool('draw', True, '是否展示图形')

FLAGS = flags.FLAGS


def generate_data():
    sample_size = 1000
    np.random.seed(10)
    num_classes = 2
    mean = np.random.randn(num_classes)
    # 二维的one-hot ndarray
    cov = np.eye(num_classes)
    samples_per_class = int(sample_size / 2)
    diff = [3.0]

    # 多元正态分布
    x0 = np.random.multivariate_normal(mean, cov, samples_per_class)
    y0 = np.zeros(samples_per_class)

    for ci, d in enumerate(diff):
        # 多元正态分布的平移
        x1 = np.random.multivariate_normal(mean + d, cov, samples_per_class)
        y1 = (ci + 1) * np.ones(samples_per_class)

        x0 = np.concatenate((x0, x1))
        y0 = np.concatenate((y0, y1))

    #random.shuffle(x0, y0)
    x, y = x0, y0

    return x, y


def draw_data(train_x, train_y, line_x, line_y):
    colors = ['r' if y0 == 0 else 'b' for y0 in train_y[:]]

    # 散点图
    plt.scatter(train_x[:, 0], train_x[:, 1], c=colors)
    plt.xlabel("Scaled age (in yrs)")
    plt.ylabel("Tumor size (in cm)")

    # 线段
    plt.plot(line_x, line_y, label='Fitted line')
    plt.legend()

    plt.show()


class Model(object):

    def __init__(self, input_dim, lab_dim, learning_rate):
        self.input_feature = tf.placeholder(tf.float32, [None, input_dim])
        self.input_labels = tf.placeholder(tf.float32, [None, lab_dim])

        self.w = tf.Variable(tf.random_normal([input_dim, lab_dim]), name="weight")
        self.b = tf.Variable(tf.zeros([lab_dim]), name="bias")

        # 对线性模型做sigmoid
        self.output = tf.nn.sigmoid(tf.matmul(self.input_feature, self.w) + self.b)

        # 损失函数为交叉熵
        cross_entropy = -(self.input_labels * tf.log(self.output) + (1 - self.input_labels) * tf.log(1 - self.output))
        self.loss = tf.reduce_mean(cross_entropy)

        # Adam优化器，用于梯度下降
        optimizer = tf.train.AdamOptimizer(learning_rate)
        self.train = optimizer.minimize(self.loss)

        # 用于计算均值方差
        ser = tf.square(self.input_labels - self.output)
        self.err = tf.reduce_mean(ser)

    def training(self, features, labels, epochs, batch_size):
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())

            for epoch in range(epochs):
                sumerr = 0
                for i in range(np.int32(len(labels) / batch_size)):
                    x1 = features[i * batch_size: (i + 1) * batch_size, :]
                    y1 = np.reshape(labels[i * batch_size: (i + 1) * batch_size], [-1, 1])
                    tf.reshape(y1, [-1, 1])
                    _, lossval, outputval, errval = sess.run([self.train, self.loss, self.output, self.err],
                                                             feed_dict={self.input_feature: x1, self.input_labels: y1})
                    sumerr = sumerr + errval

                print("Epoch:", '%04d' % (epoch + 1), "cost=", "{:.9f}".format(lossval), "err=", sumerr / batch_size)

            x = np.linspace(-1, 8, 200)
            y = -x * (sess.run(self.w)[0] / sess.run(self.w)[1]) - sess.run(self.b) / sess.run(self.w)[1]

            return x, y


def main(argv):
    """ Entry point for running one selfplay game.
    :param argv
    """
    del argv  # Unused

    x, y = generate_data()
    train_model = Model(2, 1, 0.01)
    line_x, line_y = train_model.training(x, y, 50, 25)
    draw_data(x, y, line_x, line_y)


if __name__ == '__main__':
    app.run(main)
