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
# 7.4 全连接的优化技巧
# ======================================================================================
# 笔记
# 190419:
# estimate时，如果新建session，则得不到训练出来的模型。在不同方法间共享模型，需要把模型保存为文件
#
# ======================================================================================
import tensorflow as tf
from absl import app, flags
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as plt_color
import time

default_model_dir = os.path.abspath('./model_tf_act_08/')

flags.DEFINE_integer('verbose', 1000, 'How much debug info to print.')
flags.DEFINE_string('model_dir', default_model_dir, 'Path to saving model files.')
flags.DEFINE_float('learning_rate', 0.01, '学习率')
flags.DEFINE_integer('batch_size', 1000, '批次大小')
flags.DEFINE_integer('epochs', 20000, '迭代次数')
flags.DEFINE_bool('draw', True, '是否展示图形')
flags.DEFINE_integer('hidden_count', 200, '隐藏层节点个数')
flags.DEFINE_integer('point_count', 2000, '样本数量')

FLAGS = flags.FLAGS


def generate_data(point_count, rand_seed):
    sample_size = point_count
    np.random.seed(rand_seed)
    # num_classes = 4
    mean = np.random.randn(2)
    cov = np.eye(2)
    samples_per_class = int(sample_size / 2)
    diff = [[4.0, 0], [4.0, 4.0], [0, 4.0]]

    x0 = np.random.multivariate_normal(mean, cov, samples_per_class)
    y0 = np.zeros(samples_per_class)

    for ci, d in enumerate(diff):
        x1 = np.random.multivariate_normal(mean + d, cov, samples_per_class)
        y1 = (ci + 1) * np.ones(samples_per_class)

        x0 = np.concatenate((x0, x1))
        y0 = np.concatenate((y0, y1))

    # class_ind = [y0 == class_number for class_number in range(num_classes)]
    # y0 = np.asanyarray(np.stack(class_ind, axis=1), dtype=np.float32)

    x, y = x0, y0

    return x, y


class Model(object):

    def training(self, features, labels, epochs, batch_size):
        with tf.Session() as sess:
            saver = tf.train.Saver()
            sess.run(tf.global_variables_initializer())

            feature_count = len(features)
            batch_count = int(feature_count / batch_size)
            batch_count = batch_count + 1 if feature_count % batch_size > 0 else batch_count
            for epoch in range(epochs):
                start = 0
                for batch_index in range(batch_count):
                    end = start + batch_size
                    if end > feature_count:
                        end = feature_count
                    _, loss = sess.run([self.train_step, self.loss], feed_dict={self.x: features[start:end],
                                                                                self.y: labels[start:end],
                                                                                self.keep_prob: 0.6})
                    start += batch_size

                if epoch % FLAGS.verbose == 0:
                    print({epoch, loss})

            saver.save(sess, FLAGS.model_dir)
            print(loss)

    def draw(self, train_x, train_y, point_count):
        xr = []
        xb = []

        for (l0, k) in zip(train_y, train_x):
            if l0 == 0.0:
                xr.append(k)
            else:
                xb.append(k)

        xr = np.array(xr)
        xb = np.array(xb)

        plt.scatter(xr[:, 0], xr[:, 1], c='r', marker='+')
        plt.scatter(xb[:, 0], xb[:, 1], c='b', marker='o')

        xx, yy = np.meshgrid(np.linspace(-4, 8, num=point_count), np.linspace(-4, 8, num=point_count))
        points = np.zeros((point_count, point_count))
        with tf.Session() as sess:
            saver = tf.train.Saver()
            saver.restore(sess, FLAGS.model_dir)

            for i in range(point_count):
                for j in range(point_count):
                    points[i, j] = np.around(sess.run(self.y_pred, feed_dict={self.x: [[xx[i, j], yy[i, j]]],
                                                                              self.keep_prob: 1.0}))

            print(sess.run(self.loss, feed_dict={self.x: train_x, self.y: [[m] for m in train_y], self.keep_prob: 1.0}))

        cmap = plt_color.ListedColormap([
            plt_color.colorConverter.to_rgba('r', alpha=0.30),
            plt_color.colorConverter.to_rgba('b', alpha=0.30)
        ])

        plt.contourf(xx, yy, points, cmap=cmap)

        plt.show()

    def __init__(self, learning_rate, hidden_count):
        # 输入层节点个数
        self.input_count = 2
        # 标签节点个数
        self.label_count = 1

        self.x = tf.placeholder(tf.float32, [None, self.input_count])
        self.y = tf.placeholder(tf.float32, [None, self.label_count])

        self.weights = {
            'h1': tf.Variable(tf.truncated_normal([self.input_count, hidden_count], stddev=0.1)),
            'h2': tf.Variable(tf.truncated_normal([hidden_count, self.label_count], stddev=0.1))
        }

        self.bias = {
            'h1': tf.Variable(tf.zeros([hidden_count])),
            'h2': tf.Variable(tf.zeros([self.label_count]))
        }

        self.layer_1 = tf.nn.relu(tf.add(tf.matmul(self.x, self.weights['h1']), self.bias['h1']))

        # dropout
        self.keep_prob = tf.placeholder(tf.float32)
        self.layer_1 = tf.nn.dropout(self.layer_1, keep_prob=self.keep_prob)

        self.y_pred = tf.add(tf.matmul(self.layer_1, self.weights['h2']), self.bias['h2'])
        self.y_pred = tf.maximum(self.y_pred, 0.01 * self.y_pred)

        self.loss = tf.reduce_mean((self.y_pred - self.y) ** 2)

        # L2正则化
        # reg = 0.01
        # self.loss += tf.nn.l2_loss(self.weights['h1']) * reg + tf.nn.l2_loss(self.weights['h2']) * reg

        # 退化学习率
        global_step = tf.Variable(0, trainable=False)
        decaylearning_rate = tf.train.exponential_decay(learning_rate, global_step, 1000, 0.9)
        self.train_step = tf.train.AdamOptimizer(decaylearning_rate).minimize(self.loss, global_step=global_step)


def main(argv):
    """ Entry point for running one selfplay game.
    :param argv
    """
    del argv  # Unused

    rand_seed = int(time.time())
    x, y = generate_data(FLAGS.point_count, rand_seed)
    y = y % 2
    train_model = Model(FLAGS.learning_rate, FLAGS.hidden_count)
    train_model.training(x, [[m] for m in y], FLAGS.epochs, FLAGS.batch_size)
    train_model.draw(x, y, 200)

    # estimate
    estimate_count = 100 if FLAGS.point_count / 2 > 100 else int(FLAGS.point_count / 2)
    x, y = generate_data(estimate_count, rand_seed)
    y = y % 2
    train_model.draw(x, y, 200)


if __name__ == '__main__':
    app.run(main)
