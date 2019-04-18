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
# 7.1 多层神经网络，线性多分类
# ======================================================================================
# 笔记
#
# ======================================================================================
import tensorflow as tf
from absl import app, flags
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as plt_color

default_model_dir = os.path.abspath('./model_tf_act_04/')

flags.DEFINE_integer('verbose', 1, 'How much debug info to print.')
flags.DEFINE_string('model_dir', default_model_dir, 'Path to saving model files.')
flags.DEFINE_float('learning_rate', 0.01, '学习率')
flags.DEFINE_integer('batch_size', 100, '批次大小')
flags.DEFINE_integer('epochs', 50, '迭代次数')
flags.DEFINE_bool('draw', True, '是否展示图形')

FLAGS = flags.FLAGS


def generate_data():
    sample_size = 2000
    np.random.seed(10)
    num_classes = 3
    mean = np.random.randn(2)
    cov = np.eye(2)
    samples_per_class = int(sample_size / 2)
    diff = [[3.0], [3.0, -1]]

    x0 = np.random.multivariate_normal(mean, cov, samples_per_class)
    y0 = np.zeros(samples_per_class)

    for ci, d in enumerate(diff):
        x1 = np.random.multivariate_normal(mean + d, cov, samples_per_class)
        y1 = (ci + 1) * np.ones(samples_per_class)

        x0 = np.concatenate((x0, x1))
        y0 = np.concatenate((y0, y1))

    class_ind = [y0 == class_number for class_number in range(num_classes)]
    y0 = np.asanyarray(np.stack(class_ind, axis=1), dtype=np.float32)

    x, y = x0, y0

    return x, y


def draw_data(train_x, train_y, line_x, line_y, line_plane):

    aa = [np.argmax(y0) for y0 in train_y]
    colors = ['r' if y0 == 0 else 'b' if y0 == 1 else 'y' for y0 in aa[:]]

    plt.scatter(train_x[:, 0], train_x[:, 1], c=colors)
    plt.xlabel("Scaled age (in yrs)")
    plt.ylabel("Tumor size (in cm)")

    """
    labels = ["First line", "Second line", "Third line"]
    index = 0
    for line_item_y in line_y:
        plt.plot(line_x, line_item_y, label=labels[index], lw=len(labels) - index)
        index = index + 1
    """

    cmap = plt_color.ListedColormap([plt_color.colorConverter.to_rgba('r', alpha=0.30),
                                     plt_color.colorConverter.to_rgba('b', alpha=0.30),
                                     plt_color.colorConverter.to_rgba('y', alpha=0.30)])
    plt.contourf(line_x, line_y, line_plane, cmap=cmap)
    plt.show()


class Model(object):

    def __init__(self, input_dim, lab_dim, learning_rate):
        self.input_feature = tf.placeholder(tf.float32, [None, input_dim])
        self.input_labels = tf.placeholder(tf.float32, [None, lab_dim])

        self.w = tf.Variable(tf.random_normal([input_dim, lab_dim]), name="weight")
        self.b = tf.Variable(tf.zeros([lab_dim]), name="bias")

        self.output = tf.matmul(self.input_feature, self.w) + self.b

        self.a1 = tf.argmax(tf.nn.softmax(self.output), axis=1)
        self.b1 = tf.argmax(self.input_labels, axis=1)

        self.err = tf.count_nonzero(self.a1 - self.b1)
        cross_entropy = tf.nn.softmax_cross_entropy_with_logits(labels=self.input_labels, logits=self.output)
        self.loss = tf.reduce_mean(cross_entropy)
        optimizer = tf.train.AdamOptimizer(learning_rate)
        self.train = optimizer.minimize(self.loss)

    def training(self, features, labels, epochs, batch_size):
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())

            for epoch in range(epochs):
                sumerr = 0
                for i in range(np.int32(len(labels) / batch_size)):
                    x1 = features[i * batch_size: (i + 1) * batch_size, :]
                    y1 = labels[i * batch_size: (i + 1) * batch_size, :]
                    _, lossval, outputval, errval = sess.run([self.train, self.loss, self.output, self.err],
                                                             feed_dict={self.input_feature: x1, self.input_labels: y1})
                    sumerr = sumerr + errval / batch_size

                print("Epoch:", '%04d' % (epoch + 1), "cost=", "{:.9f}".format(lossval), "err=", sumerr / batch_size)

            nb_of_xs = 200
            x = np.linspace(-1, 8, nb_of_xs)
            xx, yy = np.meshgrid(x, x)

            classification_plane = np.zeros((nb_of_xs, nb_of_xs))
            for i in range(nb_of_xs):
                for j in range(nb_of_xs):
                    classification_plane[i, j] = sess.run(self.a1,
                                                          feed_dict={self.input_feature: [[xx[i, j], yy[i, j]]]})

            # y = [-x * (sess.run(self.w)[0][i] / sess.run(self.w)[1][i]) - sess.run(self.b[i]) / sess.run(self.w)[1][i]
            #     for i in range(labels.shape[1])]

            return xx, yy, classification_plane


def main(argv):
    """ Entry point for running one selfplay game.
    :param argv
    """
    del argv  # Unused

    x, y = generate_data()
    train_model = Model(2, 3, 0.01)
    line_x, line_y, line_plane = train_model.training(x, y, 50, 25)
    draw_data(x, y, line_x, line_y, line_plane)


if __name__ == '__main__':
    app.run(main)
