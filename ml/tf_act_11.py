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
# 9.2.2 拟合回声信号 —— 换一种数据组织方式
# ======================================================================================
# 笔记
#
# 190430:
# 回声信号是线性模型，所以直接用全连接也能实现。同样的，加个隐藏层，能加快学习速度并减少特征集大小。
# 创建模型时，用fully_connected方法，比直接写wx + b要慢很多呢。
#
# 190429:
# 直接在generate_data中，生成feature的数据量
#
# ======================================================================================
import tensorflow as tf
from absl import app, flags
import os
import shutil
import numpy as np
import matplotlib.pyplot as plt

default_model_dir = os.path.abspath('./model_tf_act_08/')

flags.DEFINE_integer('verbose', 10, 'How much debug info to print.')
flags.DEFINE_string('model_dir', default_model_dir, 'Path to saving model files.')
flags.DEFINE_float('learning_rate', 0.05, '学习率')
flags.DEFINE_integer('batch_size', 50, '批次大小')
flags.DEFINE_integer('epochs', 1, '迭代次数')
flags.DEFINE_bool('draw', True, '是否展示图形')
flags.DEFINE_integer('state_size', 40, '隐藏层增加的特征数量')
flags.DEFINE_integer('feature_size', 40, '特征数量')
flags.DEFINE_integer('feature_count', 8000, '特征集大小')


FLAGS = flags.FLAGS


def generate_data(point_count, echo_step, feature_size):
    x = np.array(np.random.choice(2, point_count, p=[0.5, 0.5]))
    y = np.roll(x, echo_step)
    y[0:echo_step] = 0

    x = x.reshape((-1, feature_size))
    y = y.reshape((-1, feature_size))

    return x, y


class Model(object):

    def __init__(self, learning_rate, feature_size, state_size):
        self.feature_size = feature_size
        self.state_size = state_size
        num_classes = 1

        self.batchX_placeholder = tf.placeholder(tf.float32, [None, self.feature_size])
        self.batchY_placeholder = tf.placeholder(tf.float32, [None, self.feature_size])
        # self.init_state = tf.placeholder(tf.float32, [None, self.feature_size * self.state_size])

        #self.current_state = self.init_state
        self.predictions_series = []
        losses = []

        logits = tf.contrib.layers.fully_connected(self.batchX_placeholder, self.feature_size + state_size, activation_fn=tf.nn.tanh)
        logits = tf.contrib.layers.fully_connected(logits, self.feature_size, activation_fn=None)
        loss = (self.batchY_placeholder - logits) ** 2
        self.predictions_series.append(logits)

        self.total_loss = tf.reduce_mean(loss)
        self.optimizer = tf.train.AdamOptimizer(learning_rate).minimize(self.total_loss)

    def train(self, features, labels, batch_size, loss_list, echo_step):
        file_path = FLAGS.model_dir + 'tf_act_10'
        with tf.Session() as sess:
            saver = tf.train.Saver()
            if os.path.exists(FLAGS.model_dir):
                saver.restore(sess, file_path)
            else:
                os.mkdir(FLAGS.model_dir)
                sess.run(tf.global_variables_initializer())

            point_count = len(features)
            batch_count = point_count // batch_size + 0 if point_count % batch_size == 0 else 1

            for index in range(batch_count):
                start_idx = index * batch_size
                end_idx = start_idx + batch_size

                batch_x = features[start_idx:end_idx]
                batch_y = labels[start_idx:end_idx]

                # if current_state.shape == (6, 0) or batch_y.shape == (6, 0) or batch_x.shape == (6, 0):
                #    continue

                total_loss, _, predictions_series = sess.run(
                    [self.total_loss, self.optimizer, self.predictions_series],
                    feed_dict={
                        self.batchX_placeholder: batch_x,
                        self.batchY_placeholder: batch_y
                    }
                )
                loss_list.append(total_loss)

                if index % FLAGS.verbose == 0:
                    print("epoch: {0}: {1}".format(index, total_loss))
                    self.draw(batch_x, batch_y, echo_step, loss_list, predictions_series)

            saver.save(sess, file_path)

    def draw(self, batch_x, batch_y, echo_step, loss_list, predictions_series):
        batch_count = len(batch_x)
        batch_count = 3 if batch_count > 3 else batch_count

        plt.subplots(figsize=(6, 10))
        plt.subplot(batch_count + 1, 1, 1)
        plt.cla()
        plt.plot(loss_list)

        for m in range(batch_count):
            one_hot_output_series = np.array(predictions_series[0][m])
            single_output_series = np.array([1 if out >= 0.5 else 0 for out in one_hot_output_series])

            plt.subplot(batch_count + 1, 1, m + 2)
            plt.cla()
            plt.axis([0, self.feature_size, 0, 2])
            left_offset = range(self.feature_size)
            left_offset2 = range(echo_step, self.feature_size + echo_step)

            label1 = "past values"
            label2 = "True echo values"
            label3 = "Predictions"

            plt.plot(left_offset2, batch_x[m] * 0.2 + 1.5, "o--b", label=label1)
            plt.plot(left_offset, batch_y[m] * 0.2 + 0.8, "x--b", label=label2)
            plt.plot(left_offset, single_output_series * 0.2 + 0.1, "o--y", label=label3)

        plt.legend(loc='best')
        plt.draw()
        plt.pause(0.0001)


def main(argv):
    """ Entry point for running one selfplay game.
    :param argv
    """
    del argv  # Unused

    if os.path.exists(FLAGS.model_dir):
        shutil.rmtree(FLAGS.model_dir)

    plt.ion()
    plt.figure()
    plt.show()

    echo_step = 3
    point_count = FLAGS.feature_count * FLAGS.feature_size
    model = Model(FLAGS.learning_rate, FLAGS.feature_size, FLAGS.state_size)
    loss_list = []
    for m in range(FLAGS.epochs):
        print("batch: {0}".format(m))
        x, y = generate_data(point_count, echo_step, FLAGS.feature_size)
        model.train(x, y, FLAGS.batch_size, loss_list, echo_step)

    plt.ioff()
    plt.show()


if __name__ == '__main__':
    app.run(main)
