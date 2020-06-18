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
# 9.2.2 拟合回声信号
# ======================================================================================
# 笔记
#
# 190428:
# state_size，类似隐藏层的输出，调高能加快学习进度
#
# ======================================================================================
import tensorflow as tf
from absl import app, flags
import os
import shutil
import numpy as np
import matplotlib.pyplot as plt

default_model_dir = os.path.abspath('./model_tf_act_10/')

flags.DEFINE_integer('verbose', 100, 'How much debug info to print.')
flags.DEFINE_string('model_dir', default_model_dir, 'Path to saving model files.')
flags.DEFINE_float('learning_rate', 0.05, '学习率')
flags.DEFINE_integer('batch_size', 6, '批次大小')
flags.DEFINE_integer('epochs', 5, '迭代次数')
flags.DEFINE_bool('draw', True, '是否展示图形')
flags.DEFINE_integer('batches', 750, '每迭代训练批次')
flags.DEFINE_integer('truncated_backprop_length', 15, 'truncated_backprop_length')
flags.DEFINE_integer('state_size', 16, '状态大小')

FLAGS = flags.FLAGS


def generate_data(point_count, echo_step, batch_size):
    x = np.array(np.random.choice(2, point_count, p=[0.5, 0.5]))
    y = np.roll(x, echo_step)
    y[0:echo_step] = 0

    x = x.reshape((batch_size, -1))
    y = y.reshape((batch_size, -1))

    return x, y


class Model(object):

    def __init__(self, learning_rate, batch_size, truncated_backprop_length, state_size):
        self.truncated_backprop_length = truncated_backprop_length
        self.state_size = state_size
        num_classes = 2
        self.batch_size = batch_size

        self.batchX_placeholder = tf.placeholder(tf.float32, [batch_size, truncated_backprop_length])
        self.batchY_placeholder = tf.placeholder(tf.int32, [batch_size, truncated_backprop_length])
        self.init_state = tf.placeholder(tf.float32, [batch_size, state_size])

        inputs_series = tf.unstack(self.batchX_placeholder, axis=1)
        labels_series = tf.unstack(self.batchY_placeholder, axis=1)

        self.current_state = self.init_state
        self.predictions_series = []
        losses = []

        for current_input, label in zip(inputs_series, labels_series):
            current_input = tf.reshape(current_input, [self.batch_size, 1])
            input_and_state_concatenated = tf.concat([current_input, self.current_state], 1)

            self.current_state = tf.contrib.layers.fully_connected(input_and_state_concatenated, self.state_size,
                                                                   activation_fn=tf.nn.tanh)

            logits = tf.contrib.layers.fully_connected(self.current_state, num_classes, activation_fn=None)
            loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=label, logits=logits)
            losses.append(loss)
            self.predictions_series.append(tf.nn.softmax(logits))

        self.total_loss = tf.reduce_mean(losses)
        self.optimizer = tf.train.AdagradOptimizer(learning_rate).minimize(self.total_loss)

    def train(self, features, labels, epochs, loss_list, echo_step):
        file_path = FLAGS.model_dir + '/tf_act_10'
        with tf.Session() as sess:
            saver = tf.train.Saver()
            if os.path.exists(FLAGS.model_dir):
                saver.restore(sess, file_path)
            else:
                os.mkdir(FLAGS.model_dir)
                sess.run(tf.global_variables_initializer())
            current_state = np.zeros((self.batch_size, self.state_size))
            for epoch in range(epochs):
                start_idx = epoch * self.truncated_backprop_length
                end_idx = start_idx + self.truncated_backprop_length

                batch_x = features[:, start_idx:end_idx]
                batch_y = labels[:, start_idx:end_idx]

                # if current_state.shape == (6, 0) or batch_y.shape == (6, 0) or batch_x.shape == (6, 0):
                #    continue

                total_loss, _, current_state, predictions_series = sess.run(
                    [self.total_loss, self.optimizer, self.current_state, self.predictions_series],
                    feed_dict={
                        self.batchX_placeholder: batch_x,
                        self.batchY_placeholder: batch_y,
                        self.init_state: current_state
                    }
                )
                loss_list.append(total_loss)

                if epoch % FLAGS.verbose == 0:
                    print("epoch: {0}: {1}".format(epoch, total_loss))
                    # self.draw(batch_x, batch_y, echo_step, loss_list, predictions_series)

            saver.save(sess, file_path)

    def draw(self, batch_x, batch_y, echo_step, loss_list, predictions_series):
        plt.subplots(figsize=(12, 8))
        plt.subplot(3, 1, 1)
        plt.cla()
        plt.plot(loss_list)

        for series_idx in range(self.batch_size):
            one_hot_output_series = np.array(predictions_series)[:, series_idx, :]
            single_output_series = np.array([1 if out[0] < 0.5 else 0 for out in one_hot_output_series])

            plt.subplot(3, 3, series_idx + 4)
            plt.cla()
            plt.axis([0, self.truncated_backprop_length, 0, 2])
            left_offset = range(self.truncated_backprop_length)
            left_offset2 = range(echo_step, self.truncated_backprop_length + echo_step)

            label1 = "past values"
            label2 = "True echo values"
            label3 = "Predictions"

            plt.plot(left_offset2, batch_x[series_idx, :] * 0.2 + 1.5, "o--b", label=label1)
            plt.plot(left_offset, batch_y[series_idx, :] * 0.2 + 0.8, "x--b", label=label2)
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
    point_count = FLAGS.batches * FLAGS.batch_size * FLAGS.truncated_backprop_length
    model = Model(FLAGS.learning_rate, FLAGS.batch_size, FLAGS.truncated_backprop_length, FLAGS.state_size)
    loss_list = []
    for m in range(FLAGS.epochs):
        print("batch: {0}".format(m))
        x, y = generate_data(point_count, echo_step, FLAGS.batch_size)
        model.train(x, y, FLAGS.batches, loss_list, echo_step)

    plt.ioff()
    plt.show()


if __name__ == '__main__':
    app.run(main)
