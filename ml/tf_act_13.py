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
# 9.4.3 MNIST又来分类咯，这盘分类的姿势多
# ======================================================================================
# 笔记
#
#
# ======================================================================================
import tensorflow as tf
from absl import app, flags
import os
import numpy as np
import pylab
from tensorflow.examples.tutorials.mnist import input_data

default_model_dir = os.path.abspath('./model_tf_act_13/')
images_dir = os.path.abspath('data/MNIST')

flags.DEFINE_integer('verbose', 1, 'How much debug info to print.')
flags.DEFINE_string('model_dir', default_model_dir, 'Path to saving model files.')
flags.DEFINE_string('model_file_dir', default_model_dir + '/tf_act_02', 'File path to saving model files.')
flags.DEFINE_float('learning_rate', 0.001, '学习率')
flags.DEFINE_integer('batch_size', 100, '批次大小')
flags.DEFINE_integer('epochs', 30, '迭代次数')

FLAGS = flags.FLAGS


class Model(object):

    def __init__(self, learning_rate):
        self.input_size = 28
        self.steps_count = self.input_size
        self.hidden_count = 128
        self.label_count = 10

        self.x = tf.placeholder(tf.float32, [None, self.input_size, self.input_size])
        self.y = tf.placeholder(tf.float32, [None, self.label_count])

        self.pred = self.create_model()
        self.cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=self.pred, labels=self.y))
        self.optimizer = tf.train.AdamOptimizer(learning_rate).minimize(self.cost)

    def create_model(self):
        pass

    def train(self, train_data, epochs, batch_size):

        with tf.Session() as sess:
            saver = tf.train.Saver()
            sess.run(tf.global_variables_initializer())

            for epoch in range(epochs):
                avg_cost = 0.0
                total_batch = int(train_data.num_examples / batch_size)
                for i in range(total_batch):
                    batch_xs, batch_ys = train_data.next_batch(batch_size)
                    batch_xs = [m.reshape((self.input_size, self.input_size)) for m in batch_xs]
                    _, c = sess.run([self.optimizer, self.cost], feed_dict={self.x: batch_xs, self.y: batch_ys})
                    avg_cost += c / total_batch

                if (epoch + 1) % FLAGS.verbose == 0:
                    print("Epoch", '%04d' % (epoch + 1), "cost=", "{:.9f}".format(avg_cost))

            saver.save(sess, FLAGS.model_file_dir)
            print(" Finished!")

    def estimate(self, test_data):
        with tf.Session() as sess:
            saver = tf.train.Saver()
            saver.restore(sess, FLAGS.model_file_dir)

            correct_prediction = tf.equal(tf.argmax(self.pred, 1), tf.argmax(self.y, 1))
            accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
            print("Accuracy:", accuracy.eval({self.x: [m.reshape((self.input_size, self.input_size))
                                                       for m in test_data.images],
                                              self.y: test_data.labels}))

            output = tf.argmax(self.pred, 1)
            batch_xs, batch_ys = test_data.next_batch(10)
            batch_xs = [m.reshape((self.input_size, self.input_size)) for m in batch_xs]
            outputval, predv = sess.run([output, self.pred], feed_dict={self.x: batch_xs})
            print(outputval, '\n', predv, batch_ys)

            def show_im(image):
                image = image.reshape(-1, 28)
                pylab.imshow(image)
                pylab.show()

            for im in batch_xs:
                show_im(im)


class SingleLSTM(Model):
    def create_model(self):
        x1 = tf.unstack(self.x, self.input_size, axis=1)
        lstm_cell = tf.nn.rnn_cell.BasicLSTMCell(self.hidden_count, forget_bias=1.0)
        outputs, states = tf.nn.static_rnn(lstm_cell, x1, dtype=tf.float32)
        return tf.contrib.layers.fully_connected(outputs[-1], self.label_count, activation_fn=None)


class SingleGRU(Model):
    def create_model(self):
        x1 = tf.unstack(self.x, self.input_size, axis=1)
        gru = tf.nn.rnn_cell.GRUCell(self.hidden_count)
        outputs = tf.nn.static_rnn(gru, x1, dtype=tf.float32)
        return tf.contrib.layers.fully_connected(outputs[-1], self.label_count, activation_fn=None)


def generate_data(point_count, echo_step, batch_size):
    x = np.array(np.random.choice(2, point_count, p=[0.5, 0.5]))
    y = np.roll(x, echo_step)
    y[0:echo_step] = 0

    x = x.reshape((batch_size, -1))
    y = y.reshape((batch_size, -1))

    return x, y


def main(argv):
    """ Entry point for running one selfplay game.
    :param argv
    """
    del argv  # Unused

    mnist = input_data.read_data_sets(images_dir, one_hot=True)

    # single_lstm = SingleLSTM(FLAGS.learning_rate)
    # single_lstm.train(mnist.train, FLAGS.epochs, FLAGS.batch_size)
    # single_lstm.estimate(mnist.test)

    single_gru = SingleGRU(FLAGS.learning_rate)
    single_gru.train(mnist.train, FLAGS.epochs, FLAGS.batch_size)
    single_gru.estimate(mnist.test)


if __name__ == '__main__':
    app.run(main)
