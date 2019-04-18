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
# 7.3 全连接 MNIST
# 数据下载地址：http://yann.lecun.com/exdb/mnist/
# 数据下载后保存地址：data/MNIST
# ======================================================================================
# 笔记
# 190417：
# MNIST的分类算法，就是线性分类
#
# ======================================================================================
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import pylab
from absl import app, flags
import os
# from tensorflow.contrib.tensorboard.plugins import projector

default_model_dir = os.path.abspath('./model_tf_act_07/')
images_dir = os.path.abspath('data/MNIST')
feature_pixes = 28 * 28
hidden_pixes = 256
label_count = 10

flags.DEFINE_integer('verbose', 1, 'How much debug info to print.')
flags.DEFINE_string('model_dir', default_model_dir, 'Path to saving model files.')
flags.DEFINE_string('model_file_dir', default_model_dir + '/tf_act_02', 'File path to saving model files.')
flags.DEFINE_float('learning_rate', 0.001, '学习率')
flags.DEFINE_integer('batch_size', 100, '批次大小')
flags.DEFINE_integer('epochs', 50, '迭代次数')
flags.DEFINE_bool('draw', True, '是否展示图形')

FLAGS = flags.FLAGS


class Model(object):

    def __init__(self, learning_rate):

        self.x = tf.placeholder(tf.float32, [None, feature_pixes])
        self.y = tf.placeholder(tf.float32, [None, label_count])

        weights = {
            'h1': tf.Variable(tf.random_normal([feature_pixes, hidden_pixes]), name='weight1'),
            'h2': tf.Variable(tf.random_normal([hidden_pixes, hidden_pixes]), name='weight2'),
            'h3': tf.Variable(tf.random_normal([hidden_pixes, label_count]), name='weight3')
        }

        biases = {
            'b1': tf.Variable(tf.zeros([hidden_pixes]), name='bias1'),
            'b2': tf.Variable(tf.zeros([hidden_pixes]), name='bias2'),
            'b3': tf.Variable(tf.zeros([label_count]), name='bias3')
        }

        self.layer_1 = tf.add(tf.matmul(self.x, weights['h1']), biases['b1'], name='hidden1')
        self.layer_2 = tf.add(tf.matmul(tf.nn.relu(self.layer_1), weights['h2']), biases['b2'], name='hidden2')
        self.pred = tf.add(tf.matmul(tf.nn.relu(self.layer_2), weights['h3']), biases['b3'], name="pred")
        self.cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=self.pred, labels=self.y))
        self.optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(self.cost)

    def train(self, train_data, epochs, batch_size):
        # 用于 tensor board
        # 用于tensor board
        tf.summary.histogram("pred", self.pred)
        tf.summary.scalar("cost", self.cost)

        with tf.Session() as sess:
            saver = tf.train.Saver()
            sess.run(tf.global_variables_initializer())

            # 用于 tensor board
            summary_writer = tf.summary.FileWriter(FLAGS.model_dir, sess.graph)
            merged_summary = tf.summary.merge_all()
            # config = projector.ProjectorConfig()
            # embedding = config.embeddings.add()
            # embedding.tensor_name = "pred"
            # projector.visualize_embeddings(summary_writer, config)

            for epoch in range(epochs):
                avg_cost = 0.0
                total_batch = int(train_data.num_examples / batch_size)
                for i in range(total_batch):
                    batch_xs, batch_ys = train_data.next_batch(batch_size)
                    _, c = sess.run([self.optimizer, self.cost], feed_dict={self.x: batch_xs, self.y: batch_ys})
                    avg_cost += c / total_batch

                if (epoch + 1) % FLAGS.verbose == 0:
                    print("Epoch", '%04d' % (epoch + 1), "cost=", "{:.9f}".format(avg_cost))

                # 用于 tensor board
                summary_writer.add_summary(sess.run(merged_summary, feed_dict={self.x: batch_xs, self.y: batch_ys}), epoch)

            saver.save(sess, FLAGS.model_file_dir)
            print(" Finished!")

    def estimate(self, test_data):
        with tf.Session() as sess:
            saver = tf.train.Saver()
            saver.restore(sess, FLAGS.model_file_dir)

            correct_prediction = tf.equal(tf.argmax(self.pred, 1), tf.argmax(self.y, 1))
            accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
            print("Accuracy:", accuracy.eval({self.x: test_data.images, self.y: test_data.labels}))

            output = tf.argmax(self.pred, 1)
            batch_xs, batch_ys = test_data.next_batch(10)
            outputval, predv = sess.run([output, self.pred], feed_dict={self.x: batch_xs})
            print(outputval, predv, batch_ys)

            def show_im(image):
                image = image.reshape(-1, 28)
                pylab.imshow(image)
                pylab.show()

            for im in batch_xs:
                show_im(im)


def main(argv):
    """ Entry point for running one selfplay game.
    :param argv
    """
    del argv  # Unused

    mnist = input_data.read_data_sets(images_dir, one_hot=True)
    mnist1 = tf.keras.datasets.mnist
    mnist1.load_data()
    print(mnist.train.images)
    print(mnist.train.images.shape)

    train_model = Model(FLAGS.learning_rate)
    train_model.train(mnist.train, FLAGS.epochs, FLAGS.batch_size)
    train_model.estimate(mnist.test)


if __name__ == '__main__':
    app.run(main)
