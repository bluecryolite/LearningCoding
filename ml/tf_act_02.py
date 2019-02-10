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
# 5 MNIST
# ======================================================================================
# 笔记
# 190210：
# 好吧，又写了一遍MNIST，依然不明白W * x + b是怎么来的。先这样了。
# 读取数据集的方法过时了，未更新为最新的读取方法
# tensorboard的Projector未能展示
#
# ======================================================================================
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import pylab
from absl import app, flags
import os
# from tensorflow.contrib.tensorboard.plugins import projector

default_model_dir = os.path.abspath('./model_tf_act_02/')
images_dir = os.path.abspath('data/MNIST')
feature_pixes = 28 * 28
label_count = 10

flags.DEFINE_integer('verbose', 1, 'How much debug info to print.')
flags.DEFINE_string('model_dir', default_model_dir, 'Path to saving model files.')
flags.DEFINE_string('model_file_dir', default_model_dir + '/tf_act_02', 'File path to saving model files.')
flags.DEFINE_float('learning_rate', 0.01, '学习率')
flags.DEFINE_integer('batch_size', 100, '批次大小')
flags.DEFINE_integer('epochs', 50, '迭代次数')
flags.DEFINE_bool('draw', True, '是否展示图形')

FLAGS = flags.FLAGS

# 模型定义，用于学习和预估。暂不知道怎么定义成方法
# 创建模型占位符
x = tf.placeholder(tf.float32, [None, feature_pixes])
y = tf.placeholder(tf.float32, [None, label_count])

weight = tf.Variable(tf.random_normal([feature_pixes, label_count]), name="weight")
bias = tf.Variable(tf.zeros([label_count]), name="bias")

pred = tf.nn.softmax(tf.matmul(x, weight) + bias, name="pred")

# 用于tensor board
tf.summary.histogram("pred", pred)


def train_model(train_data, learning_rate, epochs, batch_size):
    cost = tf.reduce_mean(-tf.reduce_sum(y * tf.log(pred), reduction_indices=1))
    optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost)

    # 用于 tensor board
    tf.summary.scalar("cost", cost)

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
                _, c = sess.run([optimizer, cost], feed_dict={x: batch_xs, y: batch_ys})
                avg_cost += c / total_batch

            if (epoch + 1) % FLAGS.verbose == 0:
                print("Epoch", '%04d' % (epoch + 1), "cost=", "{:.9f}".format(avg_cost))

            # 用于 tensor board
            summary_writer.add_summary(sess.run(merged_summary, feed_dict={x: batch_xs, y: batch_ys}), epoch)

        saver.save(sess, FLAGS.model_file_dir)
        print(" Finished!")


def estimate(test_data):
    with tf.Session() as sess:
        saver = tf.train.Saver()
        saver.restore(sess, FLAGS.model_file_dir)

        correct_prediction = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        print("Accuracy:", accuracy.eval({x: test_data.images, y: test_data.labels}))

        output = tf.argmax(pred, 1)
        batch_xs, batch_ys = test_data.next_batch(10)
        outputval, predv = sess.run([output, pred], feed_dict={x: batch_xs})
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

    train_model(mnist.train, FLAGS.learning_rate, FLAGS.epochs, FLAGS.batch_size)
    estimate(mnist.test)


if __name__ == '__main__':
    app.run(main)
