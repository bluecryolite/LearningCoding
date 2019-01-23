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
# 3.1 线性回归
# 190123:
# tensor board的图依然怪头怪脑的

# ======================================================================================
import tensorflow as tf
from tensorflow.python.tools.inspect_checkpoint import print_tensors_in_checkpoint_file
import numpy as np
import matplotlib.pyplot as plt
from absl import app, flags
import os

default_model_dir = os.path.abspath('./model_tf_act_01/')

flags.DEFINE_integer('verbose', 1, 'How much debug info to print.')
flags.DEFINE_string('model_dir', default_model_dir, 'Path to saving model files.')
flags.DEFINE_string('model_file_dir', default_model_dir + '/tf_act_01', 'File path to saving model files.')
flags.DEFINE_float('learning_rate', 0.01, '学习率')
flags.DEFINE_integer('epochs', 15, '迭代次数')
flags.DEFINE_bool('draw', True, '是否展示图形')

FLAGS = flags.FLAGS

# 模型定义，用于学习和预估。暂不知道怎么定义成方法
# 创建模型占位符
x = tf.placeholder("float")
y = tf.placeholder("float")

weight = tf.Variable(tf.random_normal([1]), name="weight")
bias = tf.Variable(tf.zeros([1]), name="bias")

z = tf.multiply(x, weight) + bias


def input_fn():
    """ 生成训练数据
    :return (ndarray, ndarray)
    """
    train_x = np.linspace(-1, 1, 500)
    train_y = 2 * train_x + np.random.randn(*train_x.shape) * 0.3

    return train_x, train_y


def train_model(train_x, train_y, learning_rate, epochs):
    """ 训练
    :param train_x: 训练数据
    :param train_y: 训练数据
    :param learning_rate: 学习率
    :param epochs: 重复学习次数
    :return: 返回模型参数，及中间训练数据
    """

    # 损失函数
    cost = tf.reduce_mean(tf.square(y - z))

    # 梯度下降
    optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost)

    # 用于tensorboard
    tf.summary.histogram('z', z)
    tf.summary.scalar('loss_function', cost)

    init = tf.global_variables_initializer()
    with tf.Session() as sess:
        saver = tf.train.Saver(max_to_keep=10)
        sess.run(init)

        # 用于tensor board
        merged_summary_op = tf.summary.merge_all()
        summary_writer = tf.summary.FileWriter(FLAGS.model_dir, sess.graph)

        # 用于生成图形
        plot_datas = {"batch_size": [], "loss": []}

        for epoch in range(epochs):
            for (x_data, y_data) in zip(train_x, train_y):
                sess.run(optimizer, feed_dict={x: x_data, y: y_data})

                # 用于tensor board
                summary_str = sess.run(merged_summary_op, feed_dict={x: x_data, y: y_data})
                summary_writer.add_summary(summary_str, epoch)

            if epoch % FLAGS.verbose == 0:
                # 根据设置的信息展示级别，打印和记录中间训练结果及损失信息
                loss = sess.run(cost, feed_dict={x: train_x, y: train_y})
                print("Epoch:", epoch + 1, "cost=", loss, "W=", sess.run(weight), "b=", sess.run(bias))
                if not (loss == "NA"):
                    plot_datas["batch_size"].append(epoch)
                    plot_datas["loss"].append(loss)
                # 保存检查点
                saver.save(sess, FLAGS.model_file_dir, global_step=epoch)

        # 保存模型，打印训练结果
        saver.save(sess, FLAGS.model_file_dir)
        print(" Finished!")
        w = sess.run(weight)
        b = sess.run(bias)
        print("cost=", sess.run(cost, feed_dict={x: train_x, y: train_y}), "W=", w, "b=", b)
        print("cost=", cost.eval({x: train_x, y: train_y}))

        return w, b, plot_datas


def estimate(estimate_x):
    """ 预估
    :param estimate_x:
    :return: 无
    """
    saver = tf.train.Saver()

    print(estimate_x)

    # 加载最新模型
    print_tensors_in_checkpoint_file(FLAGS.model_file_dir, None, True)
    with tf.Session() as sess:
        saver.restore(sess, FLAGS.model_file_dir)
        print("z=", sess.run(z, feed_dict={x: estimate_x}))

    # 加载检查点
    with tf.Session() as sess:
        ckpt = tf.train.get_checkpoint_state(os.path.abspath(FLAGS.model_dir))
        if ckpt and ckpt.all_model_checkpoint_paths:  # 最新模型为：ckpt.model_checkpoint_path
            print_tensors_in_checkpoint_file(ckpt.all_model_checkpoint_paths[0], None, True)
            saver.restore(sess, ckpt.all_model_checkpoint_paths[0])
            print("z=", sess.run(z, feed_dict={x: estimate_x}))


def draw(train_x, train_y, w, b, plot_datas):
    """ 训练图示
    :param train_x: 训练数据
    :param train_y: 训练数据
    :param w: 模型数据
    :param b: 模型数据
    :param plot_datas: 中间训练数据
    :return: 无
    """
    plt.plot(train_x, train_y, 'ro', label='Original data')
    plt.plot(train_x, w * train_x + b, label="Fittedlint")
    plt.legend()
    plt.show()

    plot_datas["avgloss"] = moving_average(plot_datas["loss"])
    plt.figure(1)
    plt.subplot(211)
    plt.plot(plot_datas["batch_size"], plot_datas["loss"])
    plt.xlabel("Minibatch number")
    plt.ylabel("Loss")
    plt.title("Minibatch run vs. Training loss")
    plt.show()


def moving_average(a, w=10):
    if len(a) < w:
        return a[:]
    return [val if idx < w else sum(a[(idx - w):idx]) / w for idx, val in enumerate(a)]


def main(argv):
    """ Entry point for running one selfplay game.
    :param argv
    """
    del argv  # Unused

    train_x, train_y = input_fn()
    w, b, plot_datas = train_model(
        train_x,
        train_y,
        FLAGS.learning_rate,
        FLAGS.epochs)

    if FLAGS.draw:
        draw(train_x, train_y, w, b, plot_datas)

    estimate(np.linspace(-100, 100, 20))


if __name__ == '__main__':
    app.run(main)
