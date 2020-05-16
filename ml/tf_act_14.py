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
# 9.8.3 使用basic_rnn_seq2seq拟合曲线
# ======================================================================================
# 笔记
# 2020-05-16 训练不出来啊。很疑惑，每批之间数据并无关联，就是各自独立的训练，怎么做到loss逐渐走低的？
#
#
# ======================================================================================

import random
import math
import tensorflow as tf
import numpy as np
from absl import app, flags
import os
import shutil

default_model_dir = os.path.abspath('./model_tf_act_14/')

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


def do_generate_x_y(batch_size, seqlen):
    batch_x = []
    batch_y = []

    for _ in range(batch_size):
        offset_rand = random.random() * 2 * math.pi
        freq_rand = (random.random() - 0.5) / 1.5 * 15 + 0.5
        amp_rand = random.random() + 0.1

        sin_data = amp_rand * np.sin(np.linspace(offset_rand, seqlen / 15.0 * freq_rand * 3.0 * math.pi + offset_rand, seqlen * 2))

        offset_rand = random.random() * 2 * math.pi
        freq_rand = (random.random() - 0.5) / 1.5 * 15 + 0.5
        amp_rand = random.random() * 1.2

        sin_data = amp_rand * np.cos(np.linspace(offset_rand, seqlen / 15.0 * freq_rand * 3.0 * math.pi + offset_rand, seqlen * 2)) + sin_data

        batch_x.append(np.array([sin_data[: seqlen]]).T)
        batch_y.append(np.array([sin_data[seqlen:]]).T)

    batch_x = np.array(batch_x).transpose((1, 0, 2))
    batch_y = np.array(batch_y).transpose((1, 0, 2))

    return batch_x, batch_y

def generate_data(is_train, seq_length, batch_size):
    if is_train:
        return do_generate_x_y(batch_size, seq_length)
    else:
        return do_generate_x_y(batch_size, seq_length * 2)


class Model(object):

    def __init__(self, shape, learning_rate):
        seq_length = shape[0]
        output_dim = input_dim = shape[-1]
        hidden_dim = 12
        layers_stacked_count = 2

        # L2 正则参数
        lambda_l2_reg = 0.003

        self.encoder_input = []
        self.expected_output = []
        self.decoder_input = []

        tf.reset_default_graph()
        for i in range(seq_length):
            self.encoder_input.append(tf.placeholder(tf.float32, shape=(None, input_dim)))
            self.expected_output.append(tf.placeholder(tf.float32, shape=(None, output_dim)))
            self.decoder_input.append(tf.placeholder(tf.float32, shape=(None, input_dim)))

        tcells = []
        for i in range(layers_stacked_count):
            tcells.append(tf.contrib.rnn.GRUCell(hidden_dim))
        mcell = tf.contrib.rnn.MultiRNNCell(tcells)

        dec_outputs, dec_memory = tf.contrib.legacy_seq2seq.basic_rnn_seq2seq(self.encoder_input, self.decoder_input, mcell)

        reshaped_outputs = []
        for ii in dec_outputs:
            reshaped_outputs.append(tf.contrib.layers.fully_connected(ii, output_dim, activation_fn=None))

        # 计算L2的loss值
        self.output_loss = 0
        for _y1, _y2 in zip(reshaped_outputs, self.expected_output):
            self.output_loss += tf.reduce_mean(tf.pow(_y1 - _y2, 2))

        # 求正则化loss值
        reg_loss = 0
        for tf_var in tf.trainable_variables():
            if not ("fully_connected" in tf_var.name):
                reg_loss += tf.reduce_mean(tf.nn.l2_loss(tf_var))

        self.loss = self.output_loss + lambda_l2_reg * reg_loss
        self.train_op = tf.train.AdadeltaOptimizer(learning_rate=learning_rate).minimize(self.loss)

    def train(self, nb_iters, batch_size, seq_length, verbose):

        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())

            train_losses = []
            for k in range(nb_iters + 1):
                train_now, train_future = generate_data(True, seq_length, batch_size)

                feed_dict = {self.encoder_input[t]: train_now[t] for t in range(len(self.encoder_input))}
                feed_dict.update({self.expected_output[t]: train_future[t] for t in range(len(self.expected_output))})
                c = np.concatenate(([np.zeros_like(train_future[0])], train_future[:-1]), axis=0)
                feed_dict.update({self.decoder_input[t]: c[t] for t in range(len(c))})

                _, loss_t = sess.run([self.train_op, self.loss], feed_dict)

                train_losses.append(loss_t)
                if k % verbose == 0:
                    print("Step {} / {}, train loss: {}".format(k, nb_iters, loss_t))



def main(argv):
    """ Entry point for running one selfplay game.
    :param argv
    """
    del argv  # Unused

    if os.path.exists(FLAGS.model_dir):
        shutil.rmtree(FLAGS.model_dir)

    """
    sample_now, sample_f = generate_data(is_train=True, seq_length=15, batch_size=10)
    print("training examples:")
    print(sample_now.shape)
    print("(seq_length, batch_size, output_dim)")
    """

    # 学习率
    learning_rate = 0.04
    nb_iters = 100
    batch_size = 10
    seq_length = 15
    model = Model((seq_length, batch_size, 1), learning_rate)
    model.train(nb_iters, batch_size, seq_length, 1)


if __name__ == '__main__':
    app.run(main)
