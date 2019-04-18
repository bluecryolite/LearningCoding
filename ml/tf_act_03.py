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
# 6.5 交叉熵实验
# ======================================================================================
# 笔记
#
# ======================================================================================
import tensorflow as tf
from absl import app, flags
import os

default_model_dir = os.path.abspath('./model_tf_act_03/')

flags.DEFINE_integer('verbose', 1, 'How much debug info to print.')
flags.DEFINE_string('model_dir', default_model_dir, 'Path to saving model files.')
flags.DEFINE_float('learning_rate', 0.01, '学习率')
flags.DEFINE_integer('batch_size', 100, '批次大小')
flags.DEFINE_integer('epochs', 50, '迭代次数')
flags.DEFINE_bool('draw', True, '是否展示图形')

FLAGS = flags.FLAGS

# 模型定义，用于学习和预估。暂不知道怎么定义成方法
x = tf.placeholder(tf.float32, [None, 3])
logits_scaled = tf.nn.softmax(x)
logits_scaled2 = tf.nn.softmax(logits_scaled)

# 用于tensor board
tf.summary.histogram("scaled", logits_scaled)
tf.summary.histogram("scaled2", logits_scaled2)


def input_fn():
    labels = [[0, 0, 1], [0, 1, 0]]
    logits = [[2, 0.5, 6], [0.1, 0, 3]]
    return logits, labels


def input_fn1():
    labels = [[0.4, 0.1, 0.5], [0.3, 0.6, 0.1]]
    logits = [[2, 0.5, 6], [0.1, 0, 3]]
    return logits, labels


def train_model(logits, lables, model):
    result = tf.nn.softmax_cross_entropy_with_logits(labels=lables, logits=logits)
    with tf.Session() as sess:
        ret = sess.run(model, feed_dict={x: logits})
        print("scaled=", ret)
        print("result=", sess.run(result), "\n")
        return ret


def training(fn_input):
    logits, labels = fn_input()

    print("scaled:")
    ret = train_model(logits, labels, logits_scaled)
    print("acaled2:")
    ret1 = train_model(ret, labels, logits_scaled2)

    with tf.Session() as sess:
        print("result3=", sess.run(-tf.reduce_sum(labels*tf.log(logits_scaled), 1), feed_dict={x: logits}))


def main(argv):
    """ Entry point for running one selfplay game.
    :param argv
    """
    del argv  # Unused

    print("\none hot")
    training(input_fn)
    print("\nnon one hot")
    training(input_fn1)


if __name__ == '__main__':
    app.run(main)
