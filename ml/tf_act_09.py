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
# 9.2.1 Python实现的退位减法器
# ======================================================================================
# 笔记
#
#
#
# ======================================================================================
import copy
import numpy as np

np.random.seed(0)


def sigmoid(output):
    return 1 / (1 + np.exp(-output))


def sigmoid_output_to_derivation(output):
    return output * (1 - output)


int2binary = {}

binary_dim = 8
largest_number = pow(2, binary_dim)
binary = np.unpackbits(np.array([range(largest_number)], dtype=np.uint8).T, axis=1)

for i in range(largest_number):
    int2binary[i] = binary[i]

# 参数设置
alpha = 0.9
input_dim = 2
hidden_dim = 16
output_dim = 1

# 初始化网络
synapse_0 = (2 * np.random.random((input_dim, hidden_dim)) - 1) * 0.05
synapse_1 = (2 * np.random.random((hidden_dim, output_dim)) - 1) * 0.05
synapse_h = (2 * np.random.random((hidden_dim, hidden_dim)) - 1) * 0.05

# 反向传播的权重更新值
synapse_0_update = np.zeros_like(synapse_0)
synapse_1_update = np.zeros_like(synapse_1)
synapse_h_update = np.zeros_like(synapse_h)

epochs = 10000

# 准备训练数据
for j in range(epochs):
    a_int = np.random.randint(largest_number)
    b_int = np.random.randint(largest_number / 2)
    if a_int < b_int:
        a_int, b_int = b_int, a_int

    a = int2binary[a_int]
    b = int2binary[b_int]
    c = int2binary[a_int - b_int]

    # 存储神经网络的预测值
    d = np.zeros_like(c)
    # 总误差
    overall_errors = 0
    # 每个时间点输出层误差
    layer_2_deltas = list()
    # 每个时间点隐藏层的值
    layer_1_values = list()

    # 初始化隐藏层
    layer_1_values.append(np.ones(hidden_dim) * 0.1)

    # 正向传播
    for position in range(binary_dim):
        pos = binary_dim - position - 1
        x = np.array([[a[pos], b[pos]]])
        y = np.array([[c[pos]]]).T

        # 输入层 + 之前的隐藏层 -> 新的隐藏层
        layer_1 = sigmoid(np.dot(x, synapse_0) + np.dot(layer_1_values[-1], synapse_h))
        # 输出层
        layer_2 = sigmoid(np.dot(layer_1, synapse_1))
        # 误差
        layer_2_error = y - layer_2
        layer_2_deltas.append(layer_2_error * sigmoid_output_to_derivation(layer_2))
        # 总误差
        overall_errors += np.abs(layer_2_error[0])

        d[pos] = np.round(layer_2[0][0])
        # 保存隐藏层
        layer_1_values.append(copy.deepcopy(layer_1))

    # 反向传播
    future_layer_1_delta = np.zeros(hidden_dim)
    for position in range(binary_dim):
        x = np.array([[a[position], b[position]]])
        # 当前隐藏层
        layer_1 = layer_1_values[-position - 1]
        # 前一隐藏层
        prev_layer_1 = layer_1_values[-position - 2]

        layer_2_delta = layer_2_deltas[-position - 1]
        layer_1_delta = (future_layer_1_delta.dot(synapse_h.T)
                         + layer_2_delta.dot(synapse_1.T)) * sigmoid_output_to_derivation(layer_1)

        # 临时保存权重矩阵
        synapse_1_update += np.atleast_2d(layer_1).T.dot(layer_2_delta)
        synapse_h_update += np.atleast_2d(prev_layer_1).T.dot(layer_1_delta)
        synapse_0_update += x.T.dot(layer_1_delta)

        future_layer_1_delta = layer_1_delta

    # 更新权重矩阵
    synapse_0 += synapse_0_update * alpha
    synapse_1 += synapse_1_update * alpha
    synapse_h += synapse_h_update * alpha

    # 清空临时权重
    synapse_0_update *= 0
    synapse_1_update *= 0
    synapse_h_update *= 0

    if j % 200 == 0 or j == epochs - 1:
        print('迭代({0})：误差 {1}'.format(j, str(overall_errors)))
        # print('Pred:' + str(d))
        # print('True:' + str(c))
        out = 0
        for index, x in enumerate(reversed(d)):
            out += x * pow(2, index)
        print('{0} - {1} = {2}'.format(a_int, b_int, out))
        print('--------------------')

exit(0)
