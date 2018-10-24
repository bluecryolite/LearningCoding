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

# google机器学习教程中的使用TensorFlow的基本步骤.
# 详见：
# https://colab.research.google.com/notebooks/mlcc/first_steps_with_tensor_flow.ipynb?utm_source=mlcc&utm_campaign=colab-external&utm_medium=referral&utm_content=firststeps-colab&hl=zh-cn

# ======================================================================================
# 笔记：
# 181017：
# 如果不自定义model_dir，则模型会在一个临时目录生成，每次运行目录均不同，会重新训练
# 自定义model_dir后，模型在指定目录下，每次运行时会读取之前的训练结果，训练结果会在之前的基础上收敛
#
# 181018:
# input_fn：生成训练和评测的数据集。
# 同昨天理解，训练结果会放在model_dir下，评测的时候，会从这个目录读取这个模型
#
# 181020:
# skleran.metrics中的回归评价指标：
# mean_squared_error 均方误差  https://zh.wikipedia.org/wiki/%E5%9D%87%E6%96%B9%E8%AF%AF%E5%B7%AE
# RMSE 均方根误差 （用于更好滴表达均方误差） https://zh.wikipedia.org/wiki/%E5%9D%87%E6%96%B9%E6%A0%B9%E8%AF%AF%E5%B7%AE
# mean_absolute_error 平均绝对误差 sum|(yi - y'i)| / m
# r2_score https://zh.wikipedia.org/wiki/%E5%86%B3%E5%AE%9A%E7%B3%BB%E6%95%B0

# ======================================================================================

from __future__ import print_function
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.python.data import Dataset
from sklearn import metrics
import math
from matplotlib import pyplot as plt
from matplotlib import cm
from IPython import display
import os

csv_file_path = "~/Downloads/california_housing_train.csv"
feature_total_rooms = "total_rooms"
target_column = "median_house_value"

california_housing_dataframe = pd.read_csv(csv_file_path, sep=",")
california_housing_dataframe = california_housing_dataframe.reindex(np.random.permutation(california_housing_dataframe.index))
california_housing_dataframe[target_column] /= 1000.0


def train_model(learning_rate, steps, batch_size, input_feature=feature_total_rooms, model_dir=os.path.abspath("./model")):
    """ Trains a linear regression model of one feature

    :param learning_rate: a `float`, the learning rate.
    :param steps:  a non-zero `int`, the total number of training steps. a training step
        consists of a forward and backward pass using a single batch.
    :param batch_size:  a non-zero `int`, the batch size.
    :param input_feature: a `string` specifying a column from `california_housting_dataframe`
        to use as input feature
    :param model_dir: a `string' specifying a path to saving model
    :return:
    """

    periods = 10
    steps_per_periods = steps / periods

    # 定义特征
    my_feature_data = california_housing_dataframe[[input_feature]] #特征数组
    feature_columns = [tf.feature_column.numeric_column(feature_total_rooms)] #特征类型数组

    # 定义目标
    targets = california_housing_dataframe[target_column]

    # 使用小批量随机梯度下降训练模型
    # 并使用梯度裁剪确保梯度大小不会过大
    my_optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate)
    my_optimizer = tf.contrib.estimator.clip_gradients_by_norm(my_optimizer, 5.0)

    # 定义线性回归模型
    linear_regressor = tf.estimator.LinearRegressor(feature_columns=feature_columns, optimizer=my_optimizer, model_dir=model_dir)

    # Create input functions.
    training_input_fn = lambda: input_fn(my_feature_data, targets, batch_size=batch_size)
    prediction_input_fn = lambda: input_fn(my_feature_data, targets, num_epochs=1, shuffle=False)

    # Set up to plot the state of our model's line each period.
    plt.figure(figsize=(15, 6))

    plt.subplot(1, 4, 1)
    plt.scatter(california_housing_dataframe[input_feature], california_housing_dataframe[target_column])

    plt.subplot(1, 4, 2)
    plt.title("Learned Line by Period")
    plt.ylabel(target_column)
    plt.xlabel(input_feature)
    sample = california_housing_dataframe.sample(n=300)  #samaple没有作用，仅仅是用于帮助生成训练结果的图
    plt.scatter(sample[input_feature], sample[target_column])
    colors = [cm.coolwarm(x) for x in np.linspace(-1, 1, periods)]

    plt.subplot(1, 4, 3)
    plt.scatter(sample[input_feature], sample[target_column])

    # Train the model, but do so inside a loop so that we can periodically assess
    # loss metrics.
    print("Training model...")
    print("RMSE (on training data):")
    root_mean_squared_errors = []

    for period in range(0, periods):
        # Train the model, starting from the prior state.
        linear_regressor.train(input_fn=training_input_fn, steps=steps_per_periods)

        # Take the break and compute predication.
        preditions = linear_regressor.predict(input_fn=prediction_input_fn)
        preditions = np.array([item['predictions'][0] for item in preditions])

        # Compute loss
        mean_squared_error = metrics.mean_squared_error(preditions, targets)
        root_mean_squared_error = math.sqrt(mean_squared_error)

        print("  period %2d : %0.3f" % (period, root_mean_squared_error))

        # Add the loss metrics from this period to our list.
        root_mean_squared_errors.append(root_mean_squared_error)  #只用于了作图

        # Finally, track the weights and biases over time.
        # Apply some math to ensure that the data and line are plotted neatly.
        y_extents = np.array([0, sample[input_feature].max()])

        # Retrieve the final weight and bias generated during training.
        weight = linear_regressor.get_variable_value("linear/linear_model/%s/weights" % input_feature)[0]
        bias = linear_regressor.get_variable_value("linear/linear_model/bias_weights")

        x_extents = (y_extents - bias) / weight
        x_extents = np.maximum(np.maximum(x_extents, sample[input_feature].max()), sample[input_feature].min())
        y_extents = weight * x_extents + bias
        plt.plot(x_extents, y_extents, color=colors[period])

    print("Model training finished.")

    # Output a graph of loss metrics over periods.
    plt.subplot(1, 4, 4)
    plt.ylabel("RMSE")
    plt.xlabel("Periods")
    plt.title("Root Mean Squared Error vs. Periods")
    plt.tight_layout()
    plt.plot(root_mean_squared_errors)

    plt.show()

    # Output a table with calibration data.
    calibration_data = pd.DataFrame()
    calibration_data["predictions"] = pd.Series(preditions)
    calibration_data["targets"] = pd.Series(targets)
    display.display(calibration_data.describe())

    print("Final RMSE (on training data): %0.2f" % root_mean_squared_error)


# 定义训练集和测试集。本案中，训练集和测试集相同
def input_fn(features, targets, batch_size=1, shuffle=True, num_epochs=None):
    """ Trains a linear regression model of one feature.
    :param features: pandas DataFrame of features.
    :param targets: pandas DataFrame of targets.
    :param batch_size: Size of batches to be passed to the model.
    :param shuffle: True or False. Whether to shuffle the data.
    :param num_epochs: Number of epochs for which data should be repeated. None = repeat indefinitely
    :return: Tuple of (features, labels) for next data batch.
    """

    # Convert pandas data into a dict of np arrays.
    features = {key:np.array(value) for key, value in dict(features).items()}

    # Construct a dataset, and configure batching/repeating.
    ds = Dataset.from_tensor_slices((features, targets)) # warning: 2GB limit
    ds = ds.batch(batch_size).repeat(num_epochs)

    # Shuffle the data, if specified
    if shuffle:
        ds = ds.shuffle(buffer_size=10000)

    # Return the next batch of data.
    features, labels = ds.make_one_shot_iterator().get_next()
    return features, labels


train_model(learning_rate=0.0001, steps=100, batch_size=1)
