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

# 分解一个数为两个素数之积
# 统计从1开始到指定数的奇数序列，有多少个3
# ======================================================================================
#
#
# ======================================================================================

import math
import time
from absl import app


def is_prime(number):
    if number < 2:
        return False
    elif number == 2:
        return True
    else:
        for i in range(3, number, 2):
            if number % i == 0:
                return False

    return True


def cal_three(target):
    origin = target
    three_count = 0
    his_value = 0
    his_count = 1

    index = 0
    while target > 0:
        m = target % 10

        if index == 0:
            if m >= 3:
                three_count += 1
        else:
            three_count += m * his_count
            if m > 3:
                three_count += math.pow(10, index) / 2
            elif m == 3:
                three_count += int(his_value / 2)
            his_count = 10 * his_count + math.pow(10, index) / 2

        target = int(target / 10)
        his_value += m * pow(10, index)
        index += 1

    print("three_count is: ({0}, {1})".format(origin, int(three_count)))


def main(argv):
    del argv

    target = 707829217
    middle = int(math.sqrt(target))

    if middle % 2 == 0:
        middle = middle + 1

    print(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))
    for i in range(3, middle, 2):
        # print("i = {0}".format(i))
        if not is_prime(i):
            continue

        if target % i == 0:
            print("{0} * {1} = {2}".format(int(target / i), i, target))
            break

    print(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))

    cal_three(866278171)

    print(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))
    print("complete")
    exit(0)


if __name__ == '__main__':
    app.run(main)
