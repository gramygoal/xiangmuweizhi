import random
import numpy as np
from displayMUAPT import plot_1d_spikes
import math
from matplotlib import pyplot as plt


# Defining delta function by rectangles候选delta脉冲函数
def delta_function(x, eps):
    result = np.zeros_like(x)
    for index, xi in enumerate(x):
        print("x:{}, eps:{}".format(xi, eps))
        if -eps / 8.0 < xi < eps / 8.0:
            result[index] = 1.0  # / eps
        else:
            result[index] = 0
    return result


def delta_function2(x, eps):
    result = np.zeros_like(x)
    pos = random.randint(0, len(x))
    # print(pos)
    for index, xi in enumerate(x):
        #  / 2
        if index == pos:
            result[index] = 1.0  # / eps
        else:
            result[index] = 0
    return result


def genera_delta_train():
    a = 1.0
    x1, x2 = -0.1 + a, 0.1 + a
    eps_ = 0.1
    x_ = np.linspace(x1, x2, 100)
    n_val = np.arange(1, 5)
    for i in n_val:
        eps_ = eps_ / i
    delta = delta_function2(x_ - a, eps_)
    # plt.plot(x_, delta)
    # plt.show()
    return delta


def genera_train(num: int):
    """
    :param num: 要生成的脉冲序列由多少个脉冲构成
    :return:
    """
    num = int(num / 100)
    result = []
    for i in range(num):
        result.extend(genera_delta_train())
    return result


def display_train(data):
    """
    可以直接显示由genera_train生成的数据
    :param data:
    :return:
    """
    plot_1d_spikes(np.array([data]).T, "SEMG_simulation", "lenght", "height")
    plt.show()
