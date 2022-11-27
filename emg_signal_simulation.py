import copy
import math
import random
import scipy.signal
from displayMUAPT import plot_1d_spikes

import numpy as np
from matplotlib import pyplot as plt

def Gaussian(x, u, d):
    """
    x -- 变量
    u -- 均值
    d -- 标准差
    返回:
    p -- 高斯分布值
    """
    d_2 = d * d * 2
    zhishu = -(np.square(x - u) / d_2)
    exp = np.exp(zhishu)
    pi = np.pi
    xishu = 1 / (np.sqrt(2 * pi) * d)
    p = xishu * exp
    return p

def SFAP():
    """
    Single Fiber Action Potential, SFAP
    :return:
    """
    signal = 0
    fx, fy, fz, fcv = 1, 1, 1, 4
    fu = [-2, 2, 1]
    fv = [0.2, 0.3, 0.4]
    fc = [1.5, 1.1, 1.2]
    x = np.linspace(0,3,20000)
    for i in range(3):
        signal += fu[i] * Gaussian(x, fc[i], fv[i])
    return signal
# Defining delta function by rectangles

def delta_function(x, eps):
    result = np.zeros_like(x)
    for index, xi in enumerate(x):
        print("x:{}, eps:{}".format(xi, eps))
        if -eps / 8.0 < xi < eps / 8.0:
            result[index] = 1.0 #/ eps
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
            result[index] = 1.0 #/ eps
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

def genera_train(num:int):
    num = int(num / 100)
    result = []
    for i in range(num):
        result.extend(genera_delta_train())
    return result

if __name__ == '__main__':
    data = genera_train(20000)
    t = np.linspace(0,20000,20000)
    plot_1d_spikes(np.array([data]).T, "SEMG_simulation", "lenght", "height")
    plt.show()









