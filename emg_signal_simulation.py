import copy
import math
import random
import scipy.signal

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
    for index, xi in enumerate(x):
        # pos = eps / 2
        if -eps / 2 < xi < eps / 2:
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
    plt.plot(x_, delta)
    plt.show()
    return delta

# def fire_train():
#     x1, x2 = -0.1 + 20000, 0.1 + 20000
#     train_ = np.linspace(x1, x2, 20000)
#     # for n in range(20000):
#     #     for k in range(1,201):
#     #         si_k = np.random.uniform(-10, 10, size=None)
#     delta_function(train_, n - 100 * k + si_k)
#     return train_


def MUAPT():
    t = np.linspace(0,10,2000)
    print(np.array(t).shape)
    for k in range(200):
        pass
if __name__ == '__main__':
    # MUAPT()
    # data = SFAP()
    # x = np.linspace(0, 3, 20000)
    # train = fire_train()
    # # print(train)
    # plt.plot(train)
    # plt.show()
    data = genera_delta_train()
    print(np.array(data).shape)









