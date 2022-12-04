import random
import matplotlib.pyplot as plt
import numpy as np
import csv


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


def SFAP(fu, fv, fc, sizeofmu):
    """
    :param fu: 表示幅值， 需要由三个包含正负的数组成 例：fu = [-2, 2, 1]
    :param fv: 表示带宽，例：fv = [0.2, 0.3, 0.4]
    :param fc: 表示波峰所在的位置， 例：fc = [1.5, 1.1, 1.2]
    :return: 生成的混合肌肉纤维动作电位sfap信号
    """
    signal = 0
    x = np.linspace(0, 3, sizeofmu)
    for i in range(3):
        signal += fu[i] * Gaussian(x, fc[i], fv[i])
    return signal


def generate_mu(num, sizeofmu):
    """

    :param num:
    :param sizeofmu: 输入每个MU波形由多少数据点构成
    :return:
    """
    fu, fv, fc = [0] * 3, [0] * 3, [0] * 3
    mu_signal = []
    for _ in range(num):
        for i in range(3):
            fu[i] = random.randint(-2, 2)
            fv[i] = random.random()
            fc[i] = random.uniform(1, 2)
        mu_signal.append(SFAP(fu, fv, fc, sizeofmu))
    plt.plot(mu_signal[0])
    plt.show()
    return mu_signal


def save_musfap(data):
    for index, i in enumerate(data):
        with open("D:\semg_simulation_result\MU_sfap{}.csv".format(index), "w") as f:
            writer = csv.writer(f)
            writer.writerow(i)


def read_sfap_data_from_csv(num):
    """
    从预存的CSV文件中读取MU波形数据
    :return:
    """
    for i in range(num):
        all_data = []
        index = i + 1
        with open("/gpfs/home/xieyongbao/biye/MU_sfap{}.csv".format(index), "r") as f:
            read_csv = csv.reader(f)
            for line in read_csv:  # 循环输出csv中的所有数据
                data = list(map(float, line))
            all_data.append(data)
                # plt.plot(data)
                # plt.show()
    return all_data

if __name__ == '__main__':
    mu_signal = generate_mu(5, 1000)
    plt.plot(mu_signal[0])
    plt.show()
    # save_musfap(mu_signal)
