import numpy as np
import scipy.signal
from generate_sfap import generate_mu
from matplotlib import pyplot as plt
from genera_train import genera_train
import csv
import pandas as pd
import openpyxl as op
import xlwt

def conv_train_sfap(sfap, mu_train_data):
    """
    通过把纤维动作电位信号与脉冲序列卷积生成肌电信号
    :param sfap:
    :param mu_train_data:
    :return: 返回由输入的两个信号卷积生成的信号
    """
    new = scipy.signal.convolve(sfap, mu_train_data)
    return new


def everymu_gen_emg(lenght: int, numofmu:int):
    """
    :param lenght:要生成的MUemg由单个MU重复多少次生成，
    :return: 返回信号的长度是 lenght * (每个MU的大小 + 序列的长度 -1)
    allemg中的内容为：[[MU1的纤维动作电位信号与序列卷积生成的肌电信号],
                     [MU2的纤维动作电位信号与序列卷积生成的肌电信号],...]
    MU_train中的内容为：[[MU1的动作电位序列],
                       [MU2的动作电位序列],...]
    MU_sfap中的内容为：[[MU1的信号波形],
                      [MU2的信号波形],...]
    """
    allemg = []
    musemg = []
    MU_train = []
    MU_sfap = []
    MU_sfap_train = []
    sfap = generate_mu(numofmu, 2000)  #输入内容为：MU的数量，每个MU由多少数据点组成
    for musfap in sfap:
        for i in range(lenght):
            mu_train_data = genera_train(2000)
            MU_sfap_train.extend(mu_train_data)
            musemg.extend(conv_train_sfap(musfap, mu_train_data))
        allemg.append(musemg)
        MU_sfap.append(musfap)
        MU_train.append(MU_sfap_train)
    return allemg, MU_train, MU_sfap


def genera_semg(muemg):
    """
    将多个MU产生的emg信号叠加生成新的信号
    :param muemg:
    :return:
    """
    signal = muemg[0]
    for i in muemg[1:]:
        signal = np.array(i) + np.array(signal)
    return signal

def genera_semg_multichannel_delay(muemg, every_channel_delay_num):
    """
    将多个MU产生的emg信号叠加生成新的信号
    :param muemg:
    :every_channel_delay_num: 每个通道的肌电信号延迟多少个数据点
    :return:
    """
    signal = muemg[0]
    for i in muemg[1:]:
        for _ in range(every_channel_delay_num):
            i.insert(0, 0)
        data = i[0:-every_channel_delay_num]
        signal = np.array(data) + np.array(signal)
    return signal



# if __name__ == '__main__':
semg_simulation, MU_train, MU_sfap = everymu_gen_emg(150, 10)
emg = genera_semg_multichannel_delay(semg_simulation, 10)
# with open("D:\semg_simulation_result\ train_data.csv", "w") as f:
#     writer = csv.writer(f)
#     for index, i in enumerate(MU_train):
#         writer.writerow(i)
# with open("D:\semg_simulation_result\MU_sfap.csv", "w") as f:
#     writer = csv.writer(f)
#     for index, i in enumerate(MU_sfap):
#         writer.writerow(i)

print(len(semg_simulation))
plt.plot(emg[0:20000])
plt.show()
# t = np.linspace(0,20000,20000)
# plot_1d_spikes(np.array([data]).T, "SEMG_simulation", "lenght", "height")
# plt.show()
