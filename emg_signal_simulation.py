import scipy.signal
from generate_sfap import generate_mu
from matplotlib import pyplot as plt
from genera_train import genera_train


def generate_semg(sfap, mu_train_data):
    """
    通过把纤维动作电位信号与脉冲序列卷积生成肌电信号
    :param sfap:
    :param mu_train_data:
    :return: 返回由输入的两个信号卷积生成的信号
    """
    new = scipy.signal.convolve(sfap, mu_train_data)
    return new


def everymu_gen_emg(lenght: int):
    """
    :param lenght:要生成信号的长度，
    :return: 返回信号的长度是 lenght * (200 + 200 -1)
    """
    semg = []
    sfap = generate_mu(1, 200)
    for musfap in sfap:
        for i in range(lenght):
            mu_train_data = genera_train(200)
            semg.extend(generate_semg(musfap, mu_train_data))
    return semg


if __name__ == '__main__':
    semg_simulation = everymu_gen_emg(5)
    print(len(semg_simulation))
    plt.plot(semg_simulation)
    plt.show()
    # t = np.linspace(0,20000,20000)
    # plot_1d_spikes(np.array([data]).T, "SEMG_simulation", "lenght", "height")
    # plt.show()
