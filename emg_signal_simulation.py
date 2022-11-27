import scipy.signal
from generate_sfap import generate_mu
from matplotlib import pyplot as plt
from genera_train import genera_train


def generate_semg(sfap, mu_train_data):
    new = scipy.signal.convolve(sfap, mu_train_data)
    # semg_simulation = np.convolve(sfap, mu_train_data)#semg_simulation
    return new


def everymu_gen_emg(lenght: int):
    """
    :param lenght:要生成信号的长度，
    :return: 返回信号的长度是 lenght * (200 + 200 -1)
    """
    semg = []
    sfap = generate_mu(1)
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
