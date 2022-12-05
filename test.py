import random
import numpy as np
import pandas as pd
# from displayMUAPT import plot_1d_spikes
import math
# from matplotlib import pyplot as plt
import scipy.io as scio
import matlab.engine


def makeSpikeTrain(MU, t, recthresh, neuraldrive, g, lambdamin, lambdamax, dt):
    if recthresh[MU] < neuraldrive[t]:
        lambda_data = g * (neuraldrive[t] - recthresh[MU]) + lambdamin
        lambda_data = min(lambda_data, lambdamax)
        # ISI exponentially distributed
        exprn = -np.log(random.uniform(0, 1)) / lambda_data
        spike = exprn < dt
    else:
        spike = False
    return spike


def train_with_isi(MUAP, isimode):
    """
    :param MUAP: [n, m] number and length of MUAPs
    :param isimode: 'gauss', 'exponential', 'weibull'
    :return:
    """
    # shapeofmuap = [15, 200]
    M, L = np.array(MUAP).shape
    T = 100
    # M, L = shapeofmuap[0], shapeofmuap[1]
    dt = 16 / L * 10 ** (-1)
    isimode = 'gauss'
    # temporal resolution in [s] defined by the length of MUAP,
    # 256 data points in x ms (if x=8, sampling frequency = 32kHz)
    # 时间分辨率（单位：[s]），由MUAP长度定义，256个数据点（单位：x ms）
    # （如果x=8，采样频率=32kHz）
    twitch = np.ones((M, L))
    twitch[:, -1] = 0
    twitch[:, 1] = 0
    MUs = np.linspace(1, M, M)
    time = np.linspace(1, round(T / dt), round(T / dt))
    N = len(time)
    a = math.log(30) / M
    recthresh = np.zeros(M, dtype=np.int64)
    lambdamin = 0  # in [Hz]
    lambdamax = 100  # in [Hz]
    g = 1  # gain between neural drive and MUAP firing rate
    ndmax = recthresh[-1] * g + lambdamax  # maximum neural drive
    neuraldrive = np.linspace(0, 1, N) * ndmax
    for x in [5]:
        sEMG = np.zeros((N + L))
        force = np.zeros((N + L))
        spikes = np.zeros(N)
        reps = math.ceil((N * M) / float(2 ** 31))  #
        N_p = int(16000/1)  #
        # ISI distributed according to a Gaussian
        # 根据高斯分布的ISI
        if isimode == 'gauss':
            x_in = x ** -1
            gauss_st = np.cumsum(x_in + x_in * 0.2 * np.random.randn(T * x * 2, M), axis=0)
        for r in range(reps):
            r += 1  # 将r从0变为1
            st = np.full((M, N_p), False, dtype=bool)
            if isimode == 'gauss':
                if reps == 1:
                    tmp = []
                    eng = matlab.engine.start_matlab()
                    gauss_st = matlab.double(gauss_st)
                    eng.workspace["M"] = M
                    tmp = eng.histc(gauss_st,time[N_p * (r - 1):N_p * r] * dt)
                    eng.workspace["tmp"] = tmp
                    eng.eval('tmp(end,:) = zeros(1,M);', nargout=0)
                    tmp = eng.workspace["tmp"]
                    eng.eval('st = tmp\' > 0;', nargout=0)
                    st = eng.workspace["st"]
                    eng.quit()
                    # for i in range(np.array(gauss_st).shape[1]):
                    #     hist, bins = np.histogram(gauss_st[:][i], bins = time[N_p * (r - 1):N_p * r] * dt)
                    #     tmp.append(hist)
                    # print('tmpshape',np.array(tmp[:][-1]).shape)
                    # tmp[:][-1] = np.zeros([1, M])
                elif r == 1:
                    tmp = []
                    for i in range(np.array(gauss_st).shape[1]):
                        hist, bins = np.histogram(gauss_st, time[N_p * (r - 1):N_p * r] * dt)
                        tmp.append(hist)
                    tmp[-1, :] = []
                elif r == reps and reps * N_p == N:
                    tmp = []
                    for i in range(np.array(gauss_st).shape[1]):
                        hist, bins = np.histogram(gauss_st, time[N_p * (r - 1):N_p * r] * dt)
                        tmp.append(hist)
                    tmp[1, :] = []
                    tmp[-1, :] = np.zeros(1, M)
                else:
                    tmp = []
                    for i in range(np.array(gauss_st).shape[1]):
                        hist, bins = np.histogram(gauss_st, time[N_p * (r - 1):N_p * r] * dt)
                        tmp.append(hist)
                    tmp[1:-1] = []
                # st = tmp > 0  #.T
                # print('tmp',np.array(tmp[0]).shape)
                # tmp = np.array(tmp)
                # st = [(i > 0) + 0 for i in tmp.T]
                stt = []
                for i in st:
                    stt.append(list(map(int, i)))
                st = stt
            elif isimode == 'exponential':
                st = makeSpikeTrain(MUs, time[N_p * (r - 1) + 1:N_p * r], recthresh, neuraldrive, g, lambdamin,
                                    lambdamax, dt)
            elif isimode == 'weibull':
                pass
        # print('stshape',np.array(st).shape)
        st = np.array(st)
        for t in range(N_p * (r - 1) , N_p * r):
            # print('aaaa',np.array(st[:,t - N_p * (r - 1)]).shape)
            sEMG[t: t + L] = sEMG[t: t + L] + sum(MUAP[st[:,t - N_p * (r - 1)],:], 1)
            force[t: t + L] = force[t: t + L] + sum(twitch[st[:,t - N_p * (r - 1)], :], 1)
            spikes[t] = sum(st[:,t - N_p * (r - 1)])

        rest = N - reps * N_p
        print('%d, %d, %d\n', rest, reps * N_p, N)
        if rest > 0:
            st = np.full((M, N_p), False, dtype=bool)
            if isimode == 'gauss':
                hist, bins = np.histogram(gauss_st, time[N_p * (r - 1):N_p * r] * dt)
                tmp.append(hist)
                tmp[1,:] = []
                tmp[-1] = np.zeros([1, M])
                st = tmp.T > 0
            else:
                st = makeSpikeTrain(MUs, time[N_p * (r - 1) + 1:N_p * r], recthresh, neuraldrive, g, lambdamin,
                                    lambdamax, dt)
                for t in range(N_p * reps+1,N):
                    sEMG[1, t: t + L - 1] = sEMG[1, t: t + L - 1] + sum(MUAP[st[:, t - N_p * reps],:], 1)
                    # force(1, t: t + L - 1) = force(1, t: t + L - 1) + sum(twitch(st(:, t - N_p * reps),:), 1);
                    spikes[t] = sum(st[:, t - N_p * (r - 1)])
    return spikes
            # % spiketimes = gather(gauss_st);
            # time = gather(time);
if __name__ == '__main__':
    data_path = 'D:/biyexiangguansuanfa/EMGsim-master/EMGsim-master/Codes/MUAPs_Francesco.mat'
    MUAP = scio.loadmat(data_path)
    muapdata = MUAP['MUAPs']
    # print(np.array(muapdata[0:100][:]).shape)
    spikes = train_with_isi(muapdata[0:100][:], 'gauss')
    spikes_path = 'D:/biyexiangguansuanfa/EMGsim-master/EMGsim-master/Codes/spikes.mat'
    scio.savemat(spikes_path, {'spikes': spikes})