#!/usr/bin/env python
# coding: utf-8

# In[7]:


from matplotlib import pyplot as plt
import matplotlib
import numpy as np

#############################################################显示函数###############################################
def plot_1d_spikes(spikes: np.asarray,title: str, xlabel: str, ylabel: str, int_x_ticks=True, int_y_ticks=True,
                   plot_firing_rate=True, firing_rate_map_title='Firing Rate', dpi=200, Fs=2048,):
    '''


    :param spikes: shape=[T, N]的np数组，其中的元素只为0或1，表示N个时长为T的脉冲数据
    :param title: 热力图的标题
    :param xlabel: 热力图的x轴的label
    :param ylabel: 热力图的y轴的label
    :param int_x_ticks: x轴上是否只显示整数刻度
    :param int_y_ticks: y轴上是否只显示整数刻度
    :param plot_firing_rate: 是否画出各个脉冲发放频率
    :param firing_rate_map_title: 脉冲频率发放图的标题
    :param dpi: 绘图的dpi
    :return: 绘制好的figure

    画出N个时长为T的脉冲数据。可以用来画N个神经元在T个时刻的脉冲发放情况，示例代码：

    .. code-block:: python

        import torch
        from spikingjelly.clock_driven import neuron
        from spikingjelly import visualizing
        from matplotlib import pyplot as plt
        import numpy as np

        lif = neuron.LIFNode(tau=100.)
        x = torch.rand(size=[32]) * 4
        T = 50
        s_list = []
        v_list = []
        for t in range(T):
            s_list.append(lif(x).unsqueeze(0))
            v_list.append(lif.v.unsqueeze(0))

        s_list = torch.cat(s_list)
        v_list = torch.cat(v_list)

        visualizing.plot_1d_spikes(spikes=np.asarray(s_list), title='Membrane Potentials', xlabel='Simulating Step',
                                   ylabel='Neuron Index', dpi=200)
        plt.show()

    .. image:: ./_static/API/visualizing/plot_1d_spikes.*
        :width: 100%

    '''
    if spikes.ndim != 2:
        raise ValueError(f"Expected 2D array, got {spikes.ndim}D array instead")
    spikes_T = spikes.T
    if plot_firing_rate:
        fig = plt.figure(tight_layout=True, dpi=dpi)
        gs = matplotlib.gridspec.GridSpec(1, 5)
        spikes_map = fig.add_subplot(gs[0, 0:4])
        firing_rate_map = fig.add_subplot(gs[0, 4])
    else:
        fig, spikes_map = plt.subplots()

    spikes_map.set_title(title)
    spikes_map.set_xlabel(xlabel)
    spikes_map.set_ylabel(ylabel)

    spikes_map.xaxis.set_major_locator(matplotlib.ticker.MaxNLocator(integer=int_x_ticks))
    spikes_map.yaxis.set_major_locator(matplotlib.ticker.MaxNLocator(integer=int_y_ticks))

    spikes_map.xaxis.set_minor_locator(matplotlib.ticker.NullLocator())
    spikes_map.yaxis.set_minor_locator(matplotlib.ticker.NullLocator())

    spikes_map.set_xlim(-0.5, spikes_T.shape[1] - 0.5)
    spikes_map.set_ylim(-0.5, spikes_T.shape[0] - 0.5)
    spikes_map.invert_yaxis()
    N = spikes_T.shape[0]
    print(N)
    T = spikes_T.shape[1]
    t = np.arange(0, T)
    t_spike = spikes_T * t
    mask = (spikes_T == 1)  # eventplot中的数值是时间发生的时刻，因此需要用mask筛选出

    colormap = plt.get_cmap('tab10')  # cmap的种类参见https://matplotlib.org/gallery/color/colormap_reference.html

    for i in range(N):
        spikes_map.eventplot(t_spike[i][mask[i]], lineoffsets=i, colors=colormap(i % 10))

    if plot_firing_rate:
        spike_len=spikes_T.shape
        len1=spike_len[1]
        len2=spike_len[0]
        F=np.sum(spikes_T,axis=1)
        fir_rate=F/(len1/Fs)
        fir_rate=fir_rate.reshape(len2,1)
#         firing_rate = np.mean(spikes_T, axis=1, keepdims=True)

        max_rate = fir_rate.max()
        min_rate = fir_rate.min()

        firing_rate_map.yaxis.set_major_locator(matplotlib.ticker.MaxNLocator(integer=True))
        firing_rate_map.yaxis.set_minor_locator(matplotlib.ticker.NullLocator())
        firing_rate_map.imshow(fir_rate, cmap='magma', aspect='auto')
        for i in range(fir_rate.shape[0]):
            firing_rate_map.text(0, i, f'{fir_rate[i][0]:.2f}', ha='center', va='center', color='w' if fir_rate[i][0] < 0.7 * max_rate or min_rate == max_rate else 'black')
        firing_rate_map.get_xaxis().set_visible(False)
        firing_rate_map.set_title(firing_rate_map_title)
    return fig


if __name__ == '__main__':
     plot_1d_spikes()


# In[ ]:




