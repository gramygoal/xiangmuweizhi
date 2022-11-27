# 这是一个示例 Python 脚本。

# 按 Shift+F10 执行或将其替换为您的代码。
# 按 双击 Shift 在所有地方搜索类、文件、工具窗口、操作和设置。


import socket
from matplotlib import pyplot as plt
import numpy as np
def ReadFromOTBiolabLightQuattrocento(filename):
    nCh = 384 + 16 + 8; # set the number of channels required
    channelToPlot = 384 + 16 + 1;# set the channel to plot(ramp)
    fSample = 2048; # set sampling frequency
    fRead = 16; # set reading frequency
    nCycles = 30 * fRead; # set number of read
    timeSize = 2;
    # plt.plot()
    tRead = np.zeros(1, nCycles)
    dataAvailable = np.zeros(1, nCycles)
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)# 套接字类型AF_INET, socket.SOCK_STREAM   tcp协议，基于流式的协议
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)  # 对socket的配置重用ip和端口号
    sock.bind(('localhost', 31000)); #  写哪个ip就要运行在哪台机器上
    sock.listen(128)
    client, addr = sock.accept()
    data = client.recv(nCh*fSample*10)
    with open(filename, 'a') as f:
        f.write(data)
    sock.close()
    # for i in range(nCycles):
    #     with open(filename, 'a') as f:
            #需要再改f.read()

def live_plotter(x_vec, y_vec_data, line_realtime, identifier='', pause_time=0.1):
    if line_realtime == []:
        # this is the call to matplotlib that allows dynamic plotting
        plt.ion()
        fig = plt.figure(figsize=(13, 6))
        ax = fig.add_subplot(111)
        # create a variable for the line so we can later update it
        line_realtime, = ax.plot(x_vec, y_vec_data, '-o', alpha=0.8)
        # update plot label/title
        plt.ylabel('Y Label')
        plt.title('Title: {}'.format(identifier))
        plt.show()

    # after the figure, axis, and line are created, we only need to update the y-data
    line_realtime.set_xdata(x_vec)
    line_realtime.set_ydata(y_vec_data)
    # line_realtime.set_data(x_vec,y_vec_data)
    # adjust limits if new data goes beyond bounds
    if np.min(y_vec_data) <= line_realtime.axes.get_ylim()[0] or np.max(y_vec_data) >= \
            line_realtime.axes.get_ylim()[1]:
        plt.ylim([np.min(y_vec_data) - np.std(y_vec_data), np.max(y_vec_data) + np.std(y_vec_data)])
    # this pauses the data so the figure/axis can catch up - the amount of pause can be altered above
    plt.pause(pause_time)

    # return line so we can update it again in the next iteration
    return line_realtime
def xianshihanshu():
    print(__file__ + " start...")
    size = 100
    x_vec = np.linspace(0,1,size+1)[0:-1]
    y_vec = np.random.randn(len(x_vec))
    line_realtime = []
    while True:
        rand_val = np.random.randn(1)
        y_vec[-1] = rand_val
        line_realtime = live_plotter(x_vec,y_vec,line_realtime)
        y_vec = np.append(y_vec[1:],0.0)


# 按间距中的绿色按钮以运行脚本。
if __name__ == '__main__':
    xianshihanshu()

# 访问 https://www.jetbrains.com/help/pycharm/ 获取 PyCharm 帮助
