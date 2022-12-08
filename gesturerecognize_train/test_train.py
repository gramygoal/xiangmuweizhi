import os
import re

import numpy as np


def re_find_in_dir(path: str = '', pattern: str = ''):
    """
    在指定目录下，查找符合规则的目录、文件。规则有多个时，拼接成 '*a*b' 进行匹配
    :param path: 指定目录
    :param pattern: 匹配规则
    :return: 符合规则的结果
    """
    match_file = []
    str1 = re.compile(pattern + '''(.*?).dat''')
    file_list = os.listdir(path)
    for i in file_list:
        match_obj = re.findall(str1, i)
        if match_obj:
            match_file.append(i)
    return match_file


def load_PR_data(path, subject, session, task_type, sig_type):
    filename_path = 'pr_dataset/subject' + subject + '_session{}'.format(session)
    data_name = task_type + '_' + sig_type
    file_name = re_find_in_dir(path + filename_path, data_name)
    data_real = [([0.0]) for i in range(len(file_name))]
    for i in range(len(file_name)):
        data_name = path + 'pr_dataset/subject' + subject + '_session{}/'.format(
            session) + task_type + '_' + sig_type + '_sample{}.dat'.format(i + 1)
        data = np.fromfile(data_name, dtype='i2')
        data_tmp = data.reshape(2048, 256)
        # data_tmp = np.array(data_tmp, dtype=float)
        info_name = path + 'pr_dataset/subject' + subject + '_session{}/'.format(
            session) + task_type + '_' + sig_type + '_sample{}.hea'.format(i + 1)
        fp = open(info_name, "r")
        info_data1 = []
        for line in fp.readlines():
            '''
            当你读取文件数据时会经常遇见一种问题，
            那就是每行数据末尾都会多个换行符‘\n’，
            所以我们需要先把它们去掉
            '''
            line = line.replace('\n', '')
            # 或者line=line.strip('\n')
            # 但是这种只能去掉两头的，可以根据情况选择使用哪一种
            line = line.split()
            # 以逗号为分隔符把数据转化为列表
            info_data1.append(line)
        fp.close()
        info_data = []
        for k in info_data1:
            info_data.extend(k)
        c = task_type + sig_type + '_sample' + str(i + 1) + '.dat'
        idx = []
        a = 0
        for l in info_data:
            if l == c:
                idx.append(a)
            a += 1
        for u in range(len(idx)):
            str_tmp = info_data[idx[u] + 2]
            gain_baseline = info_data[idx[u] + 2]
            x = gain_baseline.index('(')
            y = gain_baseline.index(')')
            baseline = int(gain_baseline[int(x + 1):int(y)])
            gain = float(gain_baseline[:int(x)])
            data_tmp[:, u] = (data_tmp[:, u] - baseline) / gain
        data_real[i] = data_tmp
    return data_real


def load_label(path):
    label = []
    with open(path + 'label_dynamic.txt') as f:
        for line in f:
            label.append(list(map(int, line.strip().split(','))))
    return label[0]


def windwos_data(data):
    sig_start = 0.25  # remove the first 0.25s startup duration.
    step_len = 0.125
    window_len = 0.25
    fs_emg = 2048
    for j in range(len(data)):
        for t in range(sig_start * fs_emg, len(data[j]) - window_len * fs_emg, step_len * fs_emg):
            sig = data[j]
            pass


if __name__ == '__main__':
    path = 'D:/physionet.org/files/hd-semg/1.0.0/'
    data_path = 'D:/physionet.org/files/hd-semg/1.0.0/pr_dataset/'
    data = load_PR_data(path, '01', 1, 'dynamic', 'preprocess')
    label = load_label(path)
    print(label)
