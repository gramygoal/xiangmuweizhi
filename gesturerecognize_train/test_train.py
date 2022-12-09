import os
import re

from extra_features.feature_utils import *


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


def load_label(path, task_type):
    label = []
    with open(path + '/label_' + task_type + '.txt') as f:
        for line in f:
            label.append(list(map(int, line.strip().split(','))))
    return label[0]


def load_PR_data(path, subject, session, task_type, sig_type):
    filename_path = 'pr_dataset/subject' + subject + '_session{}'.format(session)
    label = load_label(path + filename_path, task_type)
    data_name = task_type + '_' + sig_type
    file_name = re_find_in_dir(path + filename_path, data_name)
    data_real = [([0.0]) for i in range(len(file_name))]
    for i in range(len(file_name)):
        data_name = path + 'pr_dataset/subject' + subject + '_session{}/'.format(
            session) + task_type + '_' + sig_type + '_sample{}.dat'.format(i + 1)
        data = np.fromfile(data_name, dtype='i2')
        data_tmp = data.reshape(2048, 256)
        data_tmp = np.array(data_tmp, dtype=float)
        info_name = path + 'pr_dataset/subject' + subject + '_session{}/'.format(
            session) + task_type + '_' + sig_type + '_sample{}.hea'.format(i + 1)
        fp = open(info_name, "r")
        info_data = []
        for line in fp.readlines():
            # 当你读取文件数据时会经常遇见一种问题，
            # 那就是每行数据末尾都会多个换行符‘\n’，
            # 所以我们需要先把它们去掉
            line = line.replace('\n', '')
            # 或者line=line.strip('\n')
            # 但是这种只能去掉两头的，可以根据情况选择使用哪一种
            line = line.split()
            # 以逗号为分隔符把数据转化为列表
            info_data.extend(line)
        fp.close()

        c = task_type + '_' + sig_type + '_sample' + str(i + 1) + '.dat'
        idx = [i for i, v in enumerate(info_data) if v == c]

        for u in range(len(idx)):
            str_tmp = info_data[idx[u] + 2]
            gain_baseline = info_data[idx[u] + 2]
            x = gain_baseline.index('(')
            y = gain_baseline.index(')')
            baseline = int(gain_baseline[int(x + 1):int(y)])
            gain = float(gain_baseline[:int(x)])
            data_tmp[:, u] = (data_tmp[:, u] - baseline) / gain
        array2 = data_tmp[:, 64:128]
        array4 = data_tmp[:, 192:256]
        array_conb = np.hstack((array2, array4))
        data_real[i] = array_conb
    return data_real, label


def windwos_data(data, label):
    sig_start = 0.25  # remove the first 0.25s startup duration.
    step_len = 0.125
    window_len = 0.25
    fs_emg = 2048
    start_pos = int(sig_start * fs_emg)
    step = int(step_len * fs_emg)
    featureData = []
    featureLabel = []
    for j in range(len(data)):
        stop = int(len(data[j]) - window_len * fs_emg)
        for t in range(start_pos, stop, step):
            pre_data = data[j]
            sig = pre_data[t: t + int(np.floor(window_len * fs_emg)), :]
            rms = featureRMS(sig)
            mav = featureMAV(sig)
            wl = featureWL(sig)
            zc = featureZC(sig)
            ssc = featureSSC(sig)
            featureStack = np.hstack((rms, mav, wl, zc, ssc))
            featureData.append(featureStack)
            featureLabel.append(label[j])
    return featureData, featureLabel


def load_all_data():
    subject = ['01', '02', '03', '04', '05', '06', '07', '08', '09', '10', '11', '12', '13', '14', '15', '16', '17',
               '18', '19', '20']
    session = [1, 2]
    path = 'D:/physionet.org/files/hd-semg/1.0.0/'
    all_data = []
    all_label = []
    for sub in subject:
        for se in session:
            data, label = load_PR_data(path, sub, session, 'dynamic', 'preprocess')
            featureData, featureLabel = windwos_data(data, label)
            all_data.extend(featureData)
            all_label.extend(featureLabel)
    return all_data, all_label


if __name__ == '__main__':
    path = 'D:/physionet.org/files/hd-semg/1.0.0/'
    data_path = 'D:/physionet.org/files/hd-semg/1.0.0/pr_dataset/'
    data, label = load_PR_data(path, '01', 1, 'dynamic', 'preprocess')
    aa = data[0]
    featureData, featureLabel = windwos_data(data, label)
    print(np.array(featureData).shape)
    print(np.array(featureLabel).shape)
    print(label)
