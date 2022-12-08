import math

from feature_utils import *

featureData = []
featureLabel = []
classes = 16
timeWindow = 200
strideWindow = 200

for i in range(classes):
    index = [];
    for j in range(label.shape[0]):
        if (label[j, :] == i):
            index.append(j)
    iemg = emg[index, :]
    length = math.floor((iemg.shape[0] - timeWindow) / strideWindow)
    print("class ", i, ",number of sample: ", iemg.shape[0], length)

    for j in range(length):
        rms = featureRMS(iemg[strideWindow * j:strideWindow * j + timeWindow, :])
        mav = featureMAV(iemg[strideWindow * j:strideWindow * j + timeWindow, :])
        wl = featureWL(iemg[strideWindow * j:strideWindow * j + timeWindow, :])
        zc = featureZC(iemg[strideWindow * j:strideWindow * j + timeWindow, :])
        ssc = featureSSC(iemg[strideWindow * j:strideWindow * j + timeWindow, :])

        featureStack = np.hstack((rms, mav, wl, zc, ssc))

        featureData.append(featureStack)
        featureLabel.append(i)
featureData = np.array(featureData)

print(featureData.shape)
print(len(featureLabel))
