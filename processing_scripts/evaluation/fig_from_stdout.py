import numpy as np
import re
import matplotlib.pyplot as plt

data_dir = "/proj/NIRAL/users/nic98/models/unet_2/logfiles/"
data_file = "UNET_2_run-61901773.log"

f = open(data_dir + data_file, "r")

data = []
batch = []

for line in f:
    if (re.match("LOSS", line)):
        a = re.findall('\d+', line)
        b = np.zeros(len(a))
        for i in range(len(a)):
            b[i] = int(a[i])
        data.append(b)
    elif (re.match("Load", line)):
        a = re.findall('\d+', line)
        b = (int(a[0])*415 + int(a[1])) / 415.0
        if (int(a[0]) == 10):
            break
        batch.append(b)

data = np.asarray(data)
batch = np.asarray(batch)

plt.plot(batch[::10], data[::10,0])
plt.title("Training Loss over Time " + data_file)
plt.savefig("/proj/NIRAL/users/nic98/processing_scripts/evaluation/loss_hist_fig_" + data_file + ".png")
plt.show()

# UNET_2_run-61901771.log
# UNET_2_run-61901772.log
# UNET_2_run-61901773.log
