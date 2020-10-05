import numpy as np
import re
import matplotlib.pyplot as plt

data_dir = "/proj/NIRAL/users/nic98/models/unet_2/logfiles/"
logfile = "loss_record_06-26-2020-08:25:40.log"

logtime = logfile[12:-4]

f = open(data_dir + logfile, "r")
data = []

for line in f:
    if (re.search("[A-Z] | [a-z]", line)):
        continue
    a = re.findall('\d+', line)
    b = np.zeros(len(a))
    for i in range(len(a)):
        b[i] = int(a[i])
    data.append(b)
    

data = np.asarray(data)
print(np.shape(data))
num_batches = int(np.max(data[:,1]) + 1)
num_epochs = int(np.max(data[:,0]) + 1)
batch_indices = np.arange(num_batches * num_epochs)

avg_losses = np.zeros(num_epochs)

for i in range(num_epochs):
    avg_loss = np.mean(data[i*num_batches : (i + 1)*num_batches, 2])
    avg_losses[i] = avg_loss

print(avg_losses)

plt.plot(batch_indices[::10], data[::10,2])
plt.title("Training Loss over Time " + logtime)
plt.xlabel("batches")
plt.ylabel("training loss")
plt.savefig("/proj/NIRAL/users/nic98/processing_scripts/evaluation/loss_hist_fig_" + logtime + ".png")
plt.show()

# loss_record_06-17-2020-13:17:49.log
# loss_record_06-17-2020-13:35:44.log
# loss_record_06-23-2020-14:12:24.log
# loss_record_06-23-2020-14:13:01.log
# loss_record_06-23-2020-14:15:30.log
# loss_record_06-25-2020-10:09:21.log
# loss_record_06-25-2020-10:09:30.log
# loss_record_06-25-2020-10:09:41.log
# loss_record_06-26-2020-08:25:40.log
