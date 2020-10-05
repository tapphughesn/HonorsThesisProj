import numpy as np
import re
import matplotlib.pyplot as plt

data_dir = "/proj/NIRAL/users/nic98/models/unet_4/logfiles/"
csvfile = "loss_record_10-03-2020-12:17:30.csv"

logtime = csvfile[12:-4]

f = open(data_dir + csvfile, "r")
data = []

for line in f:
    if (re.search("[A-Z]|[a-z]", line)):
        continue
    a = line.split(",")
    b = np.zeros(len(a))
    for i in range(len(a)):
        b[i] = float(a[i])
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

plt.subplot(211)
plt.plot(batch_indices[::10], data[::10,2])
plt.title("Training Loss over Time " + logtime)
plt.xlabel("batches")
plt.ylabel("training loss")

plt.subplot(212)
plt.plot(batch_indices[::10], data[::10,3])
plt.title("Training Categorical Accuracy")
plt.xlabel("batches")
plt.ylabel("Categorical Accuracy")

plt.tight_layout(pad=2.0)

plt.savefig("/proj/NIRAL/users/nic98/models/unet_4/loss_hist_fig_" + logtime + ".png")
plt.show()

# loss_record_10-03-2020-01:07:29.csv
# loss_record_10-03-2020-10:48:45.csv
# loss_record_10-03-2020-10:48:52.csv
# loss_record_10-03-2020-12:17:30.csv
# loss_record_10-03-2020-20:54:22.csv
