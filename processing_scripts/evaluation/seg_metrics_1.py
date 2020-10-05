import numpy as np

data_file = "/proj/NIRAL/users/nic98/processing_scripts/evaluation/segmetrics_data.txt"

data = []
for line in open(data_file):
    row = line.split(";")
    row = row[2:]
    for i in range(len(row)):
        row[i] = float(row[i])
    data.append(row)

data = np.asarray(data)

print("Number of Samples: ", len(data))
print("Avg (across dataset) sqr distance is: ", np.mean(data[:,7]), " mm")

