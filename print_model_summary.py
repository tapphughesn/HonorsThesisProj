import tensorflow as tf
import numpy as np
import sys
from datetime import datetime
from model import *
from glob import glob
from tensorflow.keras import *

# get file
data_files = sorted(glob("/proj/NIRAL/users/nic98/DATA/pretraining/train_on/data_arrays/*"))

# Get model
model = unet_4()

data_arrays = []
data_arrays.append(np.load(data_files[1]))
data_arrays = np.asarray(data_arrays)
print(np.shape(data_arrays))
prediction = model(data_arrays)

print(model.summary())
