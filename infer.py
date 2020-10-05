import tensorflow as tf
import os
import nibabel as nib
import pickle
import random
import SimpleITK as sitk
import numpy as np
from datetime import datetime
from model import *
from glob import glob

# model_checkpoint_load_path = "/proj/NIRAL/users/nic98/models/unet_2/checkpoints/parameters_" + "06-26-2020-08:25:40"
model_checkpoint_load_path = "/proj/NIRAL/users/nic98/models/unet_2/checkpoints/parameters_" + "09-13-2020-23:18:36"

data_files = sorted(glob("/proj/NIRAL/users/nic98/DATA/training/train_on/data_arrays/*"))
label_files = sorted(glob("/proj/NIRAL/users/nic98/DATA/training/train_on/label_arrays/*"))

# Get model
model = unet_2()
model.load_weights(model_checkpoint_load_path)

#get random sample image to segment
# sample_index = random.randrange(0, len(data_files) - 1)
sample_index = 1
print("sample_index = ", sample_index)
sample_path = data_files[sample_index]
sample_truth_path = label_files[sample_index]

seg_save_path = '/proj/NIRAL/users/nic98/models/unet_2/' + os.path.basename(sample_path)[:-4] + "_new_seg" + ".nii.gz"

sample = np.load(sample_path)

sample_truth = np.load(sample_truth_path)
sample_truth = np.argmax(sample_truth, axis=3)

li = []
li.append(sample)

sample = tf.convert_to_tensor(li, dtype=tf.float32)
#run model
prob_map = model(sample)
        
sample_seg = tf.argmax(prob_map, 4)
sample_seg = np.asarray(sample_seg)[0]

label_save_path = "/proj/NIRAL/users/nic98/models/unet_2/" + os.path.basename(sample_truth_path)[:-4] + ".nii.gz"

seg_img = sitk.GetImageFromArray(sample_seg)
label_img = sitk.GetImageFromArray(sample_truth)
s_filter = sitk.CastImageFilter()
s_filter.SetOutputPixelType(sitk.sitkUInt16)
seg_img = s_filter.Execute(seg_img)
label_img = s_filter.Execute(label_img)

writer = sitk.ImageFileWriter()
writer.SetFileName(seg_save_path)
writer.Execute(seg_img)
writer.SetFileName(label_save_path)
writer.Execute(label_img)
