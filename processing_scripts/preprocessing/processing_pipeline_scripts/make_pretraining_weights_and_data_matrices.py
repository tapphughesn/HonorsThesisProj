import nibabel as nib
import os
import pickle
import numpy as np
import tensorflow as tf
from glob import glob
from PIL import Image

print("Starting combining channels script")

print("Tensorflow version: ", tf.version)

print("-------Now Loading data files")

data_dir = '/proj/NIRAL/users/nic98/DATA/pretraining/processed_data_nifti'
save_dir = '/proj/NIRAL/users/nic98/DATA/pretraining/train_on'
data_files = glob(data_dir + '/*.nii.gz')

data_files = sorted(data_files)

def load_nifti(nifti_path):
    img = nib.load(nifti_path)
    data = img.dataobj
    affine = img.affine
    data = np.array(data)
    return data, img, affine

## Here we combine the t1w and t2w images as channels in the same image
## Also sort out labels and data
## And create and store weight matrices to be used later
## Boundary matrices are made in a different script

## Load weight dict

class_weights_dict = {}
with open("/proj/NIRAL/users/nic98/DATA/class_training_weights/pretraining_class_weights.pkl", "rb") as f:
    class_weights_dict = pickle.load(f)

i = 0
while (i < len(data_files)):
    print(i/len(data_files) * 100, "% done")

    #cut out "t1w" and "t2w" in file names
    data_name = os.path.basename(data_files[i])[0:23] + os.path.basename(data_files[i])[27:49]
    label_name = os.path.basename(data_files[i+2])[0:23] + os.path.basename(data_files[i+2])[27:41]
    weight_name = data_name + "_weight"

    half1_t1w = load_nifti(data_files[i])[0]
    half2_t1w = load_nifti(data_files[i+1])[0]
    half1_label = load_nifti(data_files[i+2])[0]
    half2_label = load_nifti(data_files[i+3])[0]
    half1_t2w = load_nifti(data_files[i+4])[0]
    half2_t2w = load_nifti(data_files[i+5])[0]

    half1_t1w = np.expand_dims(half1_t1w, axis=3)
    half2_t1w = np.expand_dims(half2_t1w, axis=3)
    half1_t2w = np.expand_dims(half1_t2w, axis=3)
    half2_t2w = np.expand_dims(half2_t2w, axis=3)
    
    half_1 = np.concatenate((half1_t1w, half1_t2w), axis=3)
    half_2 = np.concatenate((half2_t1w, half2_t2w), axis=3)

    ## Convert label arrays to one-hot encoding
    ## number of classes is 7

    pad = np.zeros((96,112,96,6))
    half1_label = np.expand_dims(half1_label, axis=3)
    half2_label = np.expand_dims(half2_label, axis=3)
    half1_label = np.concatenate((half1_label, pad), axis=3)
    half2_label = np.concatenate((half2_label, pad), axis=3)
    
    ## Construct weight matrices in the following loop as well
    ## as making the labels one-hot encoding

    half1_weight = np.zeros(np.shape(half1_label)[:-1])
    half2_weight = np.zeros(np.shape(half2_label)[:-1])

    for n in range(len(half1_label)):
        for j in range(len(half1_label[n])):
            for k in range(len(half1_label[n,j])):
                l1 = int(half1_label[n,j,k,0])
                l2 = int(half2_label[n,j,k,0])

                #do weight matrix before converting labels to one-hot
                half1_weight[n,j,k] = class_weights_dict[l1]
                half2_weight[n,j,k] = class_weights_dict[l2]

                ## convert to one-hot vectors for half1
                half1_label[n,j,k,0] = 0
                half1_label[n,j,k,l1] = 1
                ## same for half2
                half2_label[n,j,k,0] = 0
                half2_label[n,j,k,l2] = 1

    i += 6

    np.save(save_dir + "/data_arrays/" + data_name + "_half1", half_1.astype(np.float32))
    np.save(save_dir + "/data_arrays/" + data_name + "_half2", half_2.astype(np.float32))
    np.save(save_dir + "/label_arrays/" + label_name + "_half1", half1_label.astype(np.float32))
    np.save(save_dir + "/label_arrays/" + label_name + "_half2", half2_label.astype(np.float32))
    np.save(save_dir + "/weight_arrays/" + weight_name + "_half1", half1_weight.astype(np.float32))
    np.save(save_dir + "/weight_arrays/" + weight_name + "_half2", half2_weight.astype(np.float32))


