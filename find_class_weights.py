import os
import nibabel as nib
import numpy as np
from glob import glob
import pickle

data_dir = '/proj/NIRAL/users/nic98/DATA/pretraining/processed_data_nifti'
data_files = glob(data_dir + '/*.nii.gz')

data_files = sorted(data_files)

print("files matched: ", len(data_files))

def load_nifti(nifti_path):
    img = nib.load(nifti_path)
    data = img.dataobj
    affine = img.affine
    data = np.asarray(data)
    return data, img, affine

## We want to assign weights to classes according the rule weight = 1/volume occupied
## such that large labels (like background) have small weights
## We load the label files to find the total volume of each class
## Then we compute the weights
## nums will store the volumes of each of the 7 classes

nums = np.zeros(7)
i = 0
print("loading files and counting")
while (i < len(data_files)):
    ## load files
    half1_label = load_nifti(data_files[i+2])[0]
    half2_label = load_nifti(data_files[i+3])[0]
    
    ## count number of instances of each class label
    for j in range(len(half1_label)):
        for k in range(len(half1_label)):
            for l in range(len(half1_label)):
                nums[half1_label[j,k,l]] += 1
                nums[half2_label[j,k,l]] += 1
    print("completion: ", np.round((i/len(data_files)*100), 2), "%")
    i += 6

total_volume = np.sum(nums)
weight = np.zeros(7)
weight[0] = total_volume/nums[0]
weight[1] = total_volume/nums[1]
weight[2] = total_volume/nums[2]
weight[3] = total_volume/nums[3]
weight[4] = total_volume/nums[4]
weight[5] = total_volume/nums[5]
weight[6] = total_volume/nums[6]

weights_dict = {0: weight[0], 1: weight[1], 2: weight[2], 3: weight[3], 4: weight[4], 5: weight[5], 6: weight[6]}
save_dict_dir = "/proj/NIRAL/users/nic98/DATA/class_training_weights/"
filename = "pretraining_class_weights"
with open(save_dict_dir + filename + ".pkl", 'wb') as f:
    pickle.dump(weights_dict, f, pickle.HIGHEST_PROTOCOL)

print("weights:")
print(weights_dict)
