import nibabel as nib
import matplotlib.pyplot as plt
import os
import numpy as np
from glob import glob
from PIL import Image

base_dir = '/proj/NIRAL/users/nic98/DATA/pretraining/original_data_nifti'
input_files = glob(base_dir + '/*.nii.gz')
input_files = sorted(input_files)
print('number of files matched: ' + str(len(input_files)))
# print('first five files')
# print(input_files[0:5])

input_files.pop(96)
input_files.pop(198)
input_files.pop(752)
input_files.pop(991)
input_files.pop(1195)
input_files.pop(1315)
input_files.pop(1348)
input_files.pop(1646)
input_files.pop(1736)
input_files.pop(1868)
input_files.pop(1991)
input_files.pop(2288)

input_files.pop(1576)

input_files.pop(474)

input_files.pop(705)

input_files.pop(816)

input_files.pop(867)

print("num files after popping bad ones: ", len(input_files))

j=0
while (j < len(input_files)):
    comp1 = os.path.basename(input_files[j])[12:22]
    comp2 = os.path.basename(input_files[j+1])[12:22]
    comp3 = os.path.basename(input_files[j+2])[12:22]

    if ((comp1 == comp2) and (comp1 == comp3)):
        j += 3
        continue
    else:
        print("problem index ", j)
        break

print("all good")
