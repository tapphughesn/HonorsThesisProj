from glob import glob
import nibabel as nib
import os
import sys
import re

base_dir = '/proj/NIRAL/users/nic98/DATA/training/original_data_nrrd'

base_data = sorted(glob(base_dir + '/*.nrrd'))
label_pattern = re.compile('.*label.*')
t1w_pattern = re.compile('.*t1w.*')
t2w_pattern = re.compile('.*t2w.*')

for file_name in base_data:
    num = re.findall('\d+', file_name)
    num = num[1]
    if label_pattern.match(file_name):
        os.system("mv " + file_name + " " + base_dir + "/" +  num + "_label.nrrd")
    if t1w_pattern.match(file_name):
        os.system("mv " + file_name + " " + base_dir + "/" +  num + "_t1w.nrrd")
    if t2w_pattern.match(file_name):
        os.system("mv " + file_name + " " + base_dir + "/" +  num + "_t2w.nrrd")



