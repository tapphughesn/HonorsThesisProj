import nibabel as nib
import matplotlib.pyplot as plt
import SimpleITK as sitk
import os
import numpy as np
import re
from glob import glob
import nrrd

base_dir = '/proj/NIRAL/users/nic98/DATA/pretraining/original_data_nrrd'
save_dir = '/proj/NIRAL/users/nic98/processing_scripts/preprocessing/'
savename1 = 'label_pynrrd.nii.gz'
savename2 = 'label_sitk.nii.gz'

base_files = sorted(glob(base_dir + "/*"))
filename = base_files[1]

def save_nifti(img_data, affine, save_path):
    new_image = nib.Nifti1Image(img_data,affine)
    nib.nifti1.save(new_image, save_path)

nrrd_file = nrrd.read(filename)
data1 = nrrd_file[0]
header = nrrd_file[1]
data1 = np.asarray(data1)

print("data1 shape", np.shape(data1))

# save_nifti(data1, np.eye(4), save_dir + savename1)
img = sitk.GetImageFromArray(data1)
writer = sitk.ImageFileWriter()
writer.SetFileName(save_dir + savename1)
writer.Execute(img)

reader = sitk.ImageFileReader()
reader.SetFileName(filename)
img = reader.Execute()

data2 = sitk.GetArrayFromImage(img)

print("data2 shape", np.shape(data2))

img = sitk.GetImageFromArray(data2)

writer = sitk.ImageFileWriter()
writer.SetFileName(save_dir + savename2)
writer.Execute(img)


