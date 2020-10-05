import nibabel as nib
import matplotlib.pyplot as plt
import os
import numpy as np
import re
from glob import glob
from PIL import Image
import nrrd

base_dir = '/proj/NIRAL/users/nic98/DATA/pretraining/original_data_nrrd'
temp_dir = '/proj/NIRAL/users/nic98/DATA/pretraining/temp_data_nrrd'
save_dir = '/proj/NIRAL/users/nic98/DATA/pretraining/processed_data_nifti'

## These are the template files needed for Hist-Matching
t1template = "/proj/NIRAL/users/nic98/DATA/matchhistogram_templates/1year-Average-IBIS-MNI-t1w-stripped.nrrd"
t2template = "/proj/NIRAL/users/nic98/DATA/matchhistogram_templates/1year-Average-IBIS-MNI-t2w-stripped.nrrd"

original_files = glob(base_dir + '/*.nrrd')
original_files = sorted(original_files)

print("num original files: ", len(original_files))

## We will need these functions later

print("defining load/save funcs")

def load_nifti(nifti_path):
    img = nib.nifti1.load(nifti_path)
    data = img.dataobj
    affine = np.asarray(img.affine)
    data = np.asarray(data)
    return (data, affine)

def save_nifti(img_data, affine, save_path):
    new_image = nib.Nifti1Image(img_data,affine)
    nib.nifti1.save(new_image, save_path)

## Start by doing histo matching and storing .nrrd files in temporary directory

print("doing histo matching")

j=0
for file in original_files:
  print(j)
  label_pattern = re.compile(".*label.nrrd$")
  t1_pattern = re.compile(".*t1w.*")
  if label_pattern.match(file):
    os.system('cp ' + file + " " + temp_dir)
    j+=1
    continue
  if t1_pattern.match(file):
    os.system('/proj/NIRAL/tools/ImageMath ' + file + ' -matchHistogram ' + t1template + ' -outfile ' + temp_dir + '/' + os.path.basename(os.path.normpath(file)))
    j+=1
    continue
  else:
    os.system('ImageMath ' + file + ' -matchHistogram ' + t2template + ' -outfile ' + temp_dir + '/' + os.path.basename(os.path.normpath(file)))
    j+=1
  
## Now load these nrrd files as numpy arrays

histo_files = glob(temp_dir + '/*.nrrd')
histo_files = sorted(histo_files)

print("num of histomatched files: ", len(histo_files))

data_arrays = []

j = 0
for file in histo_files:
  print("loading histo files: ", 100*j/len(histo_files), "%")
  nrrd_file = nrrd.read(file)
  data = nrrd_file[0]
  header = nrrd_file[1]
  data = np.asarray(data)
  data_arrays.append(data)
  j+=1

## Now do cropping, normalizing, flipping, relabelling

print("doing cropping and normalizing")

## crop to (96,112,96)
for i in range(len(data_arrays)):
    (dim1, dim2, dim3) = np.shape(data_arrays[i])
    (buf1, buf2, buf3) = tuple(np.subtract(np.shape(data_arrays[i]), (96,112,96))//2)
    data_arrays[i] = data_arrays[i][(buf1+1):(dim1-buf1),(buf2+1):(dim2-buf2),(buf3+1):(dim3-buf3)]
    if (np.shape(data_arrays[i]) != (96,112,96)):
        print("Shape Error")
        print(np.shape(ar))
        raise Exception("Shape Error")
    ## normalize
    ## do not normalize label files
    if (len(np.unique(data_arrays[i])) < 15):
        continue
    else:
        ## k is the number of elements in ar * 0.99
        k = int(np.floor(np.product(np.shape(data_arrays[i]))*0.99))
        almost_max = np.partition(data_arrays[i], k, axis=None)[k]
        data_arrays[i] = data_arrays[i]/almost_max

print("arrays now cropped")
j = 0
while (j < 50):
    print(np.shape(data_arrays[j]))
    j+=1

## now do relabelling, flipping

print("doing relabelling")

half2_data_arrays = []
for l in range(len(data_arrays)):
    print("-------Array Number:", l, "----------")
    ar = data_arrays[l]
    ## if ar is a label, take advantage of symmetry and make 
    ## the labelling simpler
    ## 0 -> 0 (background)
    ## 1, 2 -> 1 (Amygdala)
    ## 3, 4 -> 2 (Caudate)
    ## 5, 6 -> 3 (Hippocampus)
    ## 7, 8 -> 4 (Globus Pallidus)
    ## 9, 10 -> 5 (Putamen)
    ## 40, 41 -> 6 (Thalamus)

    if len(np.unique(ar)) < 15:
        for i in range(len(ar)):
            for j in range(len(ar[i])):
                for k in range(len(ar[i,j])):
                    n = ar[i,j,k]
                    if (n == 0):
                        continue
                    if ((n == 1) or (n==2)):
                        ar[i,j,k] = 1
                    if ((n == 3) or (n==4)):
                        ar[i,j,k] = 2
                    if ((n == 5) or (n==6)):
                        ar[i,j,k] = 3
                    if ((n == 7) or (n==8)):
                        ar[i,j,k] = 4
                    if ((n == 9) or (n==10)):
                        ar[i,j,k] = 5
                    if ((n == 40) or (n==41)):
                        ar[i,j,k] = 6

    ## double the dataset by flipping
    print("doing flipping")

    half1 = ar[:48,:,:]
    half2 = ar[48:,:,:]
    ar = np.concatenate((half1, np.flip(half1,axis=0)),axis=0)
    data_arrays[l] = ar
    half2_data_arrays.append(np.concatenate((np.flip(half2,axis=0), half2), axis=0))

half1_data_arrays = data_arrays
del data_arrays

## save the processed data as nifti files
## first lets get the file names right
## the data should be in the same order it started in,
## so file name order should be preserved

print("saving processed files")

half1_filenames = []
half2_filenames = []

for file in histo_files:
    half1_file = os.path.basename(file)[:-5]+'_half1.nii.gz'
    half2_file = os.path.basename(file)[:-5]+'_half2.nii.gz'
    half1_filenames.append(half1_file)
    half2_filenames.append(half2_file)

for i in range(len(half1_data_arrays)):
    print(i)
    save_path_half1 = save_dir + '/' + half1_filenames[i]
    save_nifti(half1_data_arrays[i], np.eye(4), save_path_half1)

for i in range(len(half2_data_arrays)):
    print(i)
    save_path_half2 = save_dir + '/' + half2_filenames[i]
    save_nifti(half2_data_arrays[i], np.eye(4), save_path_half2)

## finally, remove all the files in the temporary directory
print("deleting temporary data")
os.system("rm -rf /proj/NIRAL/users/nic98/DATA/pretraining/temp_data_nrrd/*")
