import nibabel as nib
import matplotlib.pyplot as plt
import os
import numpy as np
from glob import glob
from PIL import Image

base_dir = '/proj/NIRAL/users/nic98/DATA/pretraining/han_pretraining_symmetric'
input_files = glob(base_dir + '/*.nii.gz')
label_files = glob(base_dir + '/*label.nii.gz')
print('number of files matched: ' + str(len(input_files)))
# print('first five files')
# print(input_files[0:5])

def load_nifti(nifti_path):
    img = nib.nifti1.load(nifti_path)
    data = img.dataobj
    affine = np.asarray(img.affine)
    data = np.asarray(data)
    return (data, affine)

def save_nifti(img_data, affine, save_path):
    new_image = nib.Nifti1Image(img_data,affine)
    nib.nifti1.save(new_image, save_path)

data_arrays = []
label_arrays = []

i = 0
for file in input_files:
  (data, affine) = load_nifti(file)
  print(np.shape(data)," " + os.path.basename(file))
  data_arrays.append(data)
  i = i+1
  if (i > 50):
      break

i = 0
for file in label_files:
    (data, affine) = load_nifti(file)
    print(np.unique(np.reshape(data,-1)))
    label_arrays.append(data)
    i = i+1
    if (i > 10):
        break

slices = []
for i in range(len(data_arrays)):
    dims = np.shape(data_arrays[i])
    slices.append(data_arrays[i][dims[1]//2,:,:])

num_rows = 4
num_cols = 5
num_images = num_rows*num_cols
plt.figure(figsize=(3.5*num_cols, 3*num_rows))
for i in range(num_images):
    plt.subplot(num_rows, num_cols, i+1)
    plt.grid(False)
    plt.xticks([])
    plt.yticks([])
    plt.imshow(slices[i], cmap=plt.cm.binary)
    plt.xlabel(os.path.basename(input_files[i]))

plt.show()


# All affines are np.eye(4)

# print("data affines shape", np.shape(affines_arrays))
# print(data_affines)

print("data arrays has shape: " , np.shape(data_arrays))
# print(data_arrays[0])
# print("looking at " + input_files[0])
# 

# slice1 = data_arrays[0,100,:,:]
# slice2 = data_arrays[0,:,100,:]
# slice3 = data_arrays[0,:,:,100]
# img1 = Image.fromarray(slice1)
# img2 = Image.fromarray(slice2)
# img3 = Image.fromarray(slice3)

# img1.show()
 
# max_elt = np.max(data_arrays)
# min_elt = np.min(data_arrays)
# mean_elt = np.mean(data_arrays)
# 
# print('max elt:', max_elt)
# print('min elt:', min_elt)
# print('mean elt:', mean_elt)
# 
# print(data_arrays[0].shape)
# print("labels: ", np.unique(data_arrays[0], return_counts=True))
# print(data_arrays)
