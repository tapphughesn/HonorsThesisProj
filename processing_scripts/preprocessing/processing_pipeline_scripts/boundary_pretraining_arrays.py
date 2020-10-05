import nibabel as nib
import os
import numpy as np
from glob import glob

def load_nifti(nifti_path):
    img = nib.load(nifti_path)
    data = img.dataobj
    affine = img.affine
    data = np.array(data)
    return data, img, affine

data_dir = '/proj/NIRAL/users/nic98/DATA/pretraining/processed_data_nifti'
save_dir = '/proj/NIRAL/users/nic98/DATA/pretraining/train_on/'
data_files = sorted(glob(data_dir + '/*.nii.gz'))

label_arrays = []
boundary_names = []

#grab label files to analyze and file names for saving later
i = 0
while (i < len(data_files)):
    print("getting label files:", i / len(data_files))
    data_name = os.path.basename(data_files[i])[0:23] + os.path.basename(data_files[i])[27:49]
    boundary_name = data_name + "_boundary"

    half1_label = load_nifti(data_files[i+2])[0]
    half2_label = load_nifti(data_files[i+3])[0]

    label_arrays.append(half1_label)
    label_arrays.append(half2_label)

    boundary_names.append(boundary_name + "_half1")
    boundary_names.append(boundary_name + "_half2")

    i += 6

label_arrays = np.asarray(label_arrays)

# now do boundary detection in label arrays
i = 0
for i in range(len(label_arrays)):
    print("Doing boundary detection:", i / len(label_arrays))

    label_array = label_arrays[i]
    boundary_array = np.zeros(np.shape(label_array))

    for j in range(len(label_array)):
        for k in range(len(label_array[j])):
            for l in range(len(label_array[j,k])):

                # set boundary voxels to 1.0, keep non-boundary voxels at 0
                # no need to check the voxels on the outside boundary,

                if (j == 0) or (j == np.shape(label_array)[0] - 1):
                    continue

                if (k == 0) or (k == np.shape(label_array)[1] - 1):
                    continue

                if (l == 0) or (l == np.shape(label_array)[2] - 1):
                    continue

                # skip this voxel if it is already determined to be boundary
                if (boundary_array[j,k,l] == 1.0):
                    continue

                # Now need to check all 26 adjacent voxels, to determine if this is a boundary point.
                # (a,b,c) will be the offset of the indices j,k,l to access a adjacent point
                for n in range(1,27):
                    a = (n // 9) - 1
                    b = (n % 9 // 3) - 1
                    c = (n % 9 % 3) - 1

                    # Check if adjacent point has a different label than this point
                    if (label_array[j,k,l] != label_array[j+a,k+b,l+c]):
                        # boundary point detected!
                        boundary_array[j,k,l] = 1.0
                        boundary_array[j+a,k+b,l+c] = 1.0

    np.save(save_dir + "/boundary_arrays/" + boundary_names[i], boundary_array.astype(np.float32))


