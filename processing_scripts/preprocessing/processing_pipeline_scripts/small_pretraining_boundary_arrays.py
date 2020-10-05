import os
import numpy as np
from glob import glob

label_dir = '/proj/NIRAL/users/nic98/DATA/pretraining/small_train_on/label_arrays/'
save_dir = '/proj/NIRAL/users/nic98/DATA/pretraining/small_train_on/boundary_arrays/'
label_files = sorted(glob(label_dir + "*"))

label_arrays = []
boundary_files = []

for i in range(len(label_files)):
    label_file = label_files[i]
    boundary_file = label_file[:-4] + "_boundary" + label_file[-4:]
    boundary_file = os.path.basename(boundary_file)
    label_arrays.append(np.load(label_file))
    boundary_files.append(boundary_file)

# now do boundary detection in label arrays
i = 0
for i in range(len(label_arrays)):
    print("Doing boundary detection:", i / len(label_arrays))

    label_array = label_arrays[i]
    label_array = np.argmax(label_array, axis = 3)
    boundary_array = np.zeros(np.shape(label_array))
    print(np.shape(label_array), np.shape(boundary_array))

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

    np.save(save_dir + boundary_files[i], boundary_array.astype(np.float32))


