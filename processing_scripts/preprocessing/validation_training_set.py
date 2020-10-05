from glob import glob
import os

# Move some of the training data to a validation set
# We will use 5 of the 32 data sets for validation

base_dir = '/proj/NIRAL/users/nic98/DATA/training/original_data_nrrd'
val_dir = '/proj/NIRAL/users/nic98/DATA/training/validation_data_nrrd'

base_data = sorted(glob(base_dir + '/*.nrrd'))
j = 0
for i in range(len(base_dir)//3):
    if j >= 5 :
        break

    os.system("mv " + base_data[3*i] + " " + val_dir)
    os.system("mv " + base_data[3*i+1] + " " + val_dir)
    os.system("mv " + base_data[3*i+2] + " " + val_dir)
    j = j+1


base_data = sorted(glob(base_dir + '/*.nrrd'))
val_data = sorted(glob(val_dir + '/*.nrrd'))

print(len(base_data))
print(len(val_data))
