import numpy as np
from glob import glob

boundary_files = sorted(glob("/proj/NIRAL/users/nic98/DATA/pretraining/train_on/boundary_arrays/*"))
boundary_proportions = []
i = 0
for f in boundary_files:
    print(i)
    i += 1
    boundary_arr = np.load(f)
    volume = np.product(np.shape(boundary_arr))
    nonzero = np.count_nonzero(boundary_arr)
    print(nonzero / volume)
    boundary_proportions.append(nonzero / volume)

print(np.mean(boundary_proportions))
