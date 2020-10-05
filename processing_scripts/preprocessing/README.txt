This directory contains preprocessing scripts for investigating, converting, 
and processing the original data.
Data is accessed in /proj/NIRAL/users/nic98/DATA/
You will find data in two formats here: .nrrd and .nii.gz

All scripts beginning with "investigate" serve the function of visualizing the data
or extracting information from the data, which may be useful as a sanity check

Some scripts, such as nrrd_to_nifti.py, convert data to a different file format

process_data.py is the main script here, it processes the original data to a suitable
form for input into my neural network models.
