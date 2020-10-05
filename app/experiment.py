import numpy as np
from glob import glob
import SimpleITK as sitk

nrrd_images = sorted(glob("/proj/NIRAL/users/nic98/DATA/pretraining/original_data_nrrd/*"))
nrrd_image_path = nrrd_images[10]

reader = sitk.ImageFileReader()
reader.SetFileName(nrrd_image_path)
image = reader.Execute()
nda = sitk.GetArrayFromImage(image)

array = np.asarray(nda)
print(np.shape(array))

