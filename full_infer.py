import tensorflow as tf
import nrrd
import nibabel as nib
import os
import SimpleITK as sitk
import sys
import numpy as np
from model import *
from glob import glob

model_checkpoint_load_path = "/proj/NIRAL/users/nic98/models/unet_4/checkpoints/parameters_" + "10-03-2020-12:17:30"
data_files = sorted(glob("/proj/NIRAL/users/nic98/DATA/training/validation_data_nrrd/*"))

histogram_templates = sorted(glob("/proj/NIRAL/users/nic98/DATA/matchhistogram_templates/*"))
t1_reference = histogram_templates[0]
t2_reference = histogram_templates[1]

# Get model
model = unet_4()
model.load_weights(model_checkpoint_load_path)
    
for index in range(5):
    label_file = data_files[3*index]
    t1_file = data_files[3*index+1]
    t2_file = data_files[3*index+2]
    
    # Create save directories for segmentations and their true labels
    
    seg_save_path_nifti = '/proj/NIRAL/users/nic98/models/unet_4/validation_segmentations/nifti_format/' + os.path.basename(t1_file)[:-5] + "_seg.nii.gz"
    label_save_path_nifti = '/proj/NIRAL/users/nic98/models/unet_4/validation_segmentations/nifti_format/' + os.path.basename(label_file)[:-5]  + ".nii.gz"
    
    seg_save_path_npy = '/proj/NIRAL/users/nic98/models/unet_4/validation_segmentations/npy_format/' + os.path.basename(t1_file)[:-5] + "_seg.npy"
    label_save_path_npy = '/proj/NIRAL/users/nic98/models/unet_4/validation_segmentations/npy_format/' + os.path.basename(label_file)[:-5]  + ".npy"
    
    cropped_seg_save_path = '/proj/NIRAL/users/nic98/models/unet_4/validation_segmentations/cropped_nifti/' + os.path.basename(t1_file)[:-5] + "_seg.nii.gz"
    cropped_label_save_path = '/proj/NIRAL/users/nic98/models/unet_4/validation_segmentations/cropped_nifti/' + os.path.basename(label_file)[:-5]  + ".nii.gz"
    
    # Use simpleITK to load t1, t2 images, label, and histogram-matching templates
    
    reader = sitk.ImageFileReader()
    reader.SetFileName(t1_file)
    t1_image = reader.Execute()
    reader.SetFileName(t2_file)
    t2_image = reader.Execute()
    
    reader.SetFileName(label_file)
    label_image = reader.Execute()
    
    reader.SetFileName(t1_reference)
    t1_reference_image = reader.Execute()
    reader.SetFileName(t2_reference)
    t2_reference_image = reader.Execute()
    
    # Use sitk filter to histogram match images to template
    
    matcher = sitk.HistogramMatchingImageFilter()
    matcher.ThresholdAtMeanIntensityOn()
    matcher.SetNumberOfHistogramLevels(1024)
    matcher.SetNumberOfMatchPoints(7)
    t1_image = matcher.Execute(t1_image, t1_reference_image)
    t2_image = matcher.Execute(t2_image, t2_reference_image)
    
    # Get image and label arrays 
    
    t1_array = sitk.GetArrayFromImage(t1_image)
    t2_array = sitk.GetArrayFromImage(t2_image)
    t1_array = np.asarray(t1_array)
    t2_array = np.asarray(t2_array)
    
    label_array = sitk.GetArrayFromImage(label_image)
    label_array = np.asarray(label_array)
    
    # Account for the differences between pynrrd loading and sitk loading... :(
    
    t1_array = np.swapaxes(t1_array,0,2)
    t2_array = np.swapaxes(t2_array,0,2)
    label_array = np.swapaxes(label_array,0,2)
    
    # process label according to symmetric labelling scheme
    
    ## 0 -> 0 (background)
    ## 1, 2 -> 1 (Amygdala)
    ## 3, 4 -> 2 (Caudate)
    ## 5, 6 -> 3 (Hippocampus)
    ## 7, 8 -> 4 (Globus Pallidus)
    ## 9, 10 -> 5 (Putamen)
    ## 40, 41 -> 6 (Thalamus)
    
    for i in range(len(label_array)):
        for j in range(len(label_array[i])):
            for k in range(len(label_array[i,j])):
                n = label_array[i,j,k]
                if (n == 0):
                    continue
                if ((n == 1) or (n==2)):
                    label_array[i,j,k] = 1
                    continue
                if ((n == 3) or (n==4)):
                    label_array[i,j,k] = 2
                    continue
                if ((n == 5) or (n==6)):
                    label_array[i,j,k] = 3
                    continue
                if ((n == 7) or (n==8)):
                    label_array[i,j,k] = 4
                    continue
                if ((n == 9) or (n==10)):
                    label_array[i,j,k] = 5
                    continue
                if ((n == 40) or (n==41)):
                    label_array[i,j,k] = 6
                    continue
                sys.exit("unexpected label encountered")
    
    # Crop t1 and t2 arrays
    
    (dim1, dim2, dim3) = np.shape(t1_array) 
    (buf1, buf2, buf3) = tuple(np.subtract(np.shape(t1_array), (96,112,96))//2) 
    t1_array = t1_array[(buf1+1):(dim1-buf1),(buf2+1):(dim2-buf2),(buf3+1):(dim3-buf3)]
    
    t2_array = t2_array[(buf1+1):(dim1-buf1),(buf2+1):(dim2-buf2),(buf3+1):(dim3-buf3)]
    
    cropped_label_array = label_array[(buf1+1):(dim1-buf1),(buf2+1):(dim2-buf2),(buf3+1):(dim3-buf3)]
    
    # 99 % Max Normalize t1, t2 arrays
    
    k = int(np.floor(np.product(np.shape(t1_array))*0.99))
    almost_max = np.partition(t1_array, k, axis=None)[k]
    t1_array = t1_array/almost_max
    
    k = int(np.floor(np.product(np.shape(t2_array))*0.99))
    almost_max = np.partition(t2_array, k, axis=None)[k]
    t2_array = t2_array/almost_max
    
    # Convert t1 and t2 arrays to tensorflow tensors and concatenate them along last axis
    
    t1_array = tf.convert_to_tensor(t1_array, dtype = tf.float32)
    t2_array = tf.convert_to_tensor(t2_array, dtype = tf.float32)
    t1_array = tf.expand_dims(t1_array, 3)
    t2_array = tf.expand_dims(t2_array, 3)
    input_image = tf.concat([t1_array, t2_array], axis=3)
    input_image = tf.expand_dims(input_image, 0)
    
    # Segment MR image
    
    prob_map_array = np.asarray(model(input_image))[0]
    seg_array = np.argmax(prob_map_array, axis=3)
    
    # "uncrop" segmentation by adding 0 padding
    
    cropped_seg_array = seg_array
    padding = np.zeros((dim1, dim2, dim3))
    padding[(buf1+1):(dim1-buf1),(buf2+1):(dim2-buf2),(buf3+1):(dim3-buf3)] = seg_array
    seg_array = padding
    
    # Save seg_array and label_array as .npy files
    
    np.save(seg_save_path_npy, seg_array)
    np.save(label_save_path_npy, label_array)
    
    # Cast segmentation and prob map to UInt16
    
    seg_img = sitk.GetImageFromArray(seg_array)
    label_img = sitk.GetImageFromArray(label_array)
    cropped_seg_img = sitk.GetImageFromArray(cropped_seg_array)
    cropped_label_img = sitk.GetImageFromArray(cropped_label_array)
    
    s_filter = sitk.CastImageFilter()
    s_filter.SetOutputPixelType(sitk.sitkUInt16)
    seg_img = s_filter.Execute(seg_img)
    label_img = s_filter.Execute(label_img)
    cropped_seg_img = s_filter.Execute(cropped_seg_img)
    cropped_label_img = s_filter.Execute(cropped_label_img)
    
    # Save segmentation and true label files in nifti format using sitk
    
    writer = sitk.ImageFileWriter()
    writer.SetFileName(seg_save_path_nifti)
    writer.Execute(seg_img)
    writer.SetFileName(label_save_path_nifti)
    writer.Execute(label_img)
    
    writer.SetFileName(cropped_seg_save_path)
    writer.Execute(cropped_seg_img)
    writer.SetFileName(cropped_label_save_path)
    writer.Execute(cropped_label_img)
    
    print("Successfully segmented validation image")




