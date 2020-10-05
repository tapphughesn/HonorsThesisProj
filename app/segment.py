import tensorflow as tf
import os
import nibabel as nib
import SimpleITK as sitk
import pickle
import sys
import nrrd
import argparse
import numpy as np
from datetime import datetime
from model import *
from glob import glob

# Helper Functions


# Main Function
def main(main_args):
    
    t1_image = main_args.t1
    t2_image = main_args.t2
    seg_save_path = main_args.savepath
    prob_map_save_path = seg_save_path + "_prob_map"

    if ((t1_image == "") and (t2_image == "")):
        sys.exit("T1 and T2 images were not provided")
    else if (t1_image == ""):
        sys.exit("T1_image was not provided")
    else if (t2_image == ""):
        sys.exit("T2_image was not provided")

    #use SimpleITK to read images

    reader = sitk.ImageFileReader()
    reader.SetFileName(t1_image)
    t1_image = reader.Execute()
    reader.SetFileName(t2_image)
    t2_image = reader.Execute()

    t1_array = sitk.GetArrayFromImage(t1_image)
    t2_array = sitk.GetArrayFromImage(t2_image)

    #convert t1 and t2 images to tensorflow tensors and concatenate them along last axis

    t1_image = tf.convert_to_tensor(t1_image, dtype = tf.float32)
    t2_image = tf.convert_to_tensor(t2_image, dtype = tf.float32)
    input_image = tf.concat(t1_image, t2_image, axis=3)
    tf.expand_dims(input_image, 0)
    
    # Get model
    model = unet_2()
    model_checkpoint_load_path = "./checkpoints/parameters_" + "07-01-2020-21:48:57"
    model.load_weights(model_checkpoint_load_path)
    
    prob_map = np.asarray(model(input_image))[0]
    seg = np.argmax(input_image, axis=3)

    if main_args.probmap:
        ## Check if we can get the affine from the itk image
        prob_map_img = nib.Nifti1Image(prob_map, affine=np.eye(4))
        nib.save(prob_map_img, prob_map_save_path)

    seg_img = nib.Nifti1Image(seg, affine=np.eye(4))
    nib.save(seg_img, seg_save_path)


if (__name__ = __main__):
    parser = argparse.ArgumentParser(description="Segments brain image")
    parser.add_argument("--t1", help="Specify T1 modality image", default="")
    parser.add_argument("--t2", help="Specify T2 modality image", default="")
    parser.add_argument("--suffix", help="Suffix appended to segmentation file name", default="_seg")
    parser.add_argument("--savepath", help="Specify save path for output segmentation", default="")
    parser.add_argument("--probmap", action="store_true", help="Generate probability image instead of segmentation map", default = False)
    parser.add_argument("--nosave", action="store_true", help="Do not automatically save segmentation", default=False)
    main(parser.parse_args())
    
