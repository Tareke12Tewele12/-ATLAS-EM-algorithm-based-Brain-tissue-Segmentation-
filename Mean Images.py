#!/usr/bin/env python
# coding: utf-8

## Obtain Mean Image From the Registered Images 

from os import listdir
from os.path import isdir, join
import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')

# Normalize array from 0 to 1
def normalize(target_array):
    target_array -= target_array.min()
    normalized_array = np.divide(target_array, target_array.max())
    return normalized_array

# Set path to registered images folder and import nii into python
data_path = "./registration-results/output/"
dirs = [f for f in listdir(data_path) if isdir(join(data_path, f))]

mean_im = None

for dir in dirs:
    im = join(data_path, dir,'result.1.nii.gz')
    nii_img = nib.load(im)
    nii_data = nii_img.get_data()
    nii_data = normalize(nii_data)
    
    
    if mean_im is None:
        mean_im = np.zeros_like(nii_data)
    
    mean_im += nii_data

mean_im = np.divide(mean_im, len(dirs))

#plt.imshow(mean_im[:,:,150], cmap = "gray")
#plt.show()
segment_nii = nib.Nifti1Image(mean_im, nii_img.affine, nii_img.header)
nib.save(segment_nii, 'mean_im.nii.gz')


### Normalize MNI template image and save to output path 
mni_path = "./MNITemplateAtlas-old/template.nii.gz"
min_out_path = "./MNITemplateAtlas/template.nii.gz"

built_template_path = "./built_atlas/mean_im.nii"
built_template = nib.load(built_template_path)
mni_template = nib.load(mni_path)
mni_template_data = normalize(mni_template.get_data())

segment_nii = nib.Nifti1Image(mni_template_data, built_template.affine, built_template.header)
nib.save(segment_nii, min_out_path)



