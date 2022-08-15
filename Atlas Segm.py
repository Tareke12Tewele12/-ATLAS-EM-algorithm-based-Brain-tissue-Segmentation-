#!/usr/bin/env python
# coding: utf-8

# # Atlas Based Segmentation

from os import listdir
from os.path import isdir, join
import os
import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# Normalize array from 0 to 1
def normalize(target_array):
    target_array -= target_array.min()
    normalized_array = np.divide(target_array, target_array.max())
    return normalized_array

# Read nii from path and normalize array from 0 to 1
def read_im(image_path):
    nii_img = nib.load(image_path)
    nii_data = nii_img.get_data()
    nii_data = normalize(nii_data)
    return nii_data, nii_img

def calc_dice(segmented_images, groundtruth_images):
    segData = segmented_images + groundtruth_images
    TP_value = np.amax(segmented_images) + np.amax(groundtruth_images)
    TP = (segData == TP_value).sum() 
    segData_FP = 2. * segmented_images + groundtruth_images
    segData_FN = segmented_images + 2. * groundtruth_images
    FP = (segData_FP == 2 * np.amax(segmented_images)).sum() 
    FN = (segData_FN == 2 * np.amax(groundtruth_images)).sum() 
    dice = 2*TP/(2*TP+FP+FN)
    return dice  

def dice_similarity(segmented_img, groundtruth_img):

    #Extract dice for each label  1) CSF 2) WM 3) GM  
    CSF_label = 1
    WM_label = 2
    GM_label = 3
    
    
    segmented_data = segmented_img.get_data().copy()
    groundtruth_data = groundtruth_img.get_data().copy()
    seg_CSF = (segmented_data == CSF_label) * 1
    gt_CSF = (groundtruth_data == CSF_label) * 1
    seg_GM = (segmented_data == GM_label) * 1
    gt_GM = (groundtruth_data == GM_label) * 1
    seg_WM = (segmented_data == WM_label) * 1
    gt_WM = (groundtruth_data == WM_label) * 1
    
    dice_CSF = calc_dice(seg_CSF, gt_CSF)
    dice_GM = calc_dice(seg_GM, gt_GM)
    dice_WM = calc_dice(seg_WM, gt_WM)
    
    
    return dice_CSF, dice_GM, dice_WM

def show_slice(img, slice_no):
    """
        Inputs: img (nibabel): image name
                slice_no (np slice): np.s_[:, :, 30]
        """
    data = img.get_fdata()
    plt.figure()
    plt.imshow(data[slice_no].T, cmap='gray')
    plt.show()


# ## Get atlas path (Switch between the MNI or built atlas) 
# put either 'mni' or 'built' 

use_atlas = 'mni'

if use_atlas == 'built':
    # Path to template image
    mean_im = './built_atlas/mean_im.nii.gz'
    # Path to atlases 
    GM_probs = './built_atlas/GM_probs.nii.gz'
    WM_probs = './built_atlas/WM_probs.nii.gz'
    CSF_probs = './built_atlas/CSF_probs.nii.gz'
elif use_atlas == 'mni':
    # Path to template image
    mean_im = './MNITemplateAtlas/template.nii.gz'
    # Path to atlases 
    Atlas_GM_probs = './MNITemplateAtlas/Atlas_GM_probs.nii.gz'
    Atlas_WM_probs = './MNITemplateAtlas/Atlas_WM_probs.nii.gz'
    Atlas_CSF_probs = './MNITemplateAtlas/Atlas_CSF_probs.nii.gz'

Atlas_GM_atlas, _ = read_im(Atlas_GM_probs)
Atlas_WM_atlas, _ = read_im(Atlas_WM_probs)
Atlas_CSF_atlas, _ = read_im(Atlas_CSF_probs)

# plt.imshow(GM_atlas[:,:,100], cmap = "gray")
# plt.show()
# plt.imshow(WM_atlas[:,:,100], cmap = "gray")
# plt.show()
# plt.imshow(CSF_atlas[:,:,100], cmap = "gray")
# plt.show()


# # Segmentation by Atlas

if use_atlas == 'built':
    # Path to test image
    test_im = "./test-set/registration-results/our_templates/1003/result.1.nii"
elif use_atlas == 'mni':
    # Path to test image
    test_im = "./test-set/registration-results/mni_templates/1003/result.1.nii"

test_data, test_im = read_im(test_im)


# Key the labels 
CSF_label = 1
GM_label = 3
WM_label = 2


# Assign GM, WM, CSF to voxel with highest probability
GM = GM_label * np.nan_to_num((GM_atlas > CSF_atlas) * (GM_atlas > WM_atlas))
WM = WM_label * np.nan_to_num((WM_atlas > CSF_atlas) * (WM_atlas > GM_atlas))
CSF = CSF_label * np.nan_to_num((CSF_atlas > WM_atlas) * (CSF_atlas > GM_atlas))
seg_im = GM + WM + CSF


# plt.imshow(seg_im[:,:,100], cmap = "gray")
# plt.show()


# # Segmentation Using Atlas Only

if use_atlas == 'built':
    # Path to test image folder
    template_path = "./test-set/registration-results/built_templates/"
    atlas_path = "./test-set/registration-results/built_atlases/"
    result_path = "./test-set/segmentation-results/built-atlas-segmentation/"
elif use_atlas == 'mni':
    # Path to test image folder
    template_path = "./test-set/registration-results/mni_templates/"
    atlas_path = "./test-set/registration-results/mni_atlases/"
    result_path = "./test-set/segmentation-results/mni-atlas-segmentation/"

gt_path = "./test-set/testing-labels/"

onlydirs = [f for f in listdir(template_path) if isdir(join(template_path, f))]

all_dice = np.zeros((len(onlydirs),3))

for i, f in enumerate(onlydirs):
    _, template_img = read_im(join(template_path,f,'result.1.nii'))
    GM_atlas, _ = read_im(join(atlas_path, f,'gm','result.nii.gz'))
    WM_atlas, _ = read_im(join(atlas_path, f,'wm','result.nii.gz'))
    CSF_atlas, _ = read_im(join(atlas_path, f,'csf','result.nii.gz'))

    # Assign GM, WM, CSF to voxel with highest probability
    GM = GM_label * np.nan_to_num((GM_atlas > CSF_atlas) * (GM_atlas > WM_atlas))
    WM = WM_label * np.nan_to_num((WM_atlas > CSF_atlas) * (WM_atlas > GM_atlas))
    CSF = CSF_label * np.nan_to_num((CSF_atlas > WM_atlas) * (CSF_atlas > GM_atlas))
    seg_im = GM + WM + CSF
    segmented_img = nib.Nifti1Image(seg_im, template_img.affine, template_img.header)

    # Calculate DICE
    path_gt = join(gt_path,f+"_3C.nii.gz")
    _, groundtruth_img = read_im(path_gt)
    all_dice[i,0], all_dice[i,1], all_dice[i,2] = dice_similarity(segmented_img, groundtruth_img)

    # Make directory to save result seg
    new_dir = join(result_path,f)
    #os.mkdir(new_dir)
    #nib.save(segmented_img, join(new_dir,'atlas_seg.nii.gz'))

print(all_dice)

# Write DICE values to file
if use_atlas == 'built':
    out_dice_path = './DSC_result_All/built_atlas_dice.csv'
elif use_atlas == 'mni':
    out_dice_path = './DSC_result_All/mni_atlas_dice.csv'
    
with open(out_dice_path, 'w+') as out_f:
    out_f.write('img,csf,gm,wm,\n')
    for index, row in enumerate(all_dice): 
        out_f.write(onlydirs[index] + ',' + ','.join(str(j) for j in row) + ',\n')

