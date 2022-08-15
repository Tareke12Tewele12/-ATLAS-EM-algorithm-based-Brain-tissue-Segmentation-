#!/usr/bin/env python
# coding: utf-8

# # EM + Atlas Segmentation

from os import listdir
from os.path import isdir, join
import os
import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
from sklearn.cluster import KMeans
from numpy.linalg import inv, det, norm
from math import sqrt, pi
from functools import partial


# numeric labels for brain tissue types
CSF_label = 1
GM_label = 3
WM_label = 2

def normalize(target_array): # Normalize array values from 0 to 1

    target_array -= target_array.min()
    normalized_array = np.divide(target_array, target_array.max())
    return normalized_array


def show_slice(img, slice_no): # image slicing 
    data = img.get_fdata()
    plt.figure()
    plt.imshow(data[slice_no].T, cmap='gray')
    plt.show()
    

def show_slice_data(data, slice_no): ## displaying slice of a given array

    plt.imshow(data[slice_no], cmap = "gray")
    plt.show()

    
def read_im(image_path): ## Read nii file from Image_path(str)
    
    nii_img = nib.load(image_path)
    nii_data = nii_img.get_data()
    
    return nii_data, nii_img


def calc_dice(segmented_images, groundtruth_images): # calculate dice similarity Cofficient between two volums
   
    segData = segmented_images + groundtruth_images
    TP_value = np.amax(segmented_images) + np.amax(groundtruth_images)
    
    # found a true positive: segmentation result and groundtruth match(both are positive)
    TP = (segData == TP_value).sum()
    segData_FP = 2. * segmented_images + groundtruth_images
    segData_FN = segmented_images + 2. * groundtruth_images
    
    # found a false positive: segmentation result and groundtruth mismatch
    FP = (segData_FP == 2 * np.amax(segmented_images)).sum() 
    
    # found a false negative: segmentation result and groundtruth mismatch
    FN = (segData_FN == 2 * np.amax(groundtruth_images)).sum() 
    
    return 2*TP/(2*TP+FP+FN)  # according to the definition of DICE similarity score


def dice_similarity(segmented_img, groundtruth_img): #Extract binary label images for regions
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


def apply_mask(target_data, gt_data): ##Create mask using groundtruth image and apply it.

    # Create mask: Select pixels higher than 0 in gt and set to 1
    gt_data[gt_data > 0] = 1
    
    # Apply mask
    target_data = np.multiply(target_data, gt_data)
    
    
    return target_data


def seg_data_to_nii(original_im, y_pred, features_nonzero_row_indicies):
    """
        Inputs: original_im (nibabel): original image nii file
                y_pred (np array): labels for all non-zero points
                features_nonzero_row_indicies (np array): indicies of non-zero points,
                                                          same length as y_pred
        Returns: segment_nii (nibabel): segmented labels nii file        
    """
    original_img_shape = original_im.get_data().shape
    original_img_len = original_img_shape[0] * original_img_shape[1] * original_img_shape[2]
    segment_im = np.zeros(original_img_len)
    labels = np.copy(y_pred) + 1
    segment_im[features_nonzero_row_indicies] = labels
    segment_im = np.reshape(segment_im, original_im.shape)
    segment_nii = nib.Nifti1Image(segment_im, original_im.affine, original_im.header)
    
    return segment_nii


def integrate_atlas_nii(original_im, y_pred, features_nonzero, features_nonzero_row_indicies, weights, csf_atlas, 
                           gm_atlas, wm_atlas):
    
    # Create image with all 3 classes and random labels
    y_pred = y_pred + 1
    original_img_shape = original_im.get_data().shape
    original_img_len = original_img_shape[0] * original_img_shape[1] * original_img_shape[2]
    
    segment_im = np.zeros(original_img_len)
    segment_im[features_nonzero_row_indicies] = y_pred
    segment_im = np.reshape(segment_im, original_im.shape)
    
    temp_class1_im = np.zeros_like(segment_im)
    temp_class2_im = np.zeros_like(segment_im)
    temp_class3_im = np.zeros_like(segment_im)
    
    #Assign class1 to 1
    temp_class1_im[segment_im == 1] = 1
    #Assign class2 to 2
    temp_class2_im[segment_im == 2] = 1
    #Assign class3 to 1
    temp_class3_im[segment_im == 3] = 1
    
    # Compute DICE between each class to determine which class it belongs to
    dice1 = [calc_dice(temp_class1_im, csf_atlas), calc_dice(temp_class2_im, csf_atlas), 
                                  calc_dice(temp_class3_im, csf_atlas)]
    dice2 = [calc_dice(temp_class1_im, wm_atlas), calc_dice(temp_class2_im, wm_atlas), 
                                  calc_dice(temp_class3_im, wm_atlas)]
    dice3 = [calc_dice(temp_class1_im, gm_atlas), calc_dice(temp_class2_im, gm_atlas), 
                                  calc_dice(temp_class3_im, gm_atlas)]
    csf_to_change = np.argmax(dice1) + 1
    wm_to_change = np.argmax(dice2) + 1
    gm_to_change = np.argmax(dice3) + 1
    
    
    #New y_pred
    y_pred_corrected_labels = np.zeros_like(y_pred)
    #Assign CSF to its correct label
    y_pred_corrected_labels[y_pred == csf_to_change] = CSF_label
    #Assign GM to its correct label
    y_pred_corrected_labels[y_pred == gm_to_change] = GM_label
    #Assign WM to its correct label
    y_pred_corrected_labels[y_pred == wm_to_change] = WM_label
    
    # Get weights back into original shape
    weight_csf_im = np.zeros(original_img_len)
    weight_gm_im = np.zeros(original_img_len)
    weight_wm_im = np.zeros(original_img_len)
    
    weight_csf_im[features_nonzero_row_indicies] = weights[:,csf_to_change-1]
    weight_gm_im[features_nonzero_row_indicies] = weights[:,gm_to_change-1]
    weight_wm_im[features_nonzero_row_indicies] = weights[:,wm_to_change-1]
    weight_csf_im = np.reshape(weight_csf_im, original_im.shape)
    weight_gm_im = np.reshape(weight_gm_im, original_im.shape)
    weight_wm_im = np.reshape(weight_wm_im, original_im.shape)
    
    # Multiply weights by each atlas
    csf_probs = weight_csf_im * csf_atlas
    gm_probs = weight_gm_im * gm_atlas
    wm_probs = weight_wm_im * wm_atlas
    
    # Assign GM, WM, CSF to voxel with highest probability
    GM = GM_label * np.nan_to_num((gm_probs > csf_probs) * (gm_probs > wm_probs))
    WM = WM_label * np.nan_to_num((wm_probs > csf_probs) * (wm_probs > gm_probs))
    CSF = CSF_label * np.nan_to_num((csf_probs > wm_probs) * (csf_probs > gm_probs))
    seg_im = GM + WM + CSF
    
    segment_im = np.zeros(original_img_len)
    segment_im = np.reshape(seg_im, original_im.shape)
    segment_nii = nib.Nifti1Image(segment_im, original_im.affine, original_im.header)

    return segment_nii


def seg_correct_labels_to_nii(original_im, y_pred, features_nonzero, features_nonzero_row_indicies, csf_atlas, 
                           gm_atlas, wm_atlas):
    # Create image with all 3 classes and random labels
    y_pred = y_pred + 1
    original_im_flat = original_im.get_data().copy().flatten()
    segment_im = np.zeros_like(original_im_flat)
    segment_im[features_nonzero_row_indicies] = y_pred
    segment_im = np.reshape(segment_im, original_im.shape)
    
    temp_class1_im = np.zeros_like(segment_im)
    temp_class2_im = np.zeros_like(segment_im)
    temp_class3_im = np.zeros_like(segment_im)
    
    #Assign class1 to 1
    temp_class1_im[segment_im == 1] = 1
    #Assign class2 to 2
    temp_class2_im[segment_im == 2] = 1
    #Assign class3 to 1
    temp_class3_im[segment_im == 3] = 1
    
    # Compute DICE between each class to determine which class it belongs to
    dice1 = [calc_dice(temp_class1_im, csf_atlas), calc_dice(temp_class2_im, csf_atlas), 
                                  calc_dice(temp_class3_im, csf_atlas)]
    dice2 = [calc_dice(temp_class1_im, wm_atlas), calc_dice(temp_class2_im, wm_atlas), 
                                  calc_dice(temp_class3_im, wm_atlas)]
    dice3 = [calc_dice(temp_class1_im, gm_atlas), calc_dice(temp_class2_im, gm_atlas), 
                                  calc_dice(temp_class3_im, gm_atlas)]
    csf_to_change = np.argmax(dice1) + 1
    wm_to_change = np.argmax(dice2) + 1
    gm_to_change = np.argmax(dice3) + 1
    
    #New y_pred
    y_pred_corrected_labels = np.zeros_like(y_pred)
    #Assign CSF to its correct label
    y_pred_corrected_labels[y_pred == csf_to_change] = CSF_label
    #Assign GM to its correct label
    y_pred_corrected_labels[y_pred == gm_to_change] = GM_label
    #Assign WM to its correct label
    y_pred_corrected_labels[y_pred == wm_to_change] = WM_label

    original_im_flat = original_im.get_data().copy().flatten()
    segment_im = np.zeros_like(original_im_flat)
    labels = np.copy(y_pred_corrected_labels)
    segment_im[features_nonzero_row_indicies] = labels
    segment_im = np.reshape(segment_im, original_im.shape)
    segment_nii = nib.Nifti1Image(segment_im, original_im.affine, original_im.header)

    return segment_nii


# #  Preparation of dataset

MAX_STEPS = 30
min_change = 0.01

def gaussian_mixture(features, mean, cov):
 
    return np.exp(-0.5*(features - mean) * (1/cov) * np.transpose(features - mean)) / (2 * pi * sqrt(cov))


def membership_weight(p0, p1, p2, a0, a1, a2):
  
    denominator = (p0 * a0) + (p1 * a1) + (p2 * a2)
    w0 = (p0 * a0) / denominator
    w1 = (p1 * a1) / denominator
    w2 = (p2 * a2) / denominator
    
    return np.stack((w0, w1, w2), axis=1)


def get_log_likelihood(class_dist, gauss_density): # calcuate  Loglikelihood with different parmaters 
    
    for index, alpha in enumerate(class_dist):
        if index == 0:
            total_sum = alpha * gauss_density[index]
        else:
            total_sum += alpha * gauss_density[index]
    
    return np.sum(np.log(total_sum))


##Save DICE values into a csv file.

def save_dice(out_dice_path, img_names, dice_values): 
    with open(out_dice_path, 'w+') as out_f:
        out_f.write('img,csf,gm,wm,\n')
        for index, row in enumerate(dice_values): 
            out_f.write(img_names[index] + ',' + ','.join(str(j) for j in row) + ',\n')


for use_atlas in ('built', 'mni'):   
        if use_atlas == 'built':
            # Path to test image
            test_img_path = "./test-set/testing-images/"
            atlas_path = "./test-set/registration-results/built_atlases/"
            result_path = f"./test-set/segmentation-results/EM-atlas-segmentation_kmeans-init/"
            gt_path = "./test-set/testing-labels/"
            mask_path = "./test-set/testing-mask/"

            out_dice_path = f'./dice-results/EM_atlas_kmeans_dice.csv'

        elif use_atlas == 'mni':
            # Path to test image
            test_img_path = "./test-set/testing-images/"
            atlas_path = "./test-set/registration-results/mni_atlases/"
            result_path = f"./test-set/segmentation-results/mni-EM-atlas-segmentation_kmeans-init/"
            gt_path = "./test-set/testing-labels/"
            mask_path = "./test-set/testing-mask/"

            out_dice_path = f'./dice-results/mni_EM_atlas_kmeans_dice.csv'

        onlydirs = [f[:-7] for f in listdir(test_img_path)]

        all_dice = np.zeros((len(onlydirs),3))

        for i, f in enumerate(onlydirs):
            # Load all data for EM algorithm
            test_data, test_img = read_im(join(test_img_path,f+'.nii.gz'))
            test_data = normalize(test_data)

            GM_atlas, _ = read_im(join(atlas_path, f,'gm','result.nii.gz'))
            WM_atlas, _ = read_im(join(atlas_path, f,'wm','result.nii.gz'))
            CSF_atlas, _ = read_im(join(atlas_path, f,'csf','result.nii.gz'))
            _, groundtruth_img = read_im(join(gt_path,f+"_3C.nii.gz"))
            mask_data, _ = read_im(join(mask_path,f+"_1C.nii.gz"))

            # Apply mask from GT image
            test_masked = apply_mask(test_data, mask_data)

            # Pre-process feature vector to remove background points from algorithm
            # and save those indicies to add back
            features = test_masked.copy().flatten()
            features = np.transpose(features)   
            features_nonzero_row_indicies = np.nonzero(features)
            features_nonzero = features[features_nonzero_row_indicies]

            
            features_nonzero_reshaped = features_nonzero.reshape(-1, 1)

            kmeans = KMeans(n_clusters=3, random_state=0, init='k-means++')\
            .fit(features_nonzero_reshaped)
            y_pred = kmeans.predict(features_nonzero_reshaped)
            centroids = kmeans.cluster_centers_

            # intialize EM algorithm
            class0 = features_nonzero[np.argwhere(y_pred == 0)[:,0]]
            class1 = features_nonzero[np.argwhere(y_pred == 1)[:,0]]
            class2 = features_nonzero[np.argwhere(y_pred == 2)[:,0]]

            # Compute mean and variance of each class
            mean0 = np.mean(class0, axis = 0)
            mean1 = np.mean(class1, axis = 0)
            mean2 = np.mean(class2, axis = 0)
            cov0 = np.cov(class0, rowvar = False)
            cov1 = np.cov(class1, rowvar = False)
            cov2 = np.cov(class2, rowvar = False)

            # Class distribution
            a0 = class0.shape[0] / features_nonzero.shape[0]
            a1 = class1.shape[0] / features_nonzero.shape[0]
            a2 = class2.shape[0] / features_nonzero.shape[0]

            # Compute Gaussian mixture model for each point
            p0 = gaussian_mixture(features_nonzero,  mean = mean0, cov = cov0)
            p1 = gaussian_mixture(features_nonzero,  mean = mean1, cov = cov1)
            p2 = gaussian_mixture(features_nonzero,  mean = mean2, cov = cov2)

            # # Compute membership weight for each point
            weights = membership_weight(p0, p1, p2, a0, a1, a2)
            # get initial log-likelihood
            log_likelihood = get_log_likelihood((a0, a1, a2), (p0, p1, p2))

            n_steps = 0

            while True:
                # Maximization step: Use that classification to reestimate the parameters
                # Class distribution
                counts = np.sum(weights, axis=0)

                a0 = counts[0] / len(features_nonzero)
                a1 = counts[1] / len(features_nonzero)
                a2 = counts[2] / len(features_nonzero)

                # Calculate mean and covariance for new classes
                mean0 = (1/counts[0]) * (weights[:, 0] @ features_nonzero)
                mean1 = (1/counts[1]) * (weights[:, 1] @ features_nonzero)
                mean2 = (1/counts[2]) * (weights[:, 2] @ features_nonzero)
                cov0 = (1/counts[0]) * ((weights[:, 0] * (features_nonzero - mean0)) @ (features_nonzero - mean0))
                cov1 = (1/counts[1]) * ((weights[:, 1] * (features_nonzero - mean1)) @ (features_nonzero - mean1))
                cov2 = (1/counts[2]) * ((weights[:, 2] * (features_nonzero - mean2)) @ (features_nonzero - mean2))

                p0 = gaussian_mixture(features_nonzero,  mean = mean0, cov = cov0)
                p1 = gaussian_mixture(features_nonzero,  mean = mean1, cov = cov1)
                p2 = gaussian_mixture(features_nonzero,  mean = mean2, cov = cov2)

                # Compute membership weight for each point
                weights = membership_weight(p0, p1, p2, a0, a1, a2)

                log_likelihood_new = get_log_likelihood((a0, a1, a2), (p0, p1, p2))

                dist_change = abs((log_likelihood_new - log_likelihood) / log_likelihood)
                # print(f"Img {f}")
                # print("Step %d" % n_steps)
                # print("Distribution change %f" % dist_change)
                # print(a0, a1, a2)

                n_steps += 1

                # check whether we reached desired precision or max number of steps
                if (n_steps >= MAX_STEPS) or (dist_change <= min_change):
                    print("Loop stopped")
                    break
                else:
                    log_likelihood = log_likelihood_new

            y_pred = np.argmax(weights, axis=1)
            segment_nii_atlas = integrate_atlas_nii(test_img, y_pred, features_nonzero, 
                                       features_nonzero_row_indicies, weights, CSF_atlas, 
                                       GM_atlas, WM_atlas)

            # Calculate DICE
            all_dice[i,0], all_dice[i,1], all_dice[i,2] = dice_similarity(segment_nii_atlas, groundtruth_img)

            # Make directory to save result seg
            new_dir = join(result_path,f)
            os.mkdir(new_dir)
            nib.save(segment_nii_atlas, join(new_dir,'atlas_EM_seg.nii.gz'))

        save_dice(out_dice_path, onlydirs, all_dice)
        print(np.mean(all_dice, axis=0))

