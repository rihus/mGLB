# -*- coding: utf-8 -*-
"""
Created on 17 Nov 2023

@author: Riaz Hussain, PhD
         Research Fellow, Cincinnati Children's Hospital Medical Center
"""
# %% Import required libraries and declare path
# import time
import os
import sys
import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt
from scipy.stats import norm, boxcox, skewnorm
from scipy.special import inv_boxcox

# #Path to directory containing all healthy subject datasets
PARENT_DIR = r"main_dir"

#%% Define necessary functions
def create_new_folder(initial_dir, folder_name='results', not_create_if_exist=False):
    """
    Create a new folder with a new (unique name) in the specified directory.
    If folder_name already exists and is not empty, it'll append a number to it.
    Args:
        initial_dir (str): The initial directory where the new folder will be created.
        folder_name (str): The desired name of the folder. Default: 'results'
    Returns:
        str: The path to the newly created folder or the old folder (if not empty).
    """
    counter = 1
    new_folder = os.path.join(initial_dir, folder_name)
    while os.path.exists(new_folder):
        if os.listdir(new_folder) and not_create_if_exist is False:
            new_folder = os.path.join(initial_dir, folder_name + "_" + str(counter))
            counter += 1
            print("Old folder is not empty. Will append a number")
        elif os.listdir(new_folder) and not_create_if_exist is True:
            print("Folder already exists. Not creating new!")
            return new_folder
        else:
            print("Folder already exists and is empty...")
            return new_folder
    os.makedirs(new_folder)
    print(f"New folder created: {new_folder}")
    return new_folder

def save_txt_file(data, data_path, file_name='data.txt', txt_heading=None,
              subj_name=None):
    """
    Save data (must be as string) as a text file to the specified directory.
    If the file already exists, it'll create a new file with a number appended to the filename.
    Args:
        data (str): Content to be saved as a text file.
        data_path (str): Directory where the text file will be saved.
        file_name (str): Name of the text file. Default: 'file.txt'
        txt_heading (str): if provided inserts the heading on top of data
        subj_name (str): if provided, adds a column with subject name before data
    Returns:
        str: The path to the saved text file.
    """
    file_path = os.path.join(data_path, file_name)
    counter = 1
    while os.path.exists(file_path):
        file_name_parts = os.path.splitext(file_name)
        file_name = f"{file_name_parts[0]}_{counter}{file_name_parts[1]}"
        file_path = os.path.join(data_path, file_name)
        counter += 1
        print("file already exists. Will append a number")
    with open(file_path, 'w', encoding="utf-8") as file:
        if txt_heading:
            file.write(f"#\t{txt_heading}\n")
        if subj_name:
            file.write(f"{subj_name}\t")
        file.write(str(data))
    print(f"\nText file saved: {file_path}")

def plot_histogram(data_1d, subj_name, correc=None, save_path=None, suffix=""):
    """plot and save histograms
    Args: data_1d: 1D data array
          title (str): title string for the histogram plot
          save_path (str, optional): path to save the histogram. Default: None
          suffix (str, optional): type of data. Defaults to "".
    """
    plt.figure()
    plt.hist(data_1d, bins=100, histtype='bar', density=True, color='y')
    plt.xlabel('Normalized Intensity (arb.units)', fontsize=12, fontweight='bold')
    plt.ylabel('Fraction of Pixels', fontsize=12, fontweight='bold')
    if save_path is not None:
        save_histo = os.path.join(save_path,
                            f"{subj_name}_{correc}_{suffix.lower()}_Histogram.png")
        plt.savefig(save_histo, dpi=300)
    plt.show()

def boxcox_zscores_histogram(data_1d, results_dir, correc, analysis):
    """
    Tranform data to boxcox and save real-sapce z scores
    """
    # Box Cox transformation
    boxcox_trans, boxcox_lambda = boxcox(data_1d)
    # Plot histogram
    plt.hist(boxcox_trans, bins=100, histtype='bar', density=True, color='c')
    mu_norm, std_norm = norm.fit(boxcox_trans)
    print(f"\nBoxCox Mu: {mu_norm}\nBoxCox StD: {std_norm}")
    # Plot the PDF
    x = np.linspace(min(boxcox_trans), max(boxcox_trans), 100)
    p = norm.pdf(x, mu_norm, std_norm)
    plt.plot(x, p, 'm--', linewidth=2.5)
    # Title
    plt_title = [r'$\mu =$', f'{mu_norm:.4f}', r'$\sigma =$',
                 f'{std_norm:.4f}', r'$\lambda =$', f'{boxcox_lambda:.4f}']
    plt.title(' '.join(plt_title))
    plt.xlabel('Normalized Intensity (arb.units)', fontsize=12, fontweight='bold')
    plt.ylabel('Fraction of Pixels', fontsize=12, fontweight='bold')
    # Save histogram
    hist2_save = os.path.join(results_dir, f"{correc}_{analysis}_transformed_histogram.png")
    plt.savefig(hist2_save, dpi=300)
    plt.show(block=False)
    z_scores_real = calc_zscore(mu_norm, std_norm, boxcox_lambda)
    # Save Z-scores as a text file
    z_heading = " Z-scores = [-2, \t -1, \t 0, \t 1, \t 2] \n "
    save_txt_file(str(z_scores_real), results_dir, file_name=f"{correc}_{analysis}_z_scores.txt",
              txt_heading=z_heading)
    return z_scores_real

def calc_zscore(mu_norm, std_norm, boxcox_lambda):
    """Z-score calculation from box-cox transformed data"""
    zscores_bxcx = [mu_norm + i * std_norm for i in range(-2, 3)]
    zscores_r = inv_boxcox(zscores_bxcx, boxcox_lambda)
    zscores_real = [round(item, 6) for item in zscores_r]
    print("Z-Scores are: ", list(zscores_real))
    return zscores_real

def hist_overlay(data_1d, z_scores_real, results_dir, correc, analysis):
    """plot histogram overlay and save healthy fit parameters
    """
    _, ax = plt.subplots(figsize=(8, 4))
    # Draw vertical dashed lines
    for z_score in z_scores_real:
        ax.axvline(z_score, linestyle='--', color='black', linewidth=1)
    ax.axvspan(0, z_scores_real[0], facecolor='#ff7e7e', alpha=1)
    ax.axvspan(z_scores_real[0],z_scores_real[1], facecolor='#ffda7e', alpha=1)
    ax.axvspan(z_scores_real[1],z_scores_real[2], facecolor='#b2d9b2', alpha=1)
    ax.axvspan(z_scores_real[2],z_scores_real[3], facecolor='#7eff7e', alpha=1)
    ax.axvspan(z_scores_real[3],z_scores_real[4], facecolor='#7ec7da', alpha=1)
    ax.axvspan(z_scores_real[4],max(data_1d), facecolor='#7e7eff', alpha=1)
    xdata = np.linspace(min(data_1d), max(data_1d), 100)
    fit_skew, fit_loc, fit_scale = skewnorm.fit(data_1d)
    skew_fit_results = [fit_skew, fit_loc, fit_scale]
    skew_fit_results = [round(item, 6) for item in skew_fit_results]
    print(f"{analysis} Fit Skewness, location, and scale: {list(skew_fit_results)}")
    fit_heading = "Skewness, \t Location, \t Scale \n"
    save_txt_file(str(skew_fit_results), results_dir,
            file_name=f"{correc}_{analysis}_skew_fit.txt", txt_heading=fit_heading)
    skew_fit = skewnorm.pdf(xdata, fit_skew, fit_loc, fit_scale)
    plt.plot(xdata, skew_fit, 'k--', linewidth=2.5)
    plt.hist(data_1d, bins = 50, histtype = 'bar', density=True, color='white',
            edgecolor="black", alpha=0.5)
    plt.xticks(fontsize=14, weight='bold')
    plt.yticks(fontsize=14, weight='bold')
    plt.xlabel('Normalized Intensity (arb.units)', fontsize=14, fontweight='bold')
    plt.ylabel('Fraction of Pixels', fontsize=14, fontweight='bold')
    plt.tight_layout()
    hist3_save = os.path.join(results_dir, f"{correc}_{analysis}_fit_histogram.png")
    plt.savefig(hist3_save, dpi=300)
    plt.show()

#%% Importing files from all directories, loading data, making big 1D and taking 99th percentile
##Declare the names (or commom string) of the healthy subject files to be included
CORR_DICT = {"N4": "img_ventilation_N4.nii.gz",
                "keyhole": "img_ventilation_corrected.nii.gz"}

for correction, image_fname in CORR_DICT.items():
    CORR_TYPE = correction
    OPTION = image_fname
    dirs_list = os.listdir(PARENT_DIR)
    median_glb= create_new_folder(PARENT_DIR, folder_name=f"{CORR_TYPE}_median_glb_thresholds")
    all_dirs = []
    for a_folder in dirs_list:
        if a_folder.startswith('ILD-HC-0') or a_folder.startswith('IRC740H-0'): #
            all_dirs.append(PARENT_DIR + a_folder)
    #Save all directories list in text file
    with open(os.path.join(median_glb, "all_dirs_list.txt"), 'w', encoding="utf-8") as f:
        f.write('\n'.join(all_dirs))
    #Loop to go inside all_dirs, load data and append
    N_SUBJ=0
    big1d_median_healthy=[]
    for folder1 in all_dirs:
        subject_name = os.path.basename(folder1)
        print(f"Subject name: {subject_name}")
        filelist2 = os.listdir(folder1)
        for orig_img in filelist2:
            if orig_img == OPTION:
                print(f"Current folder is: {folder1}")
                img_path = folder1 + "/" + orig_img
                nifty_imgs = nib.load(img_path)
                imgs_data = nifty_imgs.get_fdata()
                N_SUBJ = N_SUBJ + 1
        for msk_img in filelist2:
            if msk_img == 'img_ventilation_mask.nii.gz':
                msk_path = folder1 + "/" + msk_img
                nifty_msks = nib.load(msk_path)
                msks_data = nifty_msks.get_fdata()
        ##mask the data
        masked_datas = imgs_data * msks_data
        oneD_masked_nZero = masked_datas[masked_datas != 0]
        ##View each individual histogram
        plot_histogram(oneD_masked_nZero, subject_name, correc=CORR_TYPE,
                       save_path=median_glb, suffix="full")
        ##Normalize at median and save histogram
        single_median_value = np.median(oneD_masked_nZero)
        single_median_norm = oneD_masked_nZero / single_median_value
        median_percentile = np.percentile(single_median_norm, 99, axis=0)
        single_median_norm[single_median_norm > median_percentile]= median_percentile
        plot_histogram(single_median_norm, subject_name, correc=CORR_TYPE,
                       save_path=median_glb, suffix="median_norm")
        ##Append all image data in 1D array
        big1d_median_healthy=np.append(big1d_median_healthy, single_median_norm)
    print("Total number of Healthy Subjects included:", N_SUBJ)
    ColumnOfHealthies2=np.size(big1d_median_healthy)
    ## Plot the histogram for all healthy Data, BoxCox Tranformation
    plot_histogram(big1d_median_healthy, "healthy",  correc=CORR_TYPE,
                   save_path=median_glb, suffix="median")
    # ##Plot and save big data file and
    data_median = os.path.join(median_glb, f"{CORR_TYPE}_median_big1D")
    np.save(data_median, big1d_median_healthy)
    # ##Box Cox transformation and transformed space histogram, and save healthy data bins
    zScores_median = boxcox_zscores_histogram(big1d_median_healthy, median_glb,
                                      CORR_TYPE, "glb_median")
    # ##Histogram overlay and save the image
    hist_overlay(big1d_median_healthy, zScores_median, median_glb,
                 CORR_TYPE, "glb_median")

# %%
