#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 13 14:09:05 2023

@author: giavanna

A numerical approach to approximating the Cramer-Rao Lower Bound
of spectral x-ray phase-contrast imaging in the context of material decomposition
of a two-sphere object.

"""

import matplotlib.pyplot as plt
from time import time
from datetime import timedelta
from helpers import read_parameter_file
from forward import project_wave_2phantoms, free_space_propagate, detect_wave
import numpy as np
import os
import shutil
import copy

plt.rcParams.update({
    'figure.dpi': 600,
    'font.size':10,
    'font.family': 'serif',
    'font.serif': ['Computer Modern Roman'],
    'text.usetex': True,
    'axes.titlesize':10,
    'axes.labelsize':8,
    'axes.linewidth': .5,
    'xtick.top': True, 
    'ytick.right': True, 
    'xtick.direction': 'in', 
    'ytick.direction': 'in',
    'xtick.labelsize':8,
    'ytick.labelsize':8,
    'legend.fontsize':8,
    'lines.linewidth':1,
     })



#%%
if __name__ == '__main__':

    # Parameter file should have two variables: energy and propagation distance.
    # For efficiency this assumes everything else is the same.
    #param_file = 'input/params/spie2024_crlb/tissue_bone.txt'
    param_file = 'input/params/spie2024_crlb/tissue_fat.txt'
    drad = 1e-12  # change in radius for finite diff. approx.
    save_output = True

    # Extract the energies and propagation distances.
    # Other variables default to the values in the first parameter set.
    parameter_sets = read_parameter_file(param_file)
    run_id, _, N_slices, upsample_multiple, _, phantom = parameter_sets[0]
    sphere1, sphere2 = phantom  
    waves, energies, prop_dists = [], [], []
    for parameters in parameter_sets:
        _, propagation_distance, _, _, wave, _ = parameters
        if wave.energy not in energies:
            energies.append(wave.energy)
            waves.append(wave)
        if propagation_distance not in prop_dists:
            prop_dists.append(propagation_distance)
    E1, E2 = energies
    wave_E1, wave_E2 = waves
    prop_dists = np.arange(prop_dists[0], prop_dists[-1]+1e-3, 1e-3)  # generate range of propagation distances
    #prop_dists = np.arange(prop_dists[0], prop_dists[-1]+1e-3, 1e-1)  # generate range of propagation distances
    #print(prop_dists)
    
    # Create the output directory.
    print(run_id)
    main_dir = f'./output/{run_id}'
    if save_output:
        os.makedirs(main_dir, exist_ok=True)
        shutil.copy(param_file, f'{main_dir}/parameters.txt')
#%%
    ### Compute the CRLBs.
    t0 = time()

    all_imgs = []
    all_slices = []
    
    # Before looping, pre-compute the exit waves + finite difference approx.
    # As long as there is enough GPU memory, this should be more efficient.
    sphere1_diff = copy.copy(sphere1)
    sphere2_diff = copy.copy(sphere2)
    sphere1_diff.update_radius(sphere1.radius + drad)
    sphere2_diff.update_radius(sphere2.radius + drad)
    
    d_exit_wave_E1 = project_wave_2phantoms(wave_E1, sphere1, sphere2)
    d_exit_wave_E1_diff1 = project_wave_2phantoms(wave_E1, sphere1_diff, sphere2)
    d_exit_wave_E1_diff2 = project_wave_2phantoms(wave_E1, sphere1, sphere2_diff)

    d_exit_wave_E2 = project_wave_2phantoms(wave_E2, sphere1, sphere2)
    d_exit_wave_E2_diff1 = project_wave_2phantoms(wave_E2, sphere1_diff, sphere2)
    d_exit_wave_E2_diff2 = project_wave_2phantoms(wave_E2, sphere1, sphere2_diff)

    print(f'Exit waves calculated - {timedelta(seconds=int(time()-t0))}')

    all_snr1, all_snr2, all_img1, all_img2 = [], [], [], []
    for propagation_distance in prop_dists:

        d_fsp_wave_E1 = free_space_propagate(wave_E1, d_exit_wave_E1, propagation_distance)
        d_fsp_wave_E2 = free_space_propagate(wave_E1, d_exit_wave_E2, propagation_distance)
        img_E1 = detect_wave(wave_E1, d_fsp_wave_E1, upsample_multiple).get()
        img_E2 = detect_wave(wave_E2, d_fsp_wave_E2, upsample_multiple).get()

        d_fsp_wave_E1_diff1 = free_space_propagate(wave_E1, d_exit_wave_E1_diff1, propagation_distance)
        d_fsp_wave_E1_diff2 = free_space_propagate(wave_E1, d_exit_wave_E1_diff2, propagation_distance)
        d_fsp_wave_E2_diff1 = free_space_propagate(wave_E1, d_exit_wave_E2_diff1, propagation_distance)
        d_fsp_wave_E2_diff2 = free_space_propagate(wave_E1, d_exit_wave_E2_diff2, propagation_distance)
        img_E1_diff1 = detect_wave(wave_E1, d_fsp_wave_E1_diff1, upsample_multiple).get()
        img_E1_diff2 = detect_wave(wave_E1, d_fsp_wave_E1_diff2, upsample_multiple).get()
        img_E2_diff1 = detect_wave(wave_E2, d_fsp_wave_E2_diff1, upsample_multiple).get()
        img_E2_diff2 = detect_wave(wave_E2, d_fsp_wave_E2_diff2, upsample_multiple).get()
        
        diff1_E1 = (img_E1_diff1 - img_E1) / drad
        diff2_E1 = (img_E1_diff2 - img_E1) / drad
        diff1_E2 = (img_E2_diff1 - img_E2) / drad
        diff2_E2 = (img_E2_diff2 - img_E2) / drad

        # Use signals/differences to compute the Fisher matrix and CRLB.
        F = np.zeros([2,2], dtype=np.float64)  # init Fisher matrix
        F[0,0] = np.sum((diff1_E1**2 / img_E1) + (diff1_E2**2 / img_E2))
        F[0,1] = np.sum((diff1_E1 * diff2_E1 / img_E1) + (diff1_E2 * diff2_E2 / img_E2))
        F[1,0] = np.sum((diff2_E1 * diff1_E1 / img_E1) + (diff2_E2 * diff1_E2 / img_E2))
        F[1,1] = np.sum((diff2_E1**2 / img_E1) + (diff2_E2**2 / img_E2))
        Fi = np.linalg.inv(F)
        crlb1 = Fi[0,0]
        crlb2 = Fi[1,1]
        snr1 = sphere1.radius / np.sqrt(crlb1)
        snr2 = sphere2.radius / np.sqrt(crlb2)
        
        #print(f'R = {propagation_distance/1e-3:.0f} mm - {timedelta(seconds=int(time()-t0))} s - ({snr1:.2f}, \t{snr2:.2f})')
        
        # Store some outputs for plotting later.
        all_snr1.append(snr1)
        all_snr2.append(snr2)
        all_img1.append(img_E1)
        all_img2.append(diff2_E1)
        # Save the detected signal images and SNR values.
        if save_output:
            sub_dir = os.path.join(main_dir, f'R{int(propagation_distance/1e-3):03}mm')
            os.makedirs(sub_dir, exist_ok=True)
            Nx, Ny = img_E1.shape
            img_E1.astype(np.float32).tofile(f'{sub_dir}/img_E1_{Nx}_float32.bin')
            img_E2.astype(np.float32).tofile(f'{sub_dir}/img_E2_{Nx}_float32.bin')
            diff1_E1.astype(np.float32).tofile(f'{sub_dir}/diff1_E1_{Nx}_float32.bin')
            diff1_E2.astype(np.float32).tofile(f'{sub_dir}/diff1_E2_{Nx}_float32.bin')
            diff2_E1.astype(np.float32).tofile(f'{sub_dir}/diff2_E1_{Nx}_float32.bin')
            diff2_E2.astype(np.float32).tofile(f'{sub_dir}/diff2_E2_{Nx}_float32.bin')

    print(f'All signals calculated - {timedelta(seconds=int(time()-t0))}')
    
    # Save the detected signal images and SNR values.
    if save_output:
        sub_dir = os.path.join(main_dir, f'R{int(propagation_distance/1e-3):03}mm')
        os.makedirs(sub_dir, exist_ok=True)
        np.array(prop_dists).astype(np.float32).tofile(f'{main_dir}/propdists_float32.bin')
        np.array(all_snr1).astype(np.float32).tofile(f'{main_dir}/snr1_float32.bin')
        np.array(all_snr2).astype(np.float32).tofile(f'{main_dir}/snr2_float32.bin')
  
    
  
#%%
    fig_id = f'{sphere1.name.split("_")[1]}_{sphere2.name.split("_")[1]}'
    # PLOT 1 : SNR vs. prop dist
    fig, ax = plt.subplots(1,2, dpi=300, figsize=[7,3])  # sphere order swapped to match SPIE
    for i, sphere in enumerate(phantom):
        geoname, matname, diameter = sphere.name.split('_')
        radius = float(diameter[:-2])/2
        ax[i].set_title(f'Sphere {i+1} - {matname}, $r$ = {int(radius)} $\mu$m')        
        ax[i].set_xlabel('propagation distance [mm]')
        ax[i].set_ylabel(f'SNR$_{i+1}$')
    ax[0].plot(prop_dists*1e3, all_snr1, marker='')
    ax[1].plot(prop_dists*1e3, all_snr2, marker='')
    fig.tight_layout()
    if save_output:
        plt.savefig(f'{main_dir}/fig_{fig_id}_snr.pdf')
    plt.show()
    

#     #%%
    # PLOT 2 : example  imgs for 4 prop dist
    N = 100 # int(len(prop_dists)/4)
    t_imgs = all_img1[::N]
    t_propdists = prop_dists[::N]
    kw = {'cmap':'gray'}
    fig, ax = plt.subplots(1, 4, figsize=[10,2.5])
    for i in range(len(ax)):
        m = ax[i].imshow(t_imgs[i], **kw)
        fig.colorbar(m, ax=ax[i])
        ax[i].axis('off')
        ax[i].set_title(f'R = {1e3*t_propdists[i]:.0f} mm')
    fig.suptitle(f'dr = {drad:.1e} $\quad \\times${upsample_multiple} upsampling')
    fig.tight_layout(pad=1)
    if save_output:
        plt.savefig(f'{main_dir}/fig_{fig_id}_imgs.pdf')
    plt.show()
    
    
    # FOV=128*10
    # kw = {'cmap':'gray', 'vmin':0.9, 'vmax':1.1, 'extent':[-FOV/2, FOV/2, -FOV/2, FOV/2]}
    # fig, ax = plt.subplots(1, 4, figsize=[8,2.5], sharey=True)
    # ax[0].set_ylabel('$y$ [$\mu$m]')
    # for i in range(len(ax)):
    #     m = ax[i].imshow(all_img1[i], **kw)
    #     ax[i].set_xlabel('$x$ [$\mu$m]')
    #     ax[i].set_title(f'$R$ = {1e3*prop_dists[i]:.0f} mm')
    # cbaxes = fig.add_axes([1., 0.142, 0.02, 0.733]) 
    # cb = plt.colorbar(m, cax=cbaxes, format='%.2f', label='normalized counts')
    # fig.tight_layout(pad=0.3)
    # plt.show()

    
    
