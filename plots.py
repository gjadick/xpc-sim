#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 14 13:22:26 2023

@author: giavanna
"""

import os
import numpy as np
import matplotlib.pyplot as plt

save_output = True

plt.rcParams.update({
    # figure
    "figure.dpi": 600,
    # text
    "font.size":10,
    "font.family": "serif",
    "font.serif": ['Computer Modern Roman'],
    "text.usetex": True,
    # axes
    "axes.titlesize":10,
    "axes.labelsize":8,
    "axes.linewidth": .5,
    # ticks
    "xtick.top": True, 
    "ytick.right": True, 
    "xtick.direction": "in", 
    "ytick.direction": "in",
    "xtick.labelsize":8,
    "ytick.labelsize":8,
    # grid
    "axes.grid" : False, 
     "grid.color": "lightgray",
     "grid.linestyle": ":",
     # legend
     "legend.fontsize":8,
     # lines
     #"lines.markersize":5,
     "lines.linewidth":1,
     })

# #%% PLOT : SNR vs prop_dist for different grid pixel size (same upsample frequency)

# main_dir = 'output/spie2024/'
# sub_dirs = ['crlb_PMMA_Al_x16_025um', 'crlb_PMMA_Al_x32_05um', 'crlb_PMMA_Al_x64_10um']


#%% PLOT : : SNR vs. prop dist


main_dir = 'output/spie24'
sub_dirs = ['crlb_tissue_bone_x64_10um', 'crlb_tissue_fat_x64_10um']
Nx = 128


fig, ax = plt.subplots(1,2, dpi=300, figsize=[7,3])  # sphere order swapped to match SPIE
ax[0].set_title('Sphere 1, $r$ = 400 $\mu$m')        
ax[1].set_title('Sphere 2, $r$ = 100 $\mu$m')        

for sub_dir, tls, tcol in zip(sub_dirs, ['-','--'], ['b','r']):
    snr1 = np.fromfile(f'{main_dir}/{sub_dir}/snr1_float32.bin', dtype=np.float32)
    snr2 = np.fromfile(f'{main_dir}/{sub_dir}/snr2_float32.bin', dtype=np.float32)
    prop_dists = np.fromfile(f'{main_dir}/{sub_dir}/propdists_float32.bin', dtype=np.float32)
    mat1, mat2 = sub_dir.replace('fat', 'adipose').split('_')[1:3]
    ax[0].plot(prop_dists*1e3, snr1, ls=tls, color=tcol, marker='', label=f'{mat1} ({mat1}/{mat2})')
    ax[1].plot(prop_dists*1e3, snr2, ls=tls, color=tcol, marker='', label=f'{mat2} ({mat1}/{mat2})')
    
for i in range(2):
    ax[i].set_xlabel('propagation distance [mm]')
    ax[i].set_ylabel(f'SNR$_{i+1}$')
    ax[i].legend()
    
fig.tight_layout()
# if save_output:
#     plt.savefig(f'{main_dir}/fig_{fig_id}_snr.pdf')
plt.savefig(f'{main_dir}/fig_multi_snr.pdf')
plt.show()




#%% PLOT : Example noisy images.

main_dir = 'output/spie24'
sub_dirs = ['crlb_tissue_fat_x64_10um']#,'crlb_tissue_tissue_x64_10um', 'crlb_tissue_bone_x64_10um', 'crlb_PMMA_Al_x64_10um']
sI_i = '1e4'
Nx = 128
t_propdists = np.array([0, 0.1, 0.2, 0.3])

FOV = 128*10  # FOV in microns
kw = {'cmap':'gray', 'vmin':0.9, 'vmax':1.1, 'extent':[-FOV/2, FOV/2, -FOV/2, FOV/2]}
I_i = float(sI_i)
for sub_dir in sub_dirs:
    for Ei in ['E1', 'E2']:
        fig_id = '_'.join(sub_dir.split('_')[1:3]) + '_' + Ei
        print(fig_id)
        
        t_imgs = [np.fromfile(f'{main_dir}/{sub_dir}/R{int(R/1e-3):03}mm/img_{Ei}_{Nx}_float32.bin', 
                              dtype=np.float32).reshape([Nx, Nx]) for R in t_propdists]
        t_imgs_noisy = [np.random.poisson(I_i*t_img)/I_i for t_img in t_imgs]
        
        fig, ax = plt.subplots(1, len(t_propdists), figsize=[8,2.5], sharey=True)
        ax[0].set_ylabel('$y$ [$\mu$m]')
        for i in range(len(ax)):
            m = ax[i].imshow(t_imgs_noisy[i], **kw)
            ax[i].set_xlabel('$x$ [$\mu$m]')
            ax[i].set_title(f'$R$ = {1e3*t_propdists[i]:.0f} mm')
            
        cbaxes = fig.add_axes([1., 0.142, 0.02, 0.733]) 
        cb = plt.colorbar(m, cax=cbaxes, format='%.2f', label='normalized counts')
        
        fig.tight_layout(pad=0.3)
        if save_output:
            plt.savefig(f'{main_dir}/fig_{fig_id}_imgs_{sI_i}noise.pdf', bbox_inches='tight')
            print(f'{main_dir}/fig_{fig_id}_imgs_{sI_i}noise.pdf')
        plt.show()

    

# #%% Lorentzian

# PI = 3.1415926
# import cupy as cp

# #%%
# def lorentzian2D(x, y, fwhm):
#     gamma = fwhm/2
#     X, Y = cp.meshgrid(x, y)
#     return gamma / (2 * PI * (X**2 + Y**2 + gamma**2)**1.5)
    

# px_sz = 10e-6/64
# fwhm = 1e-6
# N_kernel = 27
# FOV_kern_x, FOV_kern_y = N_kernel*px_sz, N_kernel*px_sz
# d_x = cp.linspace(-FOV_kern_x/2, FOV_kern_x/2, N_kernel)
# d_y = cp.linspace(-FOV_kern_y/2, FOV_kern_y/2, N_kernel)
# d_lorentzian2d = lorentzian2D(d_x, d_y, fwhm)

# lorentzian2d = (d_lorentzian2d / cp.sum(d_lorentzian2d)).get()

# plt.imshow(lorentzian2d)
# plt.colorbar()
# plt.show()

# print(np.sum(lorentzian2d))







