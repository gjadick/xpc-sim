#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan  9 12:14:33 2024

@author: giavanna

Simulate CT scans with phase contrast.
"""


import matplotlib.pyplot as plt
from time import time
from helpers import read_parameter_file_ct
#from forward import project_wave_2phantoms, free_space_propagate, detect_wave
import numpy as np
import cupy as cp

from xtomosim.forward_project import siddons_2D
from xtomosim.back_project import pre_process, get_recon_coords, do_fbp


plt.rcParams.update({
    # figure
    "figure.dpi": 150,
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


if __name__ == '__main__':
    
    # parse params
    parameter_sets = read_parameter_file_ct('input/params/params_voxel.txt')
    params = parameter_sets[0]
    run_id, propdist, multislices, upsample_multiple, wave, phantom, ct = params

    # generate beta, delta phantom matrix
    d_phantom_delta, d_phantom_beta = phantom.delta_beta_slice(wave.energy, 0)
    
    # # %% show phantom
    show_phantom = False
    if show_phantom:
        fig, ax = plt.subplots(1, 2, figsize=[7.5,3])
        m = ax[0].imshow(d_phantom_delta.get(), cmap='bone')
        fig.colorbar(m, ax=ax[0])
        ax[0].set_title('delta')
        m = ax[1].imshow(d_phantom_beta.get(), cmap='bone')
        fig.colorbar(m, ax=ax[1])
        ax[1].set_title('beta')
        fig.tight_layout()
        plt.show()
    

    ### RAYTRACE
    show_raytrace = True
    if show_raytrace:
        # Get coordinates for each source --> detector channel
        d_thetas = cp.tile(cp.array(ct.thetas + cp.pi, dtype=cp.float32)[:, cp.newaxis], ct.N_channels).ravel()  # use newaxis for correct tiling
        d_channels = cp.tile(cp.array(ct.channels, dtype=cp.float32), ct.N_proj)
        src_x = ct.SID * cp.cos(d_thetas) + d_channels * cp.cos(d_thetas + cp.pi/2)
        src_y = ct.SID * cp.sin(d_thetas) + d_channels * cp.sin(d_thetas + cp.pi/2)
        trg_x = (ct.SDD - ct.SID) * cp.cos(d_thetas + cp.pi) + d_channels * cp.cos(d_thetas + cp.pi/2)
        trg_y = (ct.SDD - ct.SID) * cp.sin(d_thetas + cp.pi) + d_channels * cp.sin(d_thetas + cp.pi/2)
    
        matrix_stack = cp.array([d_phantom_delta, d_phantom_beta], dtype=cp.float32) 
        sino_stack = cp.zeros([ct.N_proj, ct.N_channels, len(matrix_stack)], dtype=np.float32)
        t0 = time()
        for i in range(len(matrix_stack)): 
            line_integrals = siddons_2D(src_x, src_y, trg_x, trg_y, matrix_stack[i], phantom.dx)
            sino_stack[:,:,i] = line_integrals.reshape([ct.N_proj, ct.N_channels])
            print(f'raytracing done for {i+1} / {len(matrix_stack)}, t={time() - t0:.2f}s')
        d_sino_delta, d_sino_beta = sino_stack[:,:,0], sino_stack[:,:,1]
        sino_delta, sino_beta = d_sino_delta.get(), d_sino_beta.get()

        fig, ax = plt.subplots(1, 2, figsize=[7.5,3])
        m = ax[0].imshow(sino_delta, aspect='auto', cmap='bone')
        fig.colorbar(m, ax=ax[0])
        ax[0].set_title('delta')
        m = ax[1].imshow(sino_beta, aspect='auto', cmap='bone')
        fig.colorbar(m, ax=ax[1])
        ax[1].set_title('beta')
        fig.tight_layout()
        plt.show()


    ### RECONSTRUCT
    show_recon = True
    if show_recon:
        N_matrix, FOV, ramp = 1024, 2.40955e-3, 1.0
        r_matrix, theta_matrix = get_recon_coords(N_matrix, FOV)
        sino_filtered_delta = pre_process(sino_delta, ct, ramp)
        sino_filtered_beta = pre_process(sino_beta, ct, ramp)
        
        # For big matrix, divide into patches (otherwise weird results)
        N_matrix_patch = 512  
        N_patches = np.ceil(N_matrix / N_matrix_patch).astype(int)
        if (N_matrix % N_matrix_patch) > 0:
            print('Changing N_matrix to be a multiple of {N_matrix_patch}!')
            N_matrix = N_patches * N_matrix_patch
            
        recon_delta, recon_beta = np.zeros([2, N_matrix, N_matrix], dtype=np.float32)
        for i_patch in range(N_patches):
            i0 = N_matrix_patch * i_patch
            for j_patch in range(N_patches):
                print(f'Reconstructing patch {i_patch*N_patches + j_patch + 1} / {N_patches ** 2}')
                j0 = N_matrix_patch * j_patch
                r_matrix_patch = r_matrix[i0:i0+N_matrix_patch, j0:j0+N_matrix_patch] 
                theta_matrix_patch = theta_matrix[i0:i0+N_matrix_patch, j0:j0+N_matrix_patch] 
                
                patch_delta = do_fbp(sino_filtered_delta, r_matrix_patch, theta_matrix_patch, ct.SID, ct.s, ct.dtheta)
                patch_beta = do_fbp(sino_filtered_beta, r_matrix_patch, theta_matrix_patch, ct.SID, ct.s, ct.dtheta)
                
                recon_delta[i0:i0+N_matrix_patch, j0:j0+N_matrix_patch] = patch_delta 
                recon_beta[i0:i0+N_matrix_patch, j0:j0+N_matrix_patch] = patch_beta
        
        recon_delta = recon_delta.clip(0, None)  # non-negativity
        recon_beta = recon_beta.clip(0, None)
    
        # Compare delta recon vs. original
        HFOVmm = 1e3 * FOV / 2  # half FOV [mm]
        kwd = {'vmin':3.5e-7, 'vmax':4.8e-7, 'cmap':'bone', 'aspect':'auto', 'extent':(-HFOVmm, HFOVmm, -HFOVmm, HFOVmm)}
        fig, ax = plt.subplots(1, 2, figsize=[7.5,3])
        m = ax[0].imshow(d_phantom_delta.get(), **kwd)
        fig.colorbar(m, ax=ax[0])
        ax[0].set_title('original (delta)')
        m = ax[1].imshow(recon_delta, **kwd)
        fig.colorbar(m, ax=ax[1])
        ax[1].set_title('reconstruction (delta)')
        fig.tight_layout()
        plt.show()
        
        # Compare beta recon vs. original
        kwb = {'vmin':0, 'vmax':3.5e-8, 'cmap':'bone', 'aspect':'auto', 'extent':(-HFOVmm, HFOVmm, -HFOVmm, HFOVmm)}
        fig, ax = plt.subplots(1, 2, figsize=[7.5,3])
        m = ax[0].imshow(d_phantom_beta.get(), **kwb)
        fig.colorbar(m, ax=ax[0])
        ax[0].set_title('original (beta)')
        m = ax[1].imshow(recon_beta, **kwb)
        fig.colorbar(m, ax=ax[1])
        ax[1].set_title('reconstruction (beta)')
        fig.tight_layout()
        plt.show()
        
        # Show the recons
        fig, ax = plt.subplots(1, 2, figsize=[7.5,3])
        m = ax[0].imshow(recon_delta, **kwd)
        fig.colorbar(m, ax=ax[0])
        ax[0].set_title('delta recon (phase)')
        m = ax[1].imshow(recon_beta, **kwb)
        fig.colorbar(m, ax=ax[1])
        ax[1].set_title('beta recon (absorption)')
        fig.tight_layout()
        plt.show()









