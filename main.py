#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan  9 12:14:33 2024

@author: giavanna

Simulate CT scans with phase contrast.

The main loop simulates images using two different forward models: (1) the
projection approximation, (2) the multislice approach. There is an option
to apply Paganin phase retrieval after.

xpc.py also has functions for simulating 2D planar images of either voxelized
or analytical phantoms. TODO: write a quick-start script for planar imaging.

TODO: (also)
    - Separate the Paganin phase retrieval in the CT sim function to its own function.
    - Separate out the recon and put it in the main loop.
    - Write functions for other phase retrieval methods.
    - Write radiography sim function.
"""

import numpy as np
import cupy as cp
import matplotlib.pyplot as plt
from time import time
import os
dir_path = os.path.dirname(os.path.realpath(__file__))
os.chdir(dir_path)
    
from helpers import read_parameter_file_proj, read_parameter_file_ct
from xscatter import get_delta_beta_mix
from xcompy import mixatten
from xpc import multislice_wave_voxels, free_space_propagate, detect_wave,\
    multislice_xpc_ct, do_recon_patch, paganin_thickness_sino


#################################################################
###
### INPUTS 
###

paramfile = 'input/params/params_voxel.txt'
paramfile_proj = 'input/params/params_projtest.txt'

#################################################################

    
plt.rcParams.update({
    'figure.dpi': 300,
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
    'legend.fontsize': 8,
    'lines.linewidth':1,
    })


    
def save_and_imshow(arr, title, outdir, kw={'cmap':'gray'}):
    """
    Convenience function for viewing and saving a 2D image.
    """
    arr.tofile(outdir + title + '.bin')
    if 'sino' in title:
        kw['aspect'] = 'auto'
    else:
        kw['aspect'] = 'equal'
    fig, ax = plt.subplots(1, 1, figsize=[4,3])
    m = ax.imshow(arr, **kw)
    ax.set_title(title)
    ax.set_xticks([])
    ax.set_yticks([])
    fig.colorbar(m)
    fig.tight_layout()
    plt.savefig(outdir + title + '.png')
    plt.show()
    
            
def simulate_xpc_ct(params, show_phantom=True, phase_retrieval=None, N_proj_batch_max=1000):
    run_id, R, N_slices, upx, wave, phantom, ct, N_matrix, FOV, ramp = params
    outdir = f'output/{run_id}/'
    os.makedirs(outdir, exist_ok=True)

    do_multislice = N_slices > 1
    if do_multislice:
        print(f'Using multislice forward model (N slices = {N_slices})')
    else:
        print('Using projection approximation forward model')
        
    # Show the phantom.
    d_phantom_delta, d_phantom_beta = phantom.delta_beta_slice(wave.energy, 0)
    if show_phantom:
        pfov = phantom.Nx * phantom.dx / 2 * 1e3
        kw = {'cmap':'bone', 'extent':(-pfov, pfov, -pfov, pfov)}
        fig, ax = plt.subplots(1, 2, figsize=[7.5,3])
        m = ax[0].imshow(d_phantom_delta.get(), vmin=4e-7, **kw)  # vmin chosen for TISSUE
        fig.colorbar(m, ax=ax[0])
        ax[0].set_title('delta')
        m = ax[1].imshow(d_phantom_beta.get(), **kw)
        fig.colorbar(m, ax=ax[1])
        ax[1].set_title('beta')
        for axi in ax: 
            axi.set_xlabel('x [micron]')
            axi.set_ylabel('y [micron]')
        fig.tight_layout()
        plt.savefig(outdir + 'phantom_native_contrast.png')
        plt.show()


    # Simulate CT scans at all propagation distances.
    N_batches = ct.N_proj // N_proj_batch_max
    N_proj_batches_all = [N_proj_batch_max for i in range(N_batches)]
    if ct.N_proj > N_proj_batch_max*N_batches:
        N_proj_batches_all.append(ct.N_proj - N_proj_batch_max*N_batches)  # leftover views

    d_sino = cp.zeros([ct.N_proj, ct.N_channels])
    t0 = time()
    for i, N_proj_batch in enumerate(N_proj_batches_all):
        print(f'\n*** CT batch {i+1} / {len(N_proj_batches_all)}  ({N_proj_batch} thetas)')
        i0 = i * N_proj_batch_max
        thetas_batch = ct.thetas[i0:i0+N_proj_batch]  # source locations for this batch
        d_sino[i0:i0+N_proj_batch] =  multislice_xpc_ct(ct, phantom, wave, R, upx, N_slices, thetas_batch).clip(0, None)
    print(f'(R = {int(R*1e3)} mm) exit waves done - {time() - t0:.2f} s / {(time() - t0)/60:.1f} min \n')

    logsino = -cp.log(d_sino).get()
    save_and_imshow(logsino, f'R{int(R*1e3):03}mm_logsino_{N_slices}slices_{ct.N_proj}_{ct.N_channels}_float32', outdir)


    # Reconstruct the raw phase-contrast CT image. 
    # # TODO: separate out of this mega-function.
    recon = do_recon_patch(logsino, ct, N_matrix, FOV, ramp)
    save_and_imshow(recon, f'R{int(R*1e3):03}mm_recon_{N_slices}slices_{N_matrix}_float32', outdir)
    
    
    # Perform phase retrieval. 
    # # TODO: separate out of mega-function and write a new function.
    # # TODO: output the retrieved image(s).
    # # TODO: other phase retrieval algorithms.
    if phase_retrieval is None:
        pass
    elif phase_retrieval=='paganin': 
        delta, _ =  get_delta_beta_mix('H(11.2)O(88.8)', wave.energy, 1)  # assume water
        mu = mixatten('H(11.2)O(88.8)', np.array([wave.energy]))[0]
        
        T_sino = paganin_thickness_sino(d_sino, R, ct.beam_width, mu, delta).get()
        save_and_imshow(T_sino, f'R{int(R*1e3):03}mm_paganin_sino_{N_slices}slices_{ct.N_proj}_{ct.N_channels}_float32', outdir)
        T_recon = do_recon_patch(T_sino, ct, N_matrix, FOV, ramp)
        save_and_imshow(T_recon, f'R{int(R*1e3):03}mm_paganin_recon_{N_slices}slices_{N_matrix}_float32', outdir)
    else:
        print(f'`phase_retrieval` = {phase_retrieval} is not a valid argument! Skipping phase retrieval...')

    print(f'\n***\n*** R = {int(R*1e3)} mm finished - total time {time() - t0:.2f} s / {(time() - t0)/60:.1f} min \n***\n\n')
    return logsino, recon


def simulate_xpc_projection(params):  # TODO!
    run_id, R, N_slices, upx, in_wave, phantom = params
    outdir = f'output/{run_id}/'
    os.makedirs(outdir, exist_ok=True)
    
    try:
        phantom.d_voxels  # check for VoxelPhantom class 
    except:
        print('invalid phantom type!')  # only voxel phantoms for now
    
    phantom.resample_xy(in_wave.Nx, in_wave.Ny, in_wave.dx, in_wave.dy)   
    img = phantom.voxels[0]
    
    d_exit_wave = multislice_wave_voxels(in_wave, phantom, N_slices)
    d_fsp_wave = free_space_propagate(in_wave, d_exit_wave, R)
    img = detect_wave(in_wave, d_fsp_wave, upx, blur_fwhm=0).get()

    save_and_imshow(img, f'R{int(R*1e3):03}mm_{N_slices}slices_float32', outdir)

    return img

            
if __name__ == '__main__':
    
    test_project = True
    test_ct = True
    
    if test_project:
        parameter_sets = read_parameter_file_proj(paramfile_proj)
        imgs = []
        for params in parameter_sets:
            img = simulate_xpc_projection(params)
            imgs.append(img)
                    
        # Zoom in on an ROI to better see the difference with phase contrast
        x0 = 600
        y0 = 500
        dx = 300
        dy = 300
        kw = {'cmap':'bwr', 'vmin':0, 'vmax':1.2}
        for i, img in enumerate(imgs):
            R = parameter_sets[i][1]    
            roi = img[y0:y0+dy, x0:x0+dx]  # a region-of-interest
            fig, ax = plt.subplots(1,2,figsize=[8,4])
            ax[0].imshow(img, aspect='auto', **kw)
            ax[0].plot([x0, x0, x0+dx, x0+dx, x0], [y0, y0+dy, y0+dy, y0, y0], 'k-')
            m = ax[1].imshow(roi, extent=(x0, x0+dx, y0, y0+dy), **kw)
            fig.colorbar(m, ax=ax[1])
            fig.suptitle(f'Propagation distance = {R*1e3:.0f} mm')
            fig.tight_layout()
            plt.show()
            

    if test_ct:
        parameter_sets = read_parameter_file_ct(paramfile)
        recons = []
        for params in parameter_sets:
            logsino, recon = simulate_xpc_ct(params)#, phase_retrieval='paganin')
            recons.append(recon)
        
        # Zoom in on an ROI to better see the difference with phase contrast
        x0 = 600
        y0 = 500
        dx = 300
        dy = 150
        kw = {'cmap':'bwr', 'vmin':0, 'vmax':8000}
        for i, recon in enumerate(recons):
            R = parameter_sets[i][1]
    
            roi = recon[y0:y0+dy, x0:x0+dx]  # a region-of-interest
            fig, ax = plt.subplots(1,2,figsize=[8,4])
            ax[0].imshow(recon, aspect='auto', **kw)
            ax[0].plot([x0, x0, x0+dx, x0+dx, x0], [y0, y0+dy, y0+dy, y0, y0], 'k-')
            m = ax[1].imshow(roi, extent=(x0, x0+dx, y0, y0+dy), **kw)
            fig.colorbar(m, ax=ax[1])
            fig.suptitle(f'Propagation distance = {R*1e3:.0f} mm')
            fig.tight_layout()
            plt.show()
            
        
        