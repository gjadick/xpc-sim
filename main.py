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


"""

import numpy as np
import cupy as cp
import matplotlib.pyplot as plt
from time import time
import os
dir_path = os.path.dirname(os.path.realpath(__file__))
os.chdir(dir_path)
    
from helpers import read_parameter_file_ct
from xscatter import get_delta_beta_mix
from xcompy import mixatten
from xpc import multislice_xpc_ct, do_recon_patch, paganin_thickness_sino


#################################################################
###
### INPUTS 
###

paramfile = 'input/params/params_voxel.txt'
show_phantom = True
propdists = [0, 200e-3]
do_phase_retrieval = False
N_proj_batch = 1000  # make ct.N_proj is an integer multiple of N_proj_batch

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
    
            
            
if __name__ == '__main__':
    
    # Parse params
    parameter_sets = read_parameter_file_ct(paramfile)
    params = parameter_sets[0]
    run_id, _, N_slices, upx, wave, phantom, ct, N_matrix, FOV, ramp = params
    outdir = f'output/{run_id}/'
    os.makedirs(outdir, exist_ok=True)

    do_multislice = N_slices > 1

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
    for R in propdists:
        t0 = time()
    
        N_batches = int(ct.N_proj / N_proj_batch)
        
        sino_list_prj = []
        sino_list_ms = []
        for i in range(N_batches):
            print(f'Starting theta batch {i+1} / {N_batches}')
            i0 = i * N_proj_batch
            thetas_batch = ct.thetas[i0:i0+N_proj_batch]
            
            # proj approx
            d_sino_batch_prj = multislice_xpc_ct(ct, phantom, wave, prop_dist=R, upx=upx, thetas=thetas_batch, N_slices=0)
            sino_list_prj.append(d_sino_batch_prj)
            
            # multislice
            if do_multislice:
                d_sino_batch_ms = multislice_xpc_ct(ct, phantom, wave, prop_dist=R, upx=upx, thetas=thetas_batch, N_slices=N_slices)
                sino_list_ms.append(d_sino_batch_ms)
       
        print(f'(R = {int(R*1e3)} mm) exit waves done - {time() - t0:.2f} s / {(time() - t0)/60:.1f} min')


        # Combine theta batches into one sinogram & reconstruct.
        d_sino_prj = cp.array(sino_list_prj).reshape([ct.N_proj, ct.N_channels]).clip(0, None)
        logsino_prj = -cp.log(d_sino_prj).get()
        save_and_imshow(logsino_prj, f'R{int(R*1e3):03}mm_sino_prj_{ct.N_proj}_{ct.N_channels}_float32', outdir)
        recon_prj = do_recon_patch(logsino_prj, ct, N_matrix, FOV, ramp)
        save_and_imshow(recon_prj, f'R{int(R*1e3):03}mm_recon_prj_{N_matrix}_float32', outdir)

        if do_multislice:
            d_sino_ms = cp.array(sino_list_ms).reshape([ct.N_proj, ct.N_channels]).clip(0, None)
            logsino_ms = -cp.log(d_sino_ms).get()
            save_and_imshow(logsino_ms, f'R{int(R*1e3):03}mm_sino_{N_slices}ms_{ct.N_proj}_{ct.N_channels}_float32', outdir)
            recon_ms = do_recon_patch(logsino_ms, ct, N_matrix, FOV, ramp)
            save_and_imshow(recon_ms, f'R{int(R*1e3):03}mm_recon_{N_slices}ms_{N_matrix}_float32', outdir)
        
        
        # Paganin
        if do_phase_retrieval: 
            delta, _ =  get_delta_beta_mix('H(11.2)O(88.8)', wave.energy, 1)  # assume water
            mu = mixatten('H(11.2)O(88.8)', np.array([wave.energy]))[0]
            
            T_sino_prj = paganin_thickness_sino(d_sino_prj, R, ct.beam_width, mu, delta)
            save_and_imshow(T_sino_prj.get(), f'R{int(R*1e3):03}mm_paganin_sino_prj_{ct.N_proj}_{ct.N_channels}_float32', outdir)
            T_recon_prj = do_recon_patch(T_sino_prj.get(), ct, N_matrix, FOV, ramp)
            save_and_imshow(T_recon_prj, f'R{int(R*1e3):03}mm_paganin_recon_prj_{N_matrix}_float32', outdir)

            if do_multislice:
                T_sino_ms = paganin_thickness_sino(d_sino_ms, R, ct.beam_width, mu, delta)
                save_and_imshow(T_sino_ms.get(), f'R{int(R*1e3):03}mm_paganin_sino_{N_slices}ms_{ct.N_proj}_{ct.N_channels}_float32', outdir)
                T_recon_ms = do_recon_patch(T_sino_ms.get(), ct, N_matrix, FOV, ramp)
                save_and_imshow(T_recon_ms, f'R{int(R*1e3):03}mm_paganin_recon_{N_slices}ms_{N_matrix}_float32', outdir)


        print(f'***\n*** R = {int(R*1e3)} mm finished - total time {time() - t0:.2f} s / {(time() - t0)/60:.1f} min \n***\n\n')




