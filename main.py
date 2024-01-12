#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan  9 12:14:33 2024

@author: giavanna

Simulate CT scans with phase contrast.
"""


import matplotlib.pyplot as plt
from time import time
import os
dir_path = os.path.dirname(os.path.realpath(__file__))
os.chdir(dir_path)
    
from helpers import read_parameter_file_ct
from forward import project_db, propagate_1D # project_wave_2phantoms, free_space_propagate, detect_wave
import numpy as np
import cupy as cp

from xtomosim.forward_project import siddons_2D
from xtomosim.back_project import pre_process, get_recon_coords, do_fbp


def project_xpci(ct, phantom, wave, prop_dist, upx, thetas=None):  # upx = upsample factor

    # init phantom
    d_phantom_delta, d_phantom_beta = phantom.delta_beta_slice(wave.energy, 0)
    matrix_stack = cp.array([d_phantom_delta, d_phantom_beta], dtype=cp.float32) 
    
    # init line integral coord's
    if thetas is None:
        thetas = ct.thetas
    N_thetas = len(thetas)
    s_upx = ct.s / upx
    N_channels_upx = ct.N_channels * upx
    channels_upx = np.arange(-ct.beam_width/2, ct.beam_width/2, s_upx) + s_upx/2
    if len(channels_upx) > N_channels_upx:
        channels_upx = channels_upx[:N_channels_upx]
    d_channels = cp.tile(cp.array(channels_upx, dtype=cp.float32), N_thetas)

    d_thetas = cp.tile(cp.array(thetas + cp.pi, dtype=cp.float32)[:, cp.newaxis], N_channels_upx).ravel()  
    src_x = ct.SID * cp.cos(d_thetas) + d_channels * cp.cos(d_thetas + cp.pi/2)
    src_y = ct.SID * cp.sin(d_thetas) + d_channels * cp.sin(d_thetas + cp.pi/2)
    trg_x = (ct.SDD - ct.SID) * cp.cos(d_thetas + cp.pi) + d_channels * cp.cos(d_thetas + cp.pi/2)
    trg_y = (ct.SDD - ct.SID) * cp.sin(d_thetas + cp.pi) + d_channels * cp.sin(d_thetas + cp.pi/2)

    # raytrace through phantom
    sino_stack = cp.zeros([N_thetas, N_channels_upx, 2], dtype=cp.float32)
    t0 = time()
    for i in range(2): 
        line_integrals = siddons_2D(src_x, src_y, trg_x, trg_y, matrix_stack[i], phantom.dx)
        sino_stack[:,:,i] = line_integrals.reshape([N_thetas, N_channels_upx])
        print(f'raytracing done for {i+1} / {len(matrix_stack)}, t={time() - t0:.2f}s')
    d_sino_delta, d_sino_beta = sino_stack[:,:,0], sino_stack[:,:,1] 
    
    # convert line integrals to phase contrast
    d_sino_obj = project_db(d_sino_delta, d_sino_beta, ct.SDD, wave.wavenum)  # upsampled object exit wave
    d_sino = cp.zeros([N_thetas, ct.N_channels], dtype=cp.float32)  # init the detected sino
    d_channels_0 = cp.array(ct.channels, dtype=cp.float32)  # target channels for downsampling
    d_channels_upx = cp.array(channels_upx, dtype=cp.float32)
    for i_view in range(N_thetas):
        row = d_sino_obj[i_view]
        d_fsp_row = propagate_1D(prop_dist, row, ct.beam_width, wave.wavelen)
        d_sino[i_view] = cp.interp(d_channels_0, d_channels_upx, cp.abs(d_fsp_row))  # downsample by 1-D linterp
        
    #return d_sino_delta, d_sino_beta, d_sino_obj, d_sino
    return d_sino
 
 
def multislice_xpci(ct, phantom, wave, prop_dist, upx, N_slices=0, thetas=None):  # upx = upsample factor

    # First check if N_slices is 0 -- projection approx. case
    if N_slices == 0:
        return project_xpci(ct, phantom, wave, prop_dist, upx, thetas)
    
    # Otherwise, use the multislice approach.
    # init phantom
    d_phantom_delta, d_phantom_beta = phantom.delta_beta_slice(wave.energy, 0)
    matrix_stack = cp.array([d_phantom_delta, d_phantom_beta], dtype=cp.float32) 
    
    # Init line integral coord's for the first slice
    if thetas is None:
        thetas = ct.thetas
    N_thetas = len(thetas)
    s_upx = ct.s / upx
    N_channels_upx = ct.N_channels * upx
    channels_upx = np.arange(-ct.beam_width/2, ct.beam_width/2, s_upx) + s_upx/2
    if len(channels_upx) > N_channels_upx:  # make sure size is correct
        channels_upx = channels_upx[:N_channels_upx]
    d_channels = cp.tile(cp.array(channels_upx, dtype=cp.float32), N_thetas)
    d_thetas = cp.tile(cp.array(thetas + cp.pi, dtype=cp.float32)[:, cp.newaxis], N_channels_upx).ravel()  
    src_x = ct.SID * cp.cos(d_thetas) + d_channels * cp.cos(d_thetas + cp.pi/2)
    src_y = ct.SID * cp.sin(d_thetas) + d_channels * cp.sin(d_thetas + cp.pi/2)

    # iteratively raytrace through phantom slices + free space propagate to next slice
    slice_width = ct.SDD / N_slices
    d_sino_upx = cp.ones([N_channels_upx, N_thetas], dtype=cp.float32)    
    t0 = time()
    for i_slice in range(N_slices):
        trg_x = ((i_slice+1)*slice_width - ct.SID) * cp.cos(d_thetas + cp.pi) + d_channels * cp.cos(d_thetas + cp.pi/2)
        trg_y = ((i_slice+1)*slice_width - ct.SID) * cp.sin(d_thetas + cp.pi) + d_channels * cp.sin(d_thetas + cp.pi/2)
        sino_stack = cp.zeros([N_thetas, N_channels_upx, 2], dtype=cp.float32)
        for i in range(2): 
            line_integrals = siddons_2D(src_x, src_y, trg_x, trg_y, matrix_stack[i], phantom.dx)
            sino_stack[:,:,i] = line_integrals.reshape([N_thetas, N_channels_upx])
        d_sino_delta, d_sino_beta = sino_stack[:,:,0], sino_stack[:,:,1] 
        d_sino_upx *= cp.exp(-wave.wavenum * (1j*d_sino_delta + d_sino_beta))        
        for i_view in range(N_thetas):  
            d_sino_upx[i_view] = propagate_1D(prop_dist, d_sino_upx[i_view], ct.beam_width, wave.wavelen)
        src_x = trg_x  # update source coordinates for next run
        src_y = trg_y
        print(f'multislice {i_slice+1}/{N_slices} done - {time() - t0:.2f}s')

    # finally, downsample 
    d_sino = cp.zeros([ct.N_channels, N_thetas], dtype=cp.float32)    
    d_channels_0 = cp.array(ct.channels, dtype=cp.float32)  # target channels for downsampling
    d_channels_upx = cp.array(channels_upx, dtype=cp.float32)
    for i_view in range(N_thetas):
        d_sino[i_view] = cp.interp(d_channels_0, d_channels_upx, cp.abs(d_sino_upx[i_view]))  # downsample by 1-D linterp

    return d_sino


def do_recon_patch(sino, ct, N_matrix, FOV, ramp, N_matrix_patch=512):
    r_matrix, theta_matrix = get_recon_coords(N_matrix, FOV)
    sino_filtered = pre_process(sino, ct, ramp)
        
    N_patches = np.ceil(N_matrix / N_matrix_patch).astype(int)
    if (N_matrix % N_matrix_patch) > 0:
        print('Changing N_matrix to be a multiple of {N_matrix_patch}!')
        N_matrix = N_patches * N_matrix_patch
        
    recon = np.zeros([N_matrix, N_matrix], dtype=np.float32)
    for i_patch in range(N_patches):
        i0 = N_matrix_patch * i_patch
        for j_patch in range(N_patches):
            print(f'Reconstructing patch {i_patch*N_patches + j_patch + 1} / {N_patches ** 2}')
            j0 = N_matrix_patch * j_patch
            r_matrix_patch = r_matrix[i0:i0+N_matrix_patch, j0:j0+N_matrix_patch] 
            theta_matrix_patch = theta_matrix[i0:i0+N_matrix_patch, j0:j0+N_matrix_patch] 
            
            patch = do_fbp(sino_filtered, r_matrix_patch, theta_matrix_patch, ct.SID, ct.s, ct.dtheta)
            recon[i0:i0+N_matrix_patch, j0:j0+N_matrix_patch] = patch
    
    recon = recon.clip(0, None)  # enfore non-negativity
    return recon


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

   #  #%% show phantom
    show_phantom = True
    pfov = phantom.Nx * phantom.dx / 2 * 1e3
    kw = {'cmap':'bone', 'extent':(-pfov, pfov, -pfov, pfov)}
    if show_phantom:
        fig, ax = plt.subplots(1, 2, figsize=[7.5,3])
        m = ax[0].imshow(d_phantom_delta.get(), **kw)
        fig.colorbar(m, ax=ax[0])
        ax[0].set_title('delta')
        m = ax[1].imshow(d_phantom_beta.get(), **kw)
        fig.colorbar(m, ax=ax[1])
        ax[1].set_title('beta')
        for axi in ax:  # this does not run?
            axi.set_xlabel('x [micron]')
            axi.set_ylabel('y [micron]')
        fig.tight_layout()
        plt.show()
    
#%%

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
            #%%
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
    show_recon = False
    if show_recon:
        N_matrix, FOV, ramp = 1024, 5e-3, 1.0
        
        recon_delta = do_recon_patch(sino_delta, ct, N_matrix, FOV, ramp)
        recon_beta = do_recon_patch(sino_beta, ct, N_matrix, FOV, ramp)

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

#%%  some tests with theta batching (for high upsampling memory management)

    t0 = time()
    upx = 8
    N_slices = 0  # 0 slices ==> multislice should just return project_xpci

    #d_sino = project_xpci(ct, phantom, wave, prop_dist=100e-3, upx=upx, thetas=None)
    #d_sino = multislice_xpci(ct, phantom, wave, prop_dist=100e-3, upx=upx, thetas=None, N_slices=N_slices)

    # Batching thetas, make sure ct.N_proj is an integer multiple of N_proj_batch
    N_proj_batch = 1000
    N_batches = int(ct.N_proj / N_proj_batch)
    sino_list = []
    for i in range(N_batches):
        print(f'Starting theta batch {i+1} / {N_batches}')
        i0 = i * N_proj_batch
        thetas_batch = ct.thetas[i0:i0+N_proj_batch]
        #d_sino_batch = project_xpci(ct, phantom, wave, prop_dist=100e-3, upx=upx, thetas=thetas_batch)
        d_sino_batch = multislice_xpci(ct, phantom, wave, prop_dist=100e-3, upx=upx, thetas=None, N_slices=N_slices)
        sino_list.append(d_sino_batch)
    d_sino = cp.array(sino_list).reshape([ct.N_proj, ct.N_channels])  # combine theta batches into one sinogram
    print(f'upsample x{upx} - {time() - t0:.2f} s / {(time() - t0)/60:.1f} min')

    kw = {'cmap':'bone', 'vmin':0, 'vmax':12}
    fig, ax = plt.subplots(1, 1, figsize=[4,3])
    ax.set_title(upx)
    m = ax.imshow(-cp.log(d_sino).get(), **kw)
    fig.colorbar(m)
    fig.tight_layout()
    plt.show()
    
    #%%  Reconstruct
    N_matrix, FOV, ramp = 1024, 5e-3, 1.0
    recon = do_recon_patch(d_sino.get(), ct, N_matrix, FOV, ramp)
    
    fig, ax = plt.subplots(1, 1, figsize=[4,3])
    m = ax.imshow(recon, cmap='bone')
    fig.colorbar(m)
    fig.tight_layout()
    plt.show()

    # fig, ax = plt.subplots(1, 2, figsize=[6,3])
    # m = ax[0].imshow(-cp.log(cp.abs(d_sino_obj)).get(), **kw)          
    # fig.colorbar(m, ax=ax[0])
    # m = ax[1].imshow(-cp.log(d_sino).get(), **kw)
    # fig.colorbar(m, ax=ax[1])
    # # ax[0].imshow(cp.abs(d_sino_obj).get(), **kw)
    # # ax[1].imshow(d_sino.get(), **kw)
    # fig.tight_layout()
    # plt.show()
  

    #%%
    
    # upx = 1
    # s_upx = ct.s / upx
    # channels_upx = np.arange(-ct.beam_width/2, ct.beam_width/2, s_upx)[:-1] + s_upx/2
    # d_channels_upx = cp.tile(cp.array(channels_upx, dtype=cp.float32), ct.N_proj)
    # #d_thetas = cp.tile(cp.array(ct.thetas + cp.pi, dtype=cp.float32)[:, cp.newaxis], d_channels.size).ravel()  
    
    # #d_thetas = cp.tile(cp.array(ct.thetas + cp.pi, dtype=cp.float32)[:, cp.newaxis], ct.N_channels).ravel()  # use newaxis for correct tiling
    # d_channels = cp.tile(cp.array(ct.channels, dtype=cp.float32), ct.N_proj)


# #%%
#     upx = 1
#     prop_dist = 100e-3
    
#     # init phantom
#     d_phantom_delta, d_phantom_beta = phantom.delta_beta_slice(wave.energy, 0)
#     matrix_stack = cp.array([d_phantom_delta, d_phantom_beta], dtype=cp.float32) 
    
#     # init line integral coord's
#     s_upx = ct.s / upx
#     N_channels_upx = ct.N_channels * upx
#     channels_upx = np.arange(-ct.beam_width/2, ct.beam_width/2, s_upx) + s_upx/2
#     if len(channels_upx) > N_channels_upx:
#         channels_upx = channels_upx[:N_channels_upx]
#     d_channels = cp.tile(cp.array(channels_upx, dtype=cp.float32), ct.N_proj)
#     d_thetas = cp.tile(cp.array(ct.thetas + cp.pi, dtype=cp.float32)[:, cp.newaxis], N_channels_upx).ravel()  
#     src_x = ct.SID * cp.cos(d_thetas) + d_channels * cp.cos(d_thetas + cp.pi/2)
#     src_y = ct.SID * cp.sin(d_thetas) + d_channels * cp.sin(d_thetas + cp.pi/2)
#     trg_x = (ct.SDD - ct.SID) * cp.cos(d_thetas + cp.pi) + d_channels * cp.cos(d_thetas + cp.pi/2)
#     trg_y = (ct.SDD - ct.SID) * cp.sin(d_thetas + cp.pi) + d_channels * cp.sin(d_thetas + cp.pi/2)
    
#     # raytrace through phantom
#     sino_stack = cp.zeros([ct.N_proj, N_channels_upx, 2], dtype=cp.float32)
#     t0 = time()
#     for i in range(2): 
#         line_integrals = siddons_2D(src_x, src_y, trg_x, trg_y, matrix_stack[i], phantom.dx)
#         sino_stack[:,:,i] = line_integrals.reshape([ct.N_proj, ct.N_channels])
#         print(f'raytracing done for {i+1} / {len(matrix_stack)}, t={time() - t0:.2f}s')
#     d_sino_delta, d_sino_beta = sino_stack[:,:,0], sino_stack[:,:,1] 
    
#     # convert line integrals to phase contrast
#     d_sino_obj = project_db(d_sino_delta, d_sino_beta, ct.SDD, wave.wavenum)  # upsampled object exit wave
#     d_sino = cp.zeros([ct.N_proj, ct.N_channels], dtype=cp.float32)  # init the detected sino

#     d_channels_0 = cp.array(ct.channels, dtype=cp.float32)  # target channels for downsampling
#     d_channels_upx = cp.array(channels_upx, dtype=cp.float32)
#     for i_view in range(ct.N_proj):
#         row = d_sino_obj[i_view]
#         d_fsp_row = propagate_1D(prop_dist, row, ct.beam_width, wave.wavelen)
#         d_sino[i_view] = cp.interp(d_channels_0, d_channels_upx, cp.abs(d_fsp_row))  # downsample by 1-D linterp
        
    #return d_sino_delta, d_sino_beta, d_sino_obj, d_sino
     
     
# #%%

# kw = {'cmap':'bone', 'vmin':0, 'vmax':6}
# fig, ax = plt.subplots(1, 2, figsize=[6,3])
# m = ax[0].imshow(-cp.log(cp.abs(d_sino_obj)).get(), **kw)          
# fig.colorbar(m, ax=ax[0])
# m = ax[1].imshow(-cp.log(d_sino).get(), **kw)
# fig.colorbar(m, ax=ax[1])
# # ax[0].imshow(cp.abs(d_sino_obj).get(), **kw)
# # ax[1].imshow(d_sino.get(), **kw)
# fig.tight_layout()
# plt.show()

# #%%
# kw = {'cmap':'bone', 'vmin':0, 'vmax':0.5}
# fig, ax = plt.subplots(1, 1, figsize=[6,3])
# m = ax.imshow(
#     np.abs(cp.log(cp.abs(d_sino_obj)).get() - cp.log(d_sino).get()),
#            **kw)          
# fig.colorbar(m)
# fig.tight_layout()
# plt.show()


# #%%
# plt.imshow(d_sino.get()[:, 250:3250])
# plt.show()


# #%%
# #phantom_delta = d_phantom_delta.get()
# x0, y0, Nx = 98, 70, 1550


# voxels_clipped = phantom.voxels[:, y0:y0+Nx, x0:x0+Nx]
# #plt.imshow(phantom_delta[y0:y0+Nx, x0:x0+Nx])
# plt.imshow(voxels_clipped[0])
# plt.axis('off')
# plt.show()

# phantom_filepath = "./input/phantom/zebrafish_2D/"
# phantom_filename = f"zebrafish_{Nx}_{Nx}_1.43um_4mat_uint8.bin"
# voxels_clipped.astype(np.uint8).tofile(phantom_filepath + phantom_filename)

    
    
    
    
    
    
    
    
