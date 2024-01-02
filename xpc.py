#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct  4 09:52:16 2023

@author: giavanna
"""

import matplotlib.pyplot as plt
from time import time
from datetime import timedelta
from helpers import read_parameter_file
from forward import multislice_wave_2cylinders, multislice_wave_voxels, \
                    free_space_propagate, detect_wave
import numpy as np
import os
import shutil

def measure_divergence(im, im_ref):
    assert im.size == im_ref.size
    N = im.size
    D = np.sqrt(np.sum((im - im_ref)**2) / N)
    return D

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

#%%
    
if __name__ == '__main__':

    #param_file = 'params_spheretest.txt'
    param_file = 'params_voxel.txt'
    parameter_sets = read_parameter_file(param_file)
    save_output = True
    
    
    all_imgs = []
    all_slices = []
    t0 = time()
    for i, parameters in enumerate(parameter_sets):
        
        # 1: SIMULATE WAVE PROPAGATION
        run_id, propagation_distance, N_slices, upsample_multiple, wave, phantom = parameters
        try:  #  cylinder phantom
            object1, object2 = phantom
            d_exit_wave = multislice_wave_2cylinders(wave, object1, object2, N_slices)
        except:  # voxel phantom
            d_exit_wave = multislice_wave_voxels(wave, phantom, N_slices)
        d_fsp_wave = free_space_propagate(wave, d_exit_wave, propagation_distance)
        d_detected_wave = detect_wave(d_fsp_wave, upsample_multiple)
        detected_wave = d_detected_wave.get()
        
        # 2: WRITE OUTPUT
        if save_output:
            main_dir = f'./output/{run_id}'
            run_dir = f'{main_dir}/run_{i}'
            os.makedirs(run_dir, exist_ok=True)
            shutil.copy(param_file, f'{main_dir}/all_parameters.txt')
    
            paramlist = f'{run_id}\n\n'
            paramlist += f'multislices: {N_slices}\n'
            paramlist += f'upsample multiple: {upsample_multiple}\n'
            paramlist += f'propagation distance [mm]: {propagation_distance*1e3:.1f}\n'
            paramlist += f'grid voxel size [um]: {wave.dx*upsample_multiple*1e6:.2f}\n'
            paramlist += f'grid size: {int(wave.Nx/upsample_multiple)}\n'
            paramlist += f'FOV [um]: {wave.dx*wave.Nx}\n'
            with open(f'{run_dir}/run_parameters.txt', 'w') as f:
                f.write(paramlist)
            detected_wave.astype(np.float32).tofile(f'{run_dir}/img_float32.bin')
                
        # 3: VISUALIZE OUTPUT
        all_imgs.append(detected_wave)
        all_slices.append(N_slices)
        
        # A --- ZEBRAFISH
        c = 1
        sub_img = detected_wave[c*2000:c*2300, c*1800:c*2100]
        #for img in [detected_wave, sub_img]:
        for img in [sub_img]:
            #kw = {'cmap':'gray'}
            kw = {'cmap':'gray', 'vmin':.5, 'vmax':1.2} 
            fig, ax = plt.subplots(1, 1, dpi=600)
            m = ax.imshow(img, **kw)
            ax.axis('off')
            
            # ax.set_title(f'{N_slices} slices, x{upsample_multiple}, {wave.FOVx*1e3:.3f}-mm FOV, {object2.length*1e3}-mm thick')
            ax.set_title(f'{phantom.name} ({wave.FOVx*1e3:.3f}-mm FOV, {phantom.dz*1e3}-mm thick)', fontsize=8)
            paramlist = ''
            paramlist += f'multislices: {N_slices}\n'
            paramlist += f'upsample: x{upsample_multiple}\n'
            paramlist += f'prop_dist: {propagation_distance*1e3:.1f} mm\n'
            paramlist += f'voxel_sz: {wave.dx*1e6:.2f} um'
            t = ax.text(0.02*img.shape[0], 0.02*img.shape[1], paramlist, horizontalalignment='left', verticalalignment='top', fontsize=6)
            t.set_bbox(dict(facecolor='white', alpha=0.9, edgecolor='red'))

            fig.colorbar(m, ax=ax)
            fig.tight_layout()
            plt.show()
        
        # B --- SPHERE TEST            
        # invert = False
        # if invert:
        #     import numpy as np
        #     kw = {'cmap':'gray'}
        #     t_img = -np.log(detected_wave)
        # else:
        #     kw = {'cmap':'gray', 'vmin':0.856, 'vmax':0.860} 
        #     t_img = detected_wave

        # dm = 1e6*wave.FOVy/(2*upsample_multiple)
        # for img in [t_img]:
        #     fig, ax = plt.subplots(1, 1, dpi=600)
        #     m = ax.imshow(img, extent=(-dm, dm, -dm, dm), **kw)
        #     ax.set_xlabel('um')
        #     ax.set_ylabel('um')
        #     #ax.axis('off')
            
        #     # ax.set_title(f'{N_slices} slices, x{upsample_multiple}, {wave.FOVx*1e3:.3f}-mm FOV, {object2.length*1e3}-mm thick')
        #     ax.set_title(f'{object2.name}', fontsize=8)
        #     paramlist = ''
        #     paramlist += f'multislices: {N_slices}\n'
        #     paramlist += f'upsample: x{upsample_multiple}\n'
        #     paramlist += f'prop_dist: {propagation_distance*1e3:.1f} mm'
        #     t = ax.text(-.98*dm, .98*dm, paramlist, horizontalalignment='left', verticalalignment='top', fontsize=6)
        #     t.set_bbox(dict(facecolor='white', alpha=0.9, edgecolor='red'))

        #     fig.colorbar(m, ax=ax)
        #     fig.tight_layout()
        #     plt.show()
            
        #print(f'x{upsample_multiple} - {wave.Nx} - {timedelta(seconds=int(time()-t0))}')
        print(f'multislices: {N_slices} - {timedelta(seconds=int(time()-t0))}')
    #     ax.plot(detected_wave[detected_wave.shape[0]//2])
    # ax.set_xlim(70,120)
    # plt.show()
    
    
    #%%
    
    all_divergence = [measure_divergence(im, all_imgs[-1]) for im in all_imgs[:-1]]
    fig, ax = plt.subplots(dpi=600)
    ax.plot(all_slices[:-1], all_divergence)
    ax.set_ylabel('divergence')
    ax.set_yscale('log')
    ax.set_xlabel('number of slices')
    plt.show()
    
    
    #%%
    
    
    fig, ax = plt.subplots(dpi=600)
    # line profiles
    labels = ['projection', 'proj2', f'multislice (N = {all_slices[-1]})']
    col = 'kkk'
    lss = ['--', ':', '-']
    for i in [-1, 1, 0]:
        img_i = all_imgs[i]
        # img_i = -np.log(img_i)  # for log intensity (radiograph)
        line_i = img_i[img_i.shape[0]//2]
        ax.plot(line_i, label=i)# col[i], ls=lss[i], label=labels[i])
        #ax.plot(line_i, color=col[i], ls=lss[i], label=labels[i])
    ax.legend()
    ax.set_ylabel('line profile')
    #ax.set_xlabel('line profile')
    plt.show()
    
# #%%

# import cupy as cp
# from helpers import PI

# d_wave_ft = cp.fft.fft2(d_exit_wave)
# prop_dist = 2e-1
# fx, fy = wave.fftfreqs()  # centered

# d_H = cp.exp(cp.asarray(-1j * PI * wave.wavelen * prop_dist * (fx**2 + fy**2)))  # H: FT-space Fresnel operator
# import matplotlib.pyplot as plt
# fig,ax=plt.subplots(1,2, dpi=300, figsize=[8,3])
# fig.suptitle(prop_dist)

# for i, img in enumerate([d_H.get().real, d_H.get().imag]):
#     m=ax[i].imshow(img)
#     fig.colorbar(m,ax=ax[i])
#     ax[i].axis('off')
# fig.tight_layout()
# plt.show()

# pad = int((fx.shape[0] - wave.Nx)//2)
# print(pad)
# d_wave_ft = cp.fft.fft2(cp.pad(d_exit_wave, [(pad, pad), (pad, pad)]))


# # show padded wave
# # plt.imshow(cp.pad(d_exit_wave, [(pad, pad), (pad, pad)]).get().imag)
# # plt.colorbar()
# # plt.show()



# #%%

# import numpy as np
# #%%
# for prop_dist in [2e-1]:
#     upsample_multiple = 2
#     wave.dx = wave.dx/upsample_multiple
#     wave.dy = wave.dy/upsample_multiple
#     wave.Nx = wave.Nx*upsample_multiple
#     wave.Ny = wave.Ny*upsample_multiple
    
#     fx, fy = wave.fftfreqs()  # centered

#     test = -1j * PI * wave.wavelen * prop_dist * (fx.get()**2 + fy.get()**2)
#     H = np.exp(test)
#     test = np.exp(test).real
    
#     fig,ax=plt.subplots(1,1,dpi=300)
#     plt.imshow(test)
#     plt.colorbar()
#     plt.title(prop_dist)
#     plt.show()




