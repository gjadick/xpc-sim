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


#%% PLOTs : VITM 24, sphere
from scipy.signal import convolve2d
from cupyx.scipy.signal import convolve2d as d_convolve2d
import cupy as cp

plt.rcParams.update({
    "font.size":10,
    "axes.titlesize":10,
    "axes.labelsize": 8,
    "axes.linewidth": .5,
    "xtick.labelsize":6,
    "ytick.labelsize":6,
    "ytick.major.size":2,
    "xtick.major.size":2,
})
    

def measure_divergence(im, im_ref):
    assert im.size == im_ref.size
    N = im.size
    D = np.sqrt(np.sum((im - im_ref)**2) / N)
    return D


def blur2D_gpu(im, xc, fwhm):
    gamma = fwhm/2
    d_xc = cp.array(xc)
    d_im = cp.array(im)
    X, Y = cp.meshgrid(d_xc, d_xc)
    lorentzian2d = 1 / (X**2 + Y**2 + gamma**2)**1.5
    lorentzian2d = lorentzian2d / cp.sum(lorentzian2d) 
    #plt.show()
    return d_convolve2d(d_im, lorentzian2d, mode='same').get().astype(np.float32)


def blur2D(im, x, y, fwhm):
    gamma = fwhm/2
    X, Y = np.meshgrid(x, y)
    lorentzian2d = 1 / (X**2 + Y**2 + gamma**2)**1.5
    lorentzian2d = lorentzian2d / np.sum(lorentzian2d) 
    #plt.show()
    return convolve2d(im, lorentzian2d, mode='same')


figd = './output/vitm24/figs/'
savefig = True



#%%  vitm fig 2 - spheres


propdists = np.array([0, 50, 100])  # mm
pxszs = np.array([50, 500])  # nm
maxslices = 50

fwhm = 1  # um
blur = True


fig, AX = plt.subplots(2, 4, figsize=[8.5,3.4])#, width_ratios=[1, 1, 1, 1.2])

for j, pxsz in enumerate(pxszs):
    
    ax = AX[j]
    
    for i, R in enumerate(propdists):
        if j==0:
            ax[i].set_title(f'{R:.0f} mm')
            ax[-1].set_title('Divergence')
        else:
            ax[i].set_xlabel('$\\mu$m')
            ax[-1].set_xlabel('number of slices')
        #ax[0].set_ylabel('intensity')
        #ax[-1].set_ylabel('divergence')
        ax[-1].ticklabel_format(style='sci', axis='y', scilimits=(0, 0))
        ax[0].text(-14, 0.8564, 'Intensity', rotation=90, fontsize=8)

        indir = f'output/vitm24/sphereTest_multislice_px{pxsz:03}nm_prop{R:03}mm/'
        ims = []
        for k in range(maxslices):
            im = np.fromfile(indir+f'run_{k}/img_float32.bin', dtype=np.float32)
            N = int(np.sqrt(len(im)))
            im = im.reshape([N,N])
            FOV = N * pxsz / 1e3  # um
            xcoord = (np.arange(-(N-1)/2, N/2) ) * pxsz / 1e3
            if blur:
                Nc = int(500/pxsz * 5)
                xc_crop = xcoord[N//2-Nc:N//2+Nc]
                #im = blur2D(im, xc_crop, xc_crop, fwhm=fwhm)  # all units in microns
                im = blur2D_gpu(im, xc_crop, fwhm=fwhm)  # all units in microns
            ims.append(im)
            
        dx = [0.11, 0.25][j]  # center the line profile
        ax[i].plot(xcoord-dx, ims[0][N//2], 'k-', label='projection')
        ax[i].plot(xcoord-dx, ims[-1][N//2], 'rgb'[i], label='multislice')
        ax[i].legend(loc=(0.01, 0.04), fontsize=6, frameon=False)
        ax[i].set_yticks(np.arange(0.854, 0.862, 0.002)) 
        ax[i].set_xticks(np.arange(-8,9,4))
        ax[i].set_ylim(0.8537, 0.8623)
        ax[i].set_xlim(-9, 9)      
        
    divs = [measure_divergence(im, ims[-1]) for im in ims]
    ax[-1].plot(range(1,maxslices), divs[1:], 'k')
    ax[-1].set_xlim(-1,52)
fig.tight_layout(w_pad=0.7, h_pad=0)
if savefig:
    plt.savefig(figd+f'spheres_blur{blur}.pdf', bbox_inches='tight')
plt.show()


#%% VITM fig 3 - zebrafish

from matplotlib import patches

slices =[1, 2, 4, 8, 16, 32, 64]
i_slices = [0,1,2,4]
pxsz = 500e-9
indir = 'output/vitm24/zebrafish2D_multislice_px500nm_prop050mm/'
N = 5000
blur = True
fwhm = 1  # um

# load images
ims = []
for k in i_slices:
    im = np.fromfile(indir+f'run_{k}/img_float32.bin', dtype=np.float32).reshape([N,N])
    if blur:
        xcoord = (np.arange(-(N-1)/2, N/2) ) * pxsz * 1e6  # units of micron
        Nc = 27
        xc_crop = xcoord[(N-Nc)//2:(N+Nc)//2] + (pxsz * 0.5e6)
        im = blur2D_gpu(im, xc_crop, fwhm=fwhm)  # all units in microns
    ims.append(im)

cbar_pad = 0.02
kw = {'cmap':'bwr', 'vmin':-0.089, 'vmax':0.089}
kw0 = {'cmap':'gray', 'vmin':0.01, 'vmax':1.19}
x0, y0 = 1850, 1850  # ROI
dx, dy = 300, 300

fig, AX = plt.subplots(2, 4, figsize=[11.5,5])#, width_ratios=[1, 1, 1, 1.2])
ax, ax2 = AX[0], AX[1]
for axi in AX.ravel():
    axi.set_xticks([])
    axi.set_yticks([])

ax[0].set_title('Projection approximation')    
m = ax[0].imshow(ims[0], **kw0)
fig.colorbar(m, ax=ax[0], pad=cbar_pad)

m = ax2[0].imshow(ims[0][y0:y0+dy, x0:x0+dx], **kw0)
fig.colorbar(m, ax=ax2[0], pad=cbar_pad)

for i in range(1,4):
    
    ax[i].set_title(f'Difference $N$ = {slices[i_slices[i]]}')
    diff = ims[i] - ims[0] 
    m = ax[i].imshow(diff, **kw)
    fig.colorbar(m, ax=ax[i], pad=cbar_pad)
    rect = patches.Rectangle((x0, y0), dx, dy, linewidth=1, edgecolor='r', facecolor='none')
    ax[i].add_patch(rect)
    
    m = ax2[i].imshow(diff[y0:y0+dy, x0:x0+dx], **kw)
    fig.colorbar(m, ax=ax2[i], pad=cbar_pad)

fig.tight_layout(w_pad=0, h_pad=0.7)
if savefig:
    plt.savefig(figd+f'zebrafish_blur{blur}.pdf', bbox_inches='tight')
plt.show()





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







