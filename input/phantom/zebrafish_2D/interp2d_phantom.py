#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 11 15:58:53 2023

@author: giavanna

for interpolating phantoms
"""

from scipy.interpolate import RegularGridInterpolator
import numpy as np
import matplotlib.pyplot as plt
import os
from time import time

# # the cupy function might be a bit faster, but not too much
# from cupyx.scipy.interpolate import RegularGridInterpolator


wd = os.path.dirname(__file__)+'/'  # './'


# target dimensions for the re-binned data
wave_Nx = 10000  #5000
wave_Ny = 10000  #5000
wave_dx = 0.25e-6  #0.5e-6
wave_dy = 0.25e-6  #0.5e-6
wave_FOVx, wave_FOVy = wave_Nx*wave_dx, wave_Ny*wave_dy
xvals = np.linspace(-wave_FOVx/2, wave_FOVx/2, wave_Nx) + wave_dx/2 
yvals = np.linspace(-wave_FOVy/2, wave_FOVy/2, wave_Ny) + wave_dy/2 
grid_X, grid_Y = np.meshgrid(xvals, yvals)
grid_coords_xy = list(zip(grid_X.ravel(), grid_Y.ravel()))  # target coords, ravel'd
id_background = 0
outfile = f'zebrafish_interp2d_{wave_Nx}_{wave_Ny}_{wave_dx*1e6}um_4mat_uint8.bin'


# phantom input
fname = 'zebrafish_1685_1685_1.43um_4mat_uint8.bin'
phantom_Nx = 1685
phantom_Ny = 1685
phantom_dx = 1.43e-6
phantom_dy = 1.43e-6
phantom_FOVx, phantom_FOVy = phantom_Nx*phantom_dx, phantom_Ny*phantom_dy
phantom_x = np.linspace(-phantom_FOVx/2, phantom_FOVx/2, phantom_Nx) + phantom_dx/2 
phantom_y = np.linspace(-phantom_FOVy/2, phantom_FOVy/2, phantom_Ny) + phantom_dy/2 
voxels = np.fromfile(wd+fname, dtype=np.uint8).reshape([phantom_Ny, phantom_Nx])


# interpolate phantom onto the target coordinates 
# use nearest neighbor to keep material IDs
t0 = time()
interp = RegularGridInterpolator((phantom_x, phantom_y), voxels, 
                                  method='nearest', fill_value=id_background, 
                                  bounds_error=False)
print(f'{time()-t0:.2f} s')

t0 = time()
output_ravel = interp(grid_coords_xy)
output = output_ravel.reshape([wave_Ny, wave_Nx]).T
print(f'{time()-t0:.2f} s')  # could take ~ 1 min depending on size


# show output
fig,ax = plt.subplots(1, 2, figsize=[8,3], dpi=600)
for i, img in enumerate([voxels, output]):
    m = ax[i].imshow(img)
    fig.colorbar(m, ax=ax[i])
fig.tight_layout()
plt.show()


# Store the interpolated phantom (for now).
# Ideally, would resample phantoms in each new simulation, but this seems
# to be inefficient. Maybe in future could implement a cache system?
output.astype(np.uint8).tofile(wd+outfile)
    
