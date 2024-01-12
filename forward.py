#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct  3 11:57:52 2023

@author: giavanna

All functions return objects stored in GPU memory.
"""

from helpers import PI, block_mean_2d, cp_array
import cupy as cp
from cupyx.scipy.signal import convolve2d

import matplotlib.pyplot as plt
import numpy as np


def cimshow(d_cimg, fmt='%.3f', kw={'vmin':-1.1,'vmax':1.1, 'cmap':'bwr'}): # complex imshow
    cimg = d_cimg.get()
    fig, ax = plt.subplots(1,2,figsize=[8,3.5], dpi=300)
    ims = [cimg.real, cimg.imag, np.abs(cimg)]
    titles = ['Real', 'Imag', 'Abs']
    for i in range(len(ax)):
        m = ax[i].imshow(ims[i], **kw)
        ax[i].set_title(titles[i])
        fig.colorbar(m,ax=ax[i], format=fmt)
        ax[i].axis('off')
    fig.tight_layout()
    plt.show()
    
    
def project_wave():
    pass


def get_H(prop_dist, in_wave):
    """
    Get the Fourier-space Fresnel operator.

    This uses the approximated operator presented in Goodman eq. 4-21.
    The more exact eq. 4-20, seems to require higher precision (more than single)
    to match the approximation, due to the square root operation (in my tests).
    Also note that Li et al. "Multislice..." (2017) reference eq. 4-20 in their
    eq. 3, but they have a typo (an unneeded minus sign).
    """
    fx, fy = in_wave.fftfreqs()  # centered frequencies
    # d_H = np.exp(1j * prop_dist *in_wave.wavenum * np.sqrt(1 - (in_wave.wavelen**2)*(fx**2 + fy**2)))  # Goodman eq. 4-20
    d_H = cp.exp(-1j * in_wave.wavelen * PI * prop_dist * cp_array(fx**2 + fy**2))  # Goodman eq. 4-21
    d_H = cp.fft.fftshift(d_H)  # non-centered frequencies
    return d_H 


def get_Fresnel_number(prop_dist, in_wave):
    """Assume smallest resolvable structure = 1 detector pixel"""
    upsample_multiple = 4  # !!! TODO fix
    detector_pixel_sz = min(in_wave.dx, in_wave.dy) * upsample_multiple
    N_F = detector_pixel_sz**2 / (in_wave.wavelen * prop_dist)
    print(f'{N_F > 10} N_F >> 1    (N_F = {N_F:.1f})')
    return N_F


def project_wave_voxels(in_wave, phantom, thickness=None): # !!! TODO
    # right now this only works for a 2D phantom
    
    d_delta_slice, d_beta_slice = phantom.delta_beta_slice(in_wave.energy, z_ind=0)
    if thickness is None:
        thickness = phantom.dz  # default to voxel thickness
    proj_delta, proj_beta = thickness*d_delta_slice, thickness*d_beta_slice
    d_exit_wave = in_wave.I0 * cp.exp(-in_wave.wavenum * (1j*proj_delta + proj_beta)) * cp.exp(1j * in_wave.wavenum * thickness)
    return d_exit_wave


def project_wave_2phantoms(in_wave, phantom1, phantom2):
    """
    For a monochromatic, coherent plane wave I0 along the optical axis z, 
    compute the modulation through object with complex index of refraction
    delta(x,y,z) + i*beta(x,y,z) under the projection approximation.


    Parameters
    ----------
    in_wave : Source
        DESCRIPTION.
    sphere1 : SpherePhantom
        DESCRIPTION.
    sphere2 : SpherePhantom
        DESCRIPTION.
    device : bool, optional
        Return the exit wave stored on GPU (F) or CPU (T). If there will
        be more GPU operations, it is more efficient to set this True to 
        minimize the memory transfers on/off the device.
        The default is False.

    Returns
    -------
    exit_wave : Wave !!!
        DESCRIPTION.
    """
    # Get phantom parameters
    d_t1 = phantom1.projected_thickness()
    d_t2 = phantom2.projected_thickness()
    
    delta1, beta1 = phantom1.delta_beta(in_wave.energy)
    delta2, beta2 = phantom2.delta_beta(in_wave.energy)
    
    # Project input wave through the object
    proj_delta = d_t1*delta1 + d_t2*delta2  
    proj_beta = d_t1*beta1 + d_t2*beta2
    d_exit_wave = in_wave.I0 * cp.exp(-in_wave.wavenum * (1j*proj_delta + proj_beta))
    return d_exit_wave


    
def multislice_wave_voxels(in_wave, phantom, N_slices): # !!! TODO
    # right now only works for 2D phantom
    # a lot of lines copied from projection function to eliminate redundant computations
    
    if N_slices == 1:
        return project_wave_voxels(in_wave, phantom)
    
    slice_width = phantom.dz / N_slices  
    d_delta_slice, d_beta_slice = phantom.delta_beta_slice(in_wave.energy, z_ind=0)
    proj_delta, proj_beta = slice_width*d_delta_slice, slice_width*d_beta_slice

    d_exit_wave = in_wave.I0
    for i in range(N_slices):
        d_exit_wave *= cp.exp(-in_wave.wavenum * (1j*proj_delta + proj_beta))        
        d_exit_wave = free_space_propagate(in_wave, d_exit_wave, slice_width)
        
    return d_exit_wave
    

def multislice_wave_2cylinders(in_wave, phantom1, phantom2, N_slices):
    
    if N_slices == 1:
        return project_wave_2phantoms(in_wave, phantom1, phantom2)
    
    # Get phantom parameters
    d_t1 = phantom1.projected_thickness() / N_slices
    d_t2 = phantom2.projected_thickness() / N_slices
    delta1, beta1 = phantom1.delta_beta(in_wave.energy)
    delta2, beta2 = phantom2.delta_beta(in_wave.energy)
    slice_width = cp.max(d_t1 + d_t2) / N_slices  

    proj_delta = d_t1*delta1 + d_t2*delta2  
    proj_beta = d_t1*beta1 + d_t2*beta2
    
    #get_Fresnel_number(slice_width, in_wave) # check Fresnel number, > 1?

    # Project + propagate input wave N_slices times
    d_exit_wave = in_wave.I0
    for i in range(N_slices):
        d_exit_wave *= cp.exp(-in_wave.wavenum * (1j*proj_delta + proj_beta))        
        d_exit_wave = free_space_propagate(in_wave, d_exit_wave, slice_width)
        
    return d_exit_wave

    
def free_space_propagate(in_wave, exit_wave, prop_dist):
    """
    Compute wave after propagating through free space.
    Applies the Fresnel propagator in the Fourier domain, which is best when 
    pixel size dx=dy is larger than the cutoff:
        pix_liml = wavelen*prop_dist/(dx*min(Ny, Nx))
        pix_limh = wavelen*prop_dist/(dx*max(Ny, Nx))
    in the case pix_liml=pix_limh since Ny=Nx

    Parameters
    ----------
    wave : TYPE
        DESCRIPTION.
    prop_dist : TYPE
        DESCRIPTION.

    Returns
    -------
    None.

    """
    d_wave_ft = cp.fft.fft2(exit_wave)
    d_H = get_H(prop_dist, in_wave)
    d_fsp_wave = cp.fft.ifft2(d_wave_ft * d_H) # * cp.exp(1j*in_wave.wavenum*prop_dist) 
    return d_fsp_wave


# discrete Laplacian
laplace2d = np.array([[0, 1, 0],
                      [1,-4, 1],
                      [0, 1, 0]])
d_laplace2d = cp.array(laplace2d)
def get_signal_TIE(in_wave, phantom1, phantom2, prop_dist):
    """
    Compute the detected signal using a forward model based on the
    linearized transport of intensity equation (TIE). 
    This assumes no downsampling.
    Shows essentially no phase contrast in my tests.
    """
    # Get phantom parameters
    d_t1 = phantom1.projected_thickness()
    d_t2 = phantom2.projected_thickness()
    
    delta1, beta1 = phantom1.delta_beta(in_wave.energy)
    delta2, beta2 = phantom2.delta_beta(in_wave.energy)
    
    # Project input wave through the object
    proj_delta = d_t1*delta1 + d_t2*delta2  
    proj_beta = d_t1*beta1 + d_t2*beta2
    
    # Compute sig, assume no upsampling
    d_signal = in_wave.I0 * cp.exp(-2 * in_wave.wavenum * proj_beta 
                      + prop_dist * convolve2d(proj_delta, d_laplace2d, mode='same'))    
    return d_signal


def lorentzian2D(x, y, fwhm, normalize=True):
    """Generate a 2D Lorentzian kernel."""
    gamma = fwhm/2
    X, Y = cp.meshgrid(x, y)
    kernel = gamma / (2 * PI * (X**2 + Y**2 + gamma**2)**1.5)
    if normalize:
        kernel = kernel / cp.sum(kernel)
    return kernel


def detect_wave(in_wave, wave, upsample_multiple, fwhm=10e-6, N_kernel=129):
    d_detected_wave = block_mean_2d(cp.abs(wave), upsample_multiple)
    #FOV_kern_x, FOV_kern_y = N_kernel*in_wave.dx, N_kernel*in_wave.dy
    #d_x = cp.linspace(-FOV_kern_x/2, FOV_kern_x/2, N_kernel)
    #d_y = cp.linspace(-FOV_kern_y/2, FOV_kern_y/2, N_kernel)
    #d_lorentzian2d = lorentzian2D(d_x, d_y, fwhm)
    #d_wave_blurred = convolve2d(cp.abs(wave), d_lorentzian2d, mode='same')
    #d_detected_wave = block_mean_2d(d_wave_blurred, upsample_multiple)
    return d_detected_wave
    



## 1D propagation for a CT simulation slice

def project_db(d_proj_delta, d_proj_beta, proj_thickness, wavenum, I0=1):
    """
    Computes projection through object for given delta, beta line integrals.

    Parameters
    ----------
    d_proj_delta : N-D cupy array
        DESCRIPTION.
    d_proj_beta : N-D cupy array
        DESCRIPTION.
    proj_thickness : float
        DESCRIPTION.
    wavenum : float
        DESCRIPTION.
    I0 : float OR N-D cupy array, optional
        DESCRIPTION. The default is 1.

    Returns
    -------
    N-D cupy array
        DESCRIPTION.

    """
    return I0 * cp.exp(-wavenum * (1j*d_proj_delta + d_proj_beta)) * cp.exp(1j * wavenum * proj_thickness)


def propagate_1D(prop_dist, d_wave, beam_width, wavelen): 

    # Compute 1D Fresnel propagator.
    Nx = d_wave.size
    d_fx = cp.arange(-Nx/2, Nx/2, 1)/beam_width + 1/(2*beam_width)  # centered frequencies of detector channels
    d_H = cp.fft.fftshift(cp.exp(-1j * wavelen * PI * prop_dist * d_fx**2))  # non-centered Fresnel operator
    #d_fx_grid, d_fy_grid = cp.meshgrid(d_fx, d_fx)
    #d_H_grid = cp.exp(-1j * wavelen * PI * prop_dist * (d_fx_grid**2 + d_fy_grid**2))  # non-centered Fresnel operator

    # Convolve wave in Fourier space.
    d_wave_ft = cp.fft.fft(d_wave)
    d_fsp_wave = cp.fft.ifft(d_wave_ft * d_H) # * cp.exp(1j*in_wave.wavenum*prop_dist) 
    return d_fsp_wave

 

 
 
 
 
 
 
 






