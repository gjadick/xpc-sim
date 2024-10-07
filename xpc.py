#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct  4 09:52:16 2023

@author: giavanna

Miscellaneous functions for XPCI simulations.
"""

from time import time
import cupy as cp
import numpy as np
from cupyx.scipy.signal import convolve2d
import matplotlib.pyplot as plt

from helpers import cp_array, block_mean_2d
from xtomosim.forward_project import siddons_2D
from xtomosim.back_project import pre_process, get_recon_coords, do_fbp


PI = np.pi


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



def cimshow(d_cimg, fmt='%.3f', kw={'vmin':-1.1,'vmax':1.1, 'cmap':'bwr'}): 
    """
    Convenience function for showing a complex image.
    Three panels are shown: the real and imaginary parts, and the magnitude.
    """
    try:
        cimg = d_cimg.get()  # is the image on GPU device?
    except:
        cimg = d_cimg
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


#####################################################################
#####################################################################
###
### 2D PROJECTION FUNCTIONS - planar imaging
###          ______
###         |      |                 y
###    z------->   |---->           |
###         |______|                |____x
###
### Wave propagates along the axial direction thru a 2D phantom
### defined in the x, y directions.
###

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


def project_wave_voxels(in_wave, phantom, thickness=None): 
    """
    Use the projection approximation to modulate a coherent, monochromatic 
    plane wave through a 2D voxelized phantom characterized by its index of 
    refraction delta(x,y) + i*beta(x,y) and some thickness along the z-axis.

    Parameters
    ----------
    in_wave : Wave
        Incident plane wave.
    phantom : VoxelPhantom
        2D phantom through which the wave is modulated.
    thickness : float, optional
        Phantom thickness [m]. The default is None, which defaults to the 
        phantom voxel thickness along the propagation axis (phantom.dz)

    Returns
    -------
    d_exit_wave : 2D cupy array
        Complex plane wave modulated through the phantom.
    """
    
    d_delta_slice, d_beta_slice = phantom.delta_beta_slice(in_wave.energy, z_ind=0)
    if thickness is None:
        thickness = phantom.dz  
    proj_delta, proj_beta = thickness*d_delta_slice, thickness*d_beta_slice
    d_exit_wave = in_wave.I0 * cp.exp(-in_wave.wavenum * (1j*proj_delta + proj_beta)) * cp.exp(1j * in_wave.wavenum * thickness)
    return d_exit_wave

    
def multislice_wave_voxels(in_wave, phantom, N_slices): 
    """
    Like the function project_wave_voxels(), but uses the multislice
    method to modulate in_wave with N_slices equally distributed over
    the phantom thickness dz.
    """
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
    

def project_wave_2phantoms(in_wave, phantom1, phantom2):
    """
    Use the projection approximation to modulate a coherent, monochromatic 
    plane wave through two analytical phantoms. There are currently two options
    for analytical phantoms: a sphere or a cylinder. This conveniently
    allows one to simulate simple 3D phantoms without need to generate a 
    voxelized phantom file.

    Parameters
    ----------
    in_wave : Wave
        Incident plane wave.
    phantom1, phantom2 : SpherePhantom or CylinderPhantom
        Contains the material and geometry information for each phantom.

    Returns
    -------
    d_exit_wave : 2D cupy array
        Complex plane wave modulated through the phantom.
    """
    
    # Get phantom parameters
    d_t1 = phantom1.projected_thickness()
    d_t2 = phantom2.projected_thickness()
    thickness = max(d_t1.max(), d_t2.max())
    
    delta1, beta1 = phantom1.delta_beta(in_wave.energy)
    delta2, beta2 = phantom2.delta_beta(in_wave.energy)
    
    # Project input wave through the object
    proj_delta = d_t1*delta1 + d_t2*delta2  
    proj_beta = d_t1*beta1 + d_t2*beta2
    d_exit_wave = in_wave.I0 * cp.exp(-in_wave.wavenum * (1j*proj_delta + proj_beta)) * cp.exp(1j * in_wave.wavenum * thickness)

    return d_exit_wave


def multislice_wave_2cylinders(in_wave, phantom1, phantom2, N_slices):
    """
    Like project_wave_2phantoms, but uses the multislice approach.
    ONLY accurate for CylinderPhantom.
    """
    
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
    
    # Project + propagate input wave N_slices times
    d_exit_wave = in_wave.I0
    for i in range(N_slices):
        d_exit_wave *= cp.exp(-in_wave.wavenum * (1j*proj_delta + proj_beta))        
        d_exit_wave = free_space_propagate(in_wave, d_exit_wave, slice_width)
        
    return d_exit_wave

    
def free_space_propagate(in_wave, exit_wave, prop_dist):
    """
    Compute wave after propagating through free space.
    Applies the Fresnel propagator in the Fourier domain.

    Parameters
    ----------
    in_wave : Wave
        Contains information about the incident wave geometry.
    exit_wave : 2D cupy array
        The complex plane wave after it has been modulated through a phantom.
    prop_dist : float
        Distance from the exit_wave plane to the detector plane.

    Returns
    -------
    d_fsp_wave : 2D cupy array
        Complex plane wave after FSP.
    """
    d_wave_ft = cp.fft.fft2(exit_wave)
    d_H = get_H(prop_dist, in_wave)
    d_fsp_wave = cp.fft.ifft2(d_wave_ft * d_H) # * cp.exp(1j*in_wave.wavenum*prop_dist) 
    return d_fsp_wave


def lorentzian2D(x, y, fwhm, normalize=True):
    """
    Generate a 2D Lorentzian kernel.
    x, y : 1D cupy array
        Grid coordinates [arbitrary length]
    fwhm : float
        Full-width at half-max of the Lorentzian (units must match x,y)
    """
    gamma = fwhm/2
    X, Y = cp.meshgrid(x, y)
    kernel = gamma / (2 * PI * (X**2 + Y**2 + gamma**2)**1.5)
    if normalize:
        kernel = kernel / cp.sum(kernel)
    return kernel


def detect_wave(in_wave, wave, upsample_multiple, blur_fwhm=0):
    """
    Detect a complex plane wave by downsampling the array to the detector thickness
    with a 2D block mean (integer multiples of the pixel width only!) and
    possibly applying a Lorentzian blur.
    
    TODO : add noise.
    
    Parameters
    ----------
    in_wave : Wave
        Contains information about the initial wave geometry.
    wave : 2D cupy array
        The complex plane wave incident on the detector.
    upsample_multiple : int
        Number of pixels over which to take the block mean for downsampling.
    blur_fwhm : float
        Full-width at half-max of the Lorentzian kernel, which defines the
        point-spread-function of the detector.
        If blur_fwhm == 0, no blurring is applied (ideal).

    Returns
    -------
    d_detected_wave : 2D cupy array
        Real detected plane wave.
    """
    
    if blur_fwhm > 0:
        # choose a sufficiently large kernel
        N_kernel = int(2 * blur_fwhm / in_wave.dx) + 1
        if (N_kernel > 2*in_wave.Nx + 1):
            N_kernel = 2*in_wave.Nx + 1  # crop if too large
        if (N_kernel%2 == 0):
            N_kernel -= 1
        # apply blur
        FOV_kern_x, FOV_kern_y = N_kernel*in_wave.dx, N_kernel*in_wave.dy
        d_x = cp.linspace(-FOV_kern_x/2, FOV_kern_x/2, N_kernel)
        d_y = cp.linspace(-FOV_kern_y/2, FOV_kern_y/2, N_kernel)
        d_lorentzian2d = lorentzian2D(d_x, d_y, blur_fwhm)
        d_wave_blurred = convolve2d(cp.abs(wave), d_lorentzian2d, mode='same')
        d_detected_wave = block_mean_2d(d_wave_blurred, upsample_multiple)
    else:
        d_detected_wave = block_mean_2d(cp.abs(wave), upsample_multiple)
    return d_detected_wave
    


#####################################################################
#####################################################################
###
### 1-SLICE PROJECTION FUNCTIONS - CT imaging
###         _______
###     ---| - - - |---->            y
###     ---| - - - |---->           |
###     ---| - - - |---->           |____x
###
### The incident wave source is rotated around the phantom in the x-y plane, 
### and the optic axis is through the one z-slice of the phantom
### defined in the x, y directions.
###
### TODO : comment functions
###


def project_db(d_proj_delta, d_proj_beta, proj_thickness, wavenum, I0=1):
    return I0 * cp.exp(-wavenum * (1j*d_proj_delta + d_proj_beta)) * cp.exp(1j * wavenum * proj_thickness)


def propagate_1D(prop_dist, d_wave, beam_width, wavelen): 
    Nx = d_wave.size
    d_fx = cp.arange(-Nx/2, Nx/2, 1)/beam_width + 1/(2*beam_width)  # centered frequencies of detector channels
    d_H = cp.fft.fftshift(cp.exp(-1j * wavelen * PI * prop_dist * d_fx**2))  # non-centered Fresnel operator
    d_wave_ft = cp.fft.fft(d_wave)
    d_fsp_wave = cp.fft.ifft(d_wave_ft * d_H) # * cp.exp(1j*in_wave.wavenum*prop_dist) 
    return d_fsp_wave


def project_xpc_ct(ct, phantom, wave, prop_dist, upx, thetas=None):  # upx = upsample factor
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
 
    
def multislice_xpc_ct(ct, phantom, wave, prop_dist, upx, N_slices=0, thetas=None):  # upx = upsample factor

    # First check if N_slices is 0 -- projection approx. case
    if N_slices == 0:
        return project_xpc_ct(ct, phantom, wave, prop_dist, upx, thetas)
    
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
    d_sino_upx = cp.ones([N_thetas, N_channels_upx], dtype=cp.complex64)    
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
            d_sino_upx[i_view] = propagate_1D(slice_width, d_sino_upx[i_view], ct.beam_width, wave.wavelen)
        src_x = trg_x  # update source coordinates for next run
        src_y = trg_y
        print(f'multislice {i_slice+1}/{N_slices} done - {time() - t0:.2f}s')
        
    # free-space propagate the exit wave to the detector + downsample 
    d_sino = cp.zeros([N_thetas, ct.N_channels], dtype=cp.float32)    
    d_channels_0 = cp.array(ct.channels, dtype=cp.float32)  # target channels for downsampling
    d_channels_upx = cp.array(channels_upx, dtype=cp.float32)
    for i_view in range(N_thetas):
        row = d_sino_upx[i_view]
        d_fsp_row = propagate_1D(prop_dist, row, ct.beam_width, wave.wavelen)
        d_sino[i_view] = cp.interp(d_channels_0, d_channels_upx, cp.abs(d_fsp_row))  # downsample by 1-D linterp

    return d_sino



def do_recon_patch(sino, ct, N_matrix, FOV, ramp, N_matrix_patch=256): 
    """
    The XPC-CT arrays can be too big for xtomosim at once, so this function
    reconstructs individual "patches" of the full FOV and then stitches them 
    together for a final reconstructed image.
    """
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




#####################################################################
#####################################################################
###
### PHASE RETRIEVAL 
###
### Use the XPCI simulation output for phase retrieval.
### Currently only supports Paganin's approach.
###

def paganin_thickness_1D(wave, R, FOVx, mu, delta, M=1):
    Nx = wave.size
    d_fx = cp.arange(0, Nx/2+1, 1).astype(cp.float32)/FOVx + 1/(2 * FOVx)  # real FFT freq's
    arg = mu * cp.fft.rfft(M**2 * wave) / (R * delta * d_fx / M + mu)     # argument for below
    d_T = -(1/mu) * cp.log(cp.fft.irfft(arg))  
    return d_T  


def paganin_thickness_sino(sino, R, FOVx, mu, delta, M=1):
    Ny, Nx = sino.shape
    T_sino = cp.zeros([Ny, Nx], dtype=cp.float32) 
    for i in range(Ny):
        T_sino[i] = paganin_thickness_1D(sino[i], R, FOVx, mu, delta, M)
    return T_sino



