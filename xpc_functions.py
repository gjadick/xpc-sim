import numpy as np
import os

h  = 6.62607015e-34      # Planck constant, J/Hz
c = 299792458.0          # speed of light, m/s
J_eV = 1.602176565e-19   # J per eV conversion


def get_wavelen(energy):
    # energy in keV -> returns wavelength in mm
    return h*c/(energy*J_eV)


def modulate_wave(fI, wavelen, object, dz, delta_ids, beta_ids):
    '''
    For a monochromatic, coherent plane wave fI in direction z,
    compute the modulation through object with complex index
    of refraction delta(x,y,z) + i*beta(x,y,z).
    '''
    # get delta and beta exit waves
    object_beta = np.zeros(object.shape, dtype=np.float64)
    object_delta = np.zeros(object.shape, dtype=np.float64)
    for object_id in np.unique(object):
        object_delta[object == object_id] = delta_ids[object_id]
        object_beta[object == object_id] = beta_ids[object_id]

    # compute transmission func from delta and beta projections
    proj_delta = np.sum(object_delta, axis=2)*dz   # sum instead of trapz in case Nz=1
    proj_beta  = np.sum(object_beta, axis=2)*dz
    #T = np.exp(-2*np.pi * (proj_delta - 1j*proj_beta) / wavelen)
    T = np.exp(-2*np.pi * (proj_beta + 1j*proj_delta) / wavelen)

    fO = fI*T
    return fO

def modulate_wave_multislice(fI, wavelen, object, dx, dy, dz, delta_ids, beta_ids):
    Nx, Ny, Nz = object.shape
    fO_i = fI  # initialize
    for i in range(Nz):
        slice_i = object[:,:,i,np.newaxis]
        fO_approx_i = modulate_wave(fO_i, wavelen, slice_i, dz, delta_ids, beta_ids)
        fO_i = propagate_wave(fO_approx_i, dz, wavelen, dx, dy)
    return fO_i

def get_fftfreqs(Nx, Ny, dx, dy):
    '''
    get discrete frequency samplings for computing the DFT
    '''
    fx_vals = np.arange(-Nx/2, Nx/2, 1.0).astype(np.float64)
    fy_vals = np.arange(-Ny/2, Ny/2, 1.0).astype(np.float64)
    fx_vals /= Nx*dx  # range from -1/2dx to +1/2dx
    fy_vals /= Ny*dy
    fx_coords, fy_coords = np.meshgrid(fx_vals, fy_vals)
    return fx_coords, fy_coords


def propagate_wave(fO, prop_dist, wavelen, dx, dy):
    '''
    compute wave after propagating distance prop_dist [mm]
    '''
    # get Fourier Transform of wave fO
    # we shift it so zero frequency is at the center
    FO = np.fft.fftshift(np.fft.fft2(fO))
    Nx, Ny = FO.shape

    # get the discretely-sampled Fourier-space Fresnel operator
    fx, fy = get_fftfreqs(Nx, Ny, dx, dy)   # frequency samples
    H = np.exp(-np.pi*wavelen*prop_dist*1j*(fx**2 + fy**2))  

    # propogate wave FO to the detector -> FD
    FD = FO * H
    fD = np.fft.ifft2(FD)
     
    return fD


def downsample2(img, sample):
    '''
    downsample with average of sample*sample blocks
    '''
    if sample==1:
        return img
        
    Nx_old, Ny_old = img.shape
    assert Nx_old%2 == 0
    assert Ny_old%2 == 0
    
    Nx = int(Nx_old/sample)
    Ny = int(Ny_old/sample)
    
    img_downsampled = np.zeros([Nx, Ny])
    for i in range(Nx):
        i_old = i*sample
        for j in range(Ny):
            j_old = j*sample
            img_downsampled[i,j] = np.mean(img[i_old:i_old+sample, j_old:j_old+sample])
    return img_downsampled