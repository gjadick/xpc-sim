import numpy as np
import scipy.fftpack as sfft
import matplotlib.pyplot as plt
from scipy.signal import convolve2d


### misc info

# discrete laplace
##  five point stencil
# laplace2d = np.array([[0,1,0],
#                       [1,-4,1],
#                       [0,1,0]]).astype(np.float32)
## seven point stencil
laplace2d = np.array([[0.25, 0.5, 0.25],
                      [0.5, -3.0, 0.5],
                      [0.25, 0.5, 0.25]]).astype(np.float32)
# from phasetorch
# Refractive index decrements for SiC, Teflon (C2F4), Al2O3, Polyimide (C22H10N2O5).
delta_gt = [1.67106361e-6, 1.09720611e-6, 2.02572255e-6, 7.61298395e-7]
# Absorption indices for SiC, Teflon (C2F4), Al2O3, Polyimide (C22H10N2O5).
beta_gt = [4.77064388e-9, 9.08938758e-10, 3.97340827e-9, 3.21013771e-10]


### some constants and helpful conversions


h  = 6.62607015e-34      # Planck constant, J/Hz
c = 299792458.0          # speed of light, m/s
J_eV = 1.602176565e-19   # J per eV conversion


def get_wavelen(energy): # energy in keV -> wavelength in m
    return 1e-3*h*c/(energy*J_eV)


def get_mu(beta, energy): # energy in keV -> linear atten coeff in m^-1
    return 4*np.pi*beta/get_wavelen(energy)



# functions
    
# thickness proj for SPHERE
def get_t_j(x,y,r): 
    if r**2 > x**2 + y**2:
        return 2*np.sqrt(r**2 - x**2 - y**2)
    else:
        return 0.0

def get_t_j_grid(x_vals, y_vals, r):
    t_j_grid = np.zeros([x_vals.size, y_vals.size]).astype(np.float32)
    for i,x in enumerate(x_vals):
        for j,y in enumerate(y_vals):
            t_j_grid[j,i] = get_t_j(x,y,r)
    return t_j_grid

# thickness proj for GAUSSIAN
    
def get_t_grid_gaussian(x_vals, y_vals, sigma):
    X, Y = np.meshgrid(x_vals, y_vals)
    t_j_grid = 2*sigma*np.exp(-(X**2 + Y**2)/(2*sigma**2))#/(2*np.pi*sigma**2)
    return t_j_grid
    
def get_ddt_grid_gaussian(x_vals, y_vals, sigma):
    X, Y = np.meshgrid(x_vals, y_vals)
    ddt_j_grid = 2*(X**2 + Y**2 - 2*sigma**2)*np.exp(-(X**2 + Y**2)/(2*sigma**2))/(sigma**3)
    return ddt_j_grid

def get_fftfreqs(Nx, Ny, dx, dy):
    '''
    get discrete frequency samplings for computing the DFT
    '''
    fx_vals = np.arange(-Nx/2, Nx/2, 1.0).astype(np.float64)
    fy_vals = np.arange(-Ny/2, Ny/2, 1.0).astype(np.float64)
    fx_vals /= Nx*dx  
    fy_vals /= Ny*dy
    fx_coords, fy_coords = np.meshgrid(fx_vals, fy_vals)
    return fx_coords, fy_coords

    
def modulate_wave(fI, wavelen, t1, t2, delta1, beta1, delta2, beta2):
    '''
    For a monochromatic, coherent plane wave fI in direction z,
    compute the modulation through object with complex index
    of refraction delta(x,y,z) + i*beta(x,y,z).
    '''
    proj_delta = t1*delta1 + t2*delta2
    proj_beta  = t1*beta1 + t2*beta2
    T = np.exp(-2*np.pi * (1j*proj_delta + proj_beta) / wavelen)
    fO = fI*T
    return fO


def propagate_wave(fO, prop_dist, wavelen, dx, dy):
    '''
    compute wave after propagating distance prop_dist [mm]

    Does the Fresnel prop in the Fourier domain, which is best
    assuming pixel size dx=dy is larger than the cutoff,
        pix_liml = wavelen*prop_dist/(dx*min(Ny, Nx))
        pix_limh = wavelen*prop_dist/(dx*max(Ny, Nx))
    in the case pix_liml=pix_limh since Ny=Nx
    '''
    Nx, Ny = fO.shape        

    # get non-centered (shifted) frequency samples
    fx, fy = get_fftfreqs(Nx, Ny, dx, dy)  # centered
    fx = sfft.fftshift(fx)
    fy = sfft.fftshift(fy)
    
    # FFT space with non-centered frequencies (see scipy docs)
    FO = sfft.fft2(fO)
    H = np.exp(-np.pi*wavelen*prop_dist*1j*(fx**2 + fy**2))  
    FD = FO*H
    fD = sfft.ifft2(FD)
    return fD
    

# def downsample2(img, sample):
#     '''
#     downsample with average of sample*sample blocks

#     This causes some weird averaging effects! 
#     '''
#     if sample==1:
#         return img
        
#     Nx_old, Ny_old = img.shape
#     #assert Nx_old%2 == 0
#     #assert Ny_old%2 == 0
    
#     Nx = int(Nx_old/sample)
#     Ny = int(Ny_old/sample)
    
#     img_downsampled = np.zeros([Nx, Ny])
#     for i in range(Nx):
#         i_old = i*sample
#         for j in range(Ny):
#             j_old = j*sample
#             img_downsampled[i,j] = np.mean(img[i_old:i_old+sample, j_old:j_old+sample])
#     return img_downsampled


def block_mean_2d(arr, Nblock):
    '''
    Computes the block mean over a 2D array `arr`
    in sublocks of size Nblock x Nblock.
    If an axis size of arr is not an integer multiple 
    of Nblock, then zero padding is added at the end.

    This will cause image artifacts when fringes are too close
    relative to pixel size.
    '''
    Ny, Nx = arr.shape

    # add padding if needed
    padx, pady = 0, 0
    if Nx%Nblock != 0:
        padx = Nblock - Nx%Nblock
    if Ny%Nblock != 0:
        pady = Nblock - Ny%Nblock
    arr_pad = np.pad(arr, [(0, pady), (0, padx)])
    Ny, Nx = arr_pad.shape

    # compute block mean
    block_mean = arr_pad.reshape(Ny//Nblock, Nblock, Nx//Nblock, Nblock).mean(axis=(1,-1))
    return block_mean


def get_m_mono(A_1, A_2, I_i, eta, prop_dist, mu_1, delta_1, mu_2, delta_2, x_vals):
    '''
    Requires external func `get_t_j_grid` to be defined for params A_j! 
    '''
    t_1 = get_t_j_grid(x_vals, x_vals, A_1)
    ddt_1 = convolve2d(t_1, laplace2d, mode='same')

    t_2 = get_t_j_grid(x_vals, x_vals, A_2)
    ddt_2 = convolve2d(t_2, laplace2d, mode='same')

    arg_1 = -mu_1*t_1 + prop_dist*delta_1*ddt_1
    arg_2 = -mu_2*t_2 + prop_dist*delta_2*ddt_2
        
    signal = I_i * eta * np.exp(arg_1 + arg_2)

    return signal



def get_dm_dA_mono(dA_1, dA_2, A_1, A_2, I_i, eta, prop_dist, mu_1, delta_1, mu_2, delta_2, x_vals, EPS=1e-8):
    '''
    applying FTC:
      f'(x) = lim_[h->0] (f(x+dx) - f(x))/dx
    Either dx1 or dx2 should be 0 for these partials.
    '''
    assert np.abs(dA_1)<EPS or np.abs(dA_2)<EPS  # confirm h1 or h2 = 0
    m_0 = get_m_mono(A_1,      A_2,      I_i, eta, prop_dist, mu_1, delta_1, mu_2, delta_2, x_vals)
    m_f = get_m_mono(A_1+dA_1, A_2+dA_2, I_i, eta, prop_dist, mu_1, delta_1, mu_2, delta_2, x_vals)
    dm_dA = (m_f - m_0)/(dA_1 + dA_2)
    return dm_dA

def get_m_gauss(A_1, A_2, I_i, eta, prop_dist, mu_1, delta_1, mu_2, delta_2, x_vals):
    t_1 = get_t_gauss(x_vals, x_vals, A_1)
    ddt_1 = get_ddt_gauss(x_vals, x_vals, A_1)

    t_2 = get_t_gauss(x_vals, x_vals, A_2)
    ddt_2 = get_ddt_gauss(x_vals, x_vals, A_2)

    arg_1 = -mu_1*t_1 + prop_dist*delta_1*ddt_1
    arg_2 = -mu_2*t_2 + prop_dist*delta_2*ddt_2
        
    signal = I_i * eta * np.exp(arg_1 + arg_2)

    return signal

def get_dm_dA_gauss(ind, A_1, A_2, I_i, eta, prop_dist, mu_1, delta_1, mu_2, delta_2, x_vals):
    # monoenergetic shortcut!!!
    signal = get_dm_gauss(A_1, A_2, I_i, eta, prop_dist, mu_1, delta_1, mu_2, delta_2, x_vals)
    
    assert ind==1 or ind==2
    if ind == 1:
        A = A_1
        mu = mu_1
        delta = delta_1
        t = get_t_gauss(x_vals, x_vals, A_1)
        ddt = get_ddt_gauss(x_vals, x_vals, A_1)
    else:
        A = A_2
        mu = mu_2
        delta = delta_2
        t = get_t_gauss(x_vals, x_vals, A_2)
        ddt = get_ddt_gauss(x_vals, x_vals, A_2)
    
    X, Y = np.meshgrid(x_vals, x_vals)
    term1 = -( X**2 + Y**2 + 2*A**2 )*np.exp((X**2 + Y**2)/(2*A**2))/(2*np.pi*A**5)
    term2 = -( 8*A**4 + 8*A**2*(X**2 + Y**2) + X**4 + 2*X**2*Y**2 + Y**4 )*np.exp((X**2 + Y**2)/(2*A**2))/(2*np.pi*A**9)
    
    return signal*(-mu*term1 - prop_dist*delta*term2)    



def get_m_mono_fresnel(A_1, A_2, I_i, E, prop_dist, beta_1, delta_1, beta_2, delta_2, x_vals, px_sz, upsampx=None):
    '''
    Requires external func `get_t_j_grid` to be defined for params A_j! 
    '''
    t_1 = get_t_j_grid(x_vals, x_vals, A_1)
    t_2 = get_t_j_grid(x_vals, x_vals, A_2)
    f_obj_proj = modulate_wave(I_i, get_wavelen(E), t_1, t_2, delta_1, beta_1, delta_2, beta_2)
    f_det_proj = propagate_wave(f_obj_proj, prop_dist,  get_wavelen(E), px_sz, px_sz)
    signal_proj = np.abs(f_det_proj)
    if upsampx is not None:
        #signal_proj = downsample2(signal_proj, upsampx)
        signal_proj = block_mean_2d(signal_proj, upsampx)
    return signal_proj


def get_dm_dA_mono_fresnel(dA_1, dA_2, A_1, A_2, I_i, E, prop_dist, beta_1, delta_1, beta_2, delta_2, x_vals, px_sz, EPS=1e-8, upsampx=None):
    '''
    applying FTC:
      f'(x) = lim_[h->0] (f(x+dx) - f(x))/dx
    Either dx1 or dx2 should be 0 for these partials.
    '''
    assert np.abs(dA_1)<EPS or np.abs(dA_2)<EPS  # confirm h1 or h2 = 0
    m_0 = get_m_mono_fresnel(A_1,      A_2,      I_i, E, prop_dist, beta_1, delta_1, beta_2, delta_2, x_vals, px_sz)
    m_f = get_m_mono_fresnel(A_1+dA_1, A_2+dA_2, I_i, E, prop_dist, beta_1, delta_1, beta_2, delta_2, x_vals, px_sz)
    dm_dA = (m_f - m_0)/(dA_1 + dA_2)
    if upsampx is not None:
        #dm_dA = downsample2(dm_dA, upsampx)
        dm_dA = block_mean_2d(dm_dA, upsampx)
    return dm_dA


def test_plot_objsig(sig, r1, r2):
    t_1 = get_t_j_grid(x_vals, x_vals, r1)
    t_2 = get_t_j_grid(x_vals, x_vals, r2)

    fig, ax = plt.subplots(1,3,figsize=[11,3], dpi=300)
    ax[0].set_title('$t_1(x,y)$')
    m = ax[0].imshow(t_1)
    fig.colorbar(m, ax=ax[0])
    
    ax[1].set_title('$t_2(x,y)$')
    m = ax[1].imshow(t_2)
    fig.colorbar(m, ax=ax[1])
    
    ax[2].set_title('image')
    m = ax[2].imshow(sig)
    fig.colorbar(m, ax=ax[2])

    for axi in ax.ravel():
        axi.axis('off')
    fig.tight_layout()
    plt.show()





