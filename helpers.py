#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct  3 12:06:52 2023

@author: giavanna

!!! TODO !!!
    Clean up the phantom classes -- make "analytical phantom"
    and "voxelized phantom"
    Use subclasses, `super`? 

    Add a "modulated wave" class or some modulation methods to the Wave class
    
    Add Detector class with blur (Lorentzian?) and/or block mean downsampling
    
    Create methods for Waves/Phantoms/Detectors to interact with different sized grids
"""

import numpy as np
import cupy as cp
from cupyx.scipy.interpolate import RegularGridInterpolator
import json
import csv
import os
import xscatter as xsf  # x-ray scatter factors
import xcompy as xc  # x-ray linear attenuation coefficients
from xtomosim.system import ParallelBeamGeometry

# Physical constants
PLANCK = 6.62607015e-34  # [J/Hz] Planck constant h
SPEED_OF_LIGHT = 2.99792458e8   # [m/s] speed of light in vacuum c
J_PER_KEV = 1.602176565e-16   # unit conversion J --> keV
PI = 3.141592653589793
ELEMENTS =  ['H','He','Li','Be','B','C','N','O','F','Ne','Na','Mg','Al','Si',\
    'P','S','Cl','Ar','K','Ca','Sc','Ti','V','Cr','Mn','Fe','Co','Ni','Cu','Zn',\
    'Ga','Ge','As','Se','Br','Kr','Rb','Sr','Y','Zr','Nb','Mo','Tc','Ru','Rh',\
    'Pd','Ag','Cd','In','Sn','Sb','Te','I','Xe','Cs','Ba','La','Ce','Pr','Nd',\
    'Pm','Sm','Eu','Gd','Tb','Dy','Ho','Er','Tm','Yb','Lu','Hf','Ta','W','Re',\
    'Os','Ir','Pt','Au','Hg','Tl','Pb','Bi','Po','At','Rn','Fr','Ra','Ac','Th',\
    'Pa','U','Np','Pu','Am','Cm','Bk','Cf','Es','Fm']


# double or single precision? -- double might be needed for Fresnel! 
#real_type = cp.float32
#complex_type = cp.complex64
real_type = cp.float64
complex_type = cp.complex128


def cp_array(arr):
    if cp.iscomplexobj(arr):
        return cp.array(arr, dtype=complex_type)
    else:
        return cp.array(arr, dtype=real_type)
    

def energy_to_wavelen(energy): 
    """
    Convert energy [keV] -> wavelength [m]
    """
    return PLANCK*SPEED_OF_LIGHT / (energy*J_PER_KEV)


def energy_to_wavenum(energy): 
    """
    Convert energy [keV] -> wavenumber [m^-1]
    wavenum = 2 * PI / wavelen
    """
    return (2*PI*energy*J_PER_KEV) / (PLANCK*SPEED_OF_LIGHT)


def beta_to_mu(beta, energy): 
    """
    Convert energy [keV] -> linear attenuation coefficient [m^-1]
    """
    return 4*PI*beta/energy_to_wavelen(energy)


def get_delta_beta(energy, matcomp, density):
    """
    Get real and imaginary parts of the complex index of refraction,
    delta (real, refractive) and beta (imaginary, absorptive) at the
    energy/energies [keV] of interest.
    """
    delta, beta = xsf.get_delta_beta_mix(matcomp, energy, density) 
    return delta, beta


def get_mu(energy, matcomp, density):
    """
    Get linear attenuation coefficient mu from xcompy. In theory this
    is equal to 2*k*beta, but because the beta and mu values use 
    different databases, they are a bit off. 
    
    TODO: check difference in 2*k*beta and mu values.
    """
    mu = density*xc.mixatten(matcomp, np.array([energy]).astype(np.float64).squeeze())
    mu = 100*mu  # convert units cm^-1 --> m^-1
    if mu.size == 1:
        mu = mu[0]  # for a single energy, don't return an array
    return mu       
  

def block_mean_2d(arr, Nblock):
    """
    Computes the block mean over a 2D array `arr`
    in sublocks of size Nblock x Nblock.
    If an axis size of arr is not an integer multiple 
    of Nblock, then zero padding is added at the end.

    This will cause image artifacts when fringes are too close
    relative to pixel size.
    """
    Ny, Nx = arr.shape
    padx, pady = 0, 0
    if Nx%Nblock != 0:
        padx = Nblock - Nx%Nblock
    if Ny%Nblock != 0:
        pady = Nblock - Ny%Nblock
    d_arr = cp.array(arr)
    d_arr_pad = cp.pad(d_arr, [(0, pady), (0, padx)])
    Ny, Nx = d_arr_pad.shape
    d_block_mean = d_arr_pad.reshape(Ny//Nblock, Nblock, Nx//Nblock, Nblock).mean(axis=(1,-1))
    return d_block_mean


class Wave():
    """
    A class to handle an x-ray wave front. Currently only for a coherent,
    monochromatic plane wave.
    """
    def __init__(self, amplitude, energy, plane_shape, pixel_size):  # initialize the plane
        self.I0 = amplitude
        self.energy = energy  # [keV]
        self.Nx, self.Ny = plane_shape
        self.dx, self.dy = pixel_size  # [m]
        self.wavelen = energy_to_wavelen(energy)
        self.wavenum = energy_to_wavenum(energy)
        self.FOVx = self.Nx*self.dx
        self.FOVy = self.Ny*self.dy
        
    def fftfreqs(self):   # Compute centered, discrete frequency samplings on the grid for DFT
        d_fx_vals = cp.arange(-self.Nx/2, self.Nx/2, 1).astype(real_type) / self.FOVx + 1/(2*self.FOVx)
        d_fy_vals = cp.arange(-self.Ny/2, self.Ny/2, 1).astype(real_type) / self.FOVy + 1/(2*self.FOVy)
        d_fx_grid, d_fy_grid = cp.meshgrid(d_fx_vals, d_fy_vals)
        return d_fx_grid, d_fy_grid
    
    def update_energy(self, new_energy):
        self.energy = new_energy  # [keV]
        self.wavelen = energy_to_wavelen(new_energy)
        self.wavenum = energy_to_wavenum(new_energy)
     
        
# class ModulatedWave():
#     # Like a wave, but modulated through a phantom
#     def __init__(self, in_wave, phantom):
#         self.in_wave = in_wave
#         self.phantom = phantom
#     def project_wave(self):
#     def free_space_propagate(self, prop_dist):
#
    
# superclass
class Phantom():
    """
    A class to contain geometric and material properties of a voxelized, 
    computational phantom.    
    
    TODO: material properties, projection methods
    """
    def __init__(self, id, shape, voxel_size):
        self.name = id
        self.Nx, self.Ny, self.Nz = shape
        self.dx, self.dy, self.dz = voxel_size  # [m]
        self.FOVx = self.Nx*self.dx
        self.FOVy = self.Ny*self.dy
        self.FOVz = self.Nz*self.dz
        
    def grid_coordinates(self):  # Define the grid coordinates
        d_x_vals = cp.linspace(-self.FOVx/2, self.FOVx/2, self.Nx).astype(real_type) + self.dx/2 
        d_y_vals = cp.linspace(-self.FOVy/2, self.FOVy/2, self.Ny).astype(real_type) + self.dy/2
        d_X, d_Y = cp.meshgrid(d_x_vals, d_y_vals)
        return d_X, d_Y
    
    
# subclass
class VoxelPhantom(Phantom):
    def __init__(self, id, phantom_filename, material_filename,
                 shape, voxel_size, dtype): 
        super().__init__(id, shape, voxel_size)
        self.phantom_filename = phantom_filename
        self.material_filename = material_filename
                
        # Load phantom from file. Voxels correspond to a material ID number
        self.voxels = np.fromfile(phantom_filename, dtype=dtype).reshape(shape[::-1])
        self.d_voxels = cp.array(self.voxels)
        
        # Load the material properties
        self.material_dict = material_csv_to_dict(material_filename)
        self.material_keys = list(self.material_dict.keys())
             
    def resample_xy(self, Nx_new, Ny_new, dx_new, dy_new):
        d_x_old = cp.linspace(-self.FOVx/2, self.FOVx/2, self.Nx).copy().astype(real_type) + self.dx/2 
        d_y_old = cp.linspace(-self.FOVy/2, self.FOVy/2, self.Ny).copy().astype(real_type) + self.dy/2
        d_z = cp.linspace(-self.FOVz/2, self.FOVz/2, self.Nz).astype(real_type) + self.dz/2
        interp = RegularGridInterpolator((d_z, d_x_old, d_y_old), self.d_voxels, 
                                         method='nearest', bounds_error=False, fill_value=0)

        # update params
        self.Nx, self.Ny = Nx_new, Ny_new
        self.dx, self.dy = dx_new, dy_new
        self.FOVx = Nx_new*dx_new
        self.FOVy = Ny_new*dy_new
        d_x_new = cp.linspace(-self.FOVx/2, self.FOVx/2, self.Nx).astype(real_type) + self.dx/2 
        d_y_new = cp.linspace(-self.FOVy/2, self.FOVy/2, self.Ny).astype(real_type) + self.dy/2
        
        Z, Y, X = cp.meshgrid(d_z, d_y_new, d_x_new)
        pts = cp.array([Z.ravel(), Y.ravel(), X.ravel()]).T
        self.d_voxels = interp(pts).reshape([self.Nz, self.Ny, self.Nx]).astype(cp.uint8)
        self.voxels = self.d_voxels.get()
                    
    def delta_beta(self, energy, voxel_id):
        density, matcomp = self.material_dict[voxel_id]
        return get_delta_beta(energy, matcomp, density)
    
    def mu(self, energy, voxel_id):
        density, matcomp = self.material_dict[voxel_id]
        return get_mu(energy, matcomp, density)  
    
    def delta_beta_slice(self, energy, z_ind=0):
        d_slice = self.d_voxels[z_ind, :, :]
        d_delta_slice = cp.zeros(d_slice.shape)
        d_beta_slice = cp.zeros(d_slice.shape)
        for m in self.material_keys:
            delta, beta = self.delta_beta(energy, m)
            d_delta_slice[d_slice == m] = delta
            d_beta_slice[d_slice == m] = beta
        return d_delta_slice, d_beta_slice
        
        
        
class SpherePhantom(Phantom):
    """
    A class to more efficiently handle a single-material, spherical phantom
    in vacuum. Using SpherePhantom instead of Phantom speeds up operations 
    like computing the projected thickness (which can be done analytically)
    and eliminates the need to store large 3D array files.
    
    The main purpose of the sphere class is for use in the Cramer Rao Lower
    Bound (CRLB) calculations. 
    """
    def __init__(self, radius, material_id, material_composition, density,
                 plane_shape, pixel_scale):
        if len(plane_shape) == 2:
            plane_shape = list(plane_shape) + [0]
            pixel_scale = list(pixel_scale) + [0]
        super().__init__(f'sphere_{material_id}_{2*radius*1e6}um', plane_shape, pixel_scale)
        self.radius = radius  # [m]
        self.matcomp = material_composition
        self.density = density
        self.d_thickness = None  # init
        
    def update_radius(self, new_radius):
        self.radius = new_radius  # [m]
        self.d_thickness = None  # init
        
    def projected_thickness(self):
        """ Analytically compute projected thickness through the sphere. """
        if self.d_thickness is None:
            d_X, d_Y = self.grid_coordinates()
            d_thickness = 2*cp.sqrt(self.radius**2 - d_X**2 - d_Y**2)
            d_thickness = d_thickness.ravel()
            d_thickness[cp.isnan(d_thickness)] = 0.0
            d_thickness = d_thickness.reshape(d_X.shape)
            self.d_thickness = d_thickness
        return self.d_thickness
    
    def delta_beta(self, energy):
        return get_delta_beta(energy, self.matcomp, self.density)
    
    def mu(self, energy):
        return get_mu(energy, self.matcomp, self.density)  
    
    
class CylinderPhantom(Phantom):
    """
    A class to more efficiently handle a single-material, cylinder phantom
    in vacuum. 
    
    Useful for investigating the effect of object absorption (increasing
    cylinder length) without getting super large sphere matrix sizes.
    """
    def __init__(self, radius, material_id, material_composition, density,
                 plane_shape, pixel_scale, cylinder_length):
        if len(plane_shape) == 2:
            plane_shape = list(plane_shape) + [0]
            pixel_scale = list(pixel_scale) + [0]
        super().__init__(f'cylinder_{material_id}_{2*radius*1e6}um', plane_shape, pixel_scale)
        self.radius = radius  # [m]
        self.matcomp = material_composition
        self.density = density
        self.length = cylinder_length

    def projected_thickness(self):
        """ Get grid of the projected thickness. (just a circle on 0s) """
        d_X, d_Y = self.grid_coordinates()
        d_thickness = cp.zeros(d_X.shape, dtype=real_type)
        d_thickness[d_X**2 + d_Y**2 < self.radius**2] = self.length
        return d_thickness
    
    def delta_beta(self, energy):
        return get_delta_beta(energy, self.matcomp, self.density)
    
    def mu(self, energy):
        return get_mu(energy, self.matcomp, self.density)                         
    
    
class Detector():
    """
    A class to handle geometric and physics properties of an x-ray detector.
    Assumes flat panel geometry. 
    
    TODO: add geometry, detective efficiency, up/downsampling, blur, etc.
    It would be nice to downsample at the detector with x2 multiple block avg.
    (apply weighted averaging over partial pixels or a Gaussian blur?)
    
    Lorentzian detector blur model? (get rid of high frequency components that cause aliasing)
    """
    def __init__(self, plane_shape, pixel_size):
        pass 


def make_combos(var_list, combos=None, this_col=None):
    """
    Recursive function to generate a list of mixed-type lists, all possible 
    combinations of the sub-list variables in var_list. Used for creating 
    combinations of looping variables. Example:
        
    >>> make_combos([[True, False], [100,200], ['a','b','c']])
    [[True, 100, 'a'], [True, 100, 'b'], [True, 100, 'c'], [True, 200, 'a'],  
     [True, 200, 'b'], [True, 200, 'c'], [False, 100, 'a'], [False, 100, 'b'],  
     [False, 100, 'c'], [False, 200, 'a'], [False, 200, 'b'], [False, 200, 'c']]
    """

    def place_col(M, col, x):
        for i in range(len(x)):
            M[i][col] = x[i]

    def place_M(M, mini_M, i, j):
        for x in range(len(mini_M[0])):
            for y in range(len(mini_M)):
                M[j+y][i+x] = mini_M[y][x]
                
    # make everything a list
    for i in range(len(var_list)):
        if not isinstance(var_list[i], list):
            var_list[i] = [var_list[i]]  # assuming this is a single value...

    N = np.prod([len(v) for v in var_list])  # total num of combination vectors
    if combos is None:  # initialize things that update recursively later
        combos = [['foo' for i in range(len(var_list))] for j in range(N)]  
        this_col = 0  

    m = int(N / len(var_list[0])) # number of times to repeat first vector
    place_col(combos, 0, np.repeat(var_list[0], m))

    if var_list[1:]: # if there are more variables, loop over them
        this_col += 1
        sub_combo = make_combos(var_list[1:], combos=[v[1:] for v in combos][:m], this_col=this_col)
        for i_opt in range(len(var_list[0])):
            place_M(combos, sub_combo, 1, i_opt*m)

    return combos


def material_csv_to_dict(filename):
    """
    Read a CSV file containing material data into a dictionary. Each row
    gives the density and chemical composition by weight fraction for a given
    material name. These entries should correspond to a voxelized phantom with
    voxel values equal to different ID numbers (0 to 255) that are used to
    identify the material of that voxel. So the CSV file contains the needed
    material information for computing material-dependent parameters like the
    linear attenuation coefficient and anomolous scattering factors.
    
    An example file is `input/materials.csv`. 

    Parameters
    ----------
    filename : str
        Path to the material data file (csv format).

    Returns
    -------
    material_dictionary : dict
        Dictionary of material density and chemical composition by weight
        corresponding to each material ID number in a voxelized phantom.

    """
    material_dictionary = {}  
    with open(filename) as f:
        reader = csv.DictReader(f)
        for row in reader:
            key = int(row['ID'])
            density = float(row['density'])
            matcomp = row_to_matcomp(row)
            material_dictionary[key] = [density, matcomp]
    return material_dictionary


def row_to_matcomp(row):
    """
    Convert a row from a CSV file (read from the file to a dictionary format)
    with the chemical weight fractions to a formatted material composition 
    string, e.g. for water 'H(11.2)O(88.8)'.

    Parameters
    ----------
    row : dict
        Dictionary of row items read from the material CSV file.

    Returns
    -------
    matcomp : str
        Formatted material composition string of elements by weight fractions.

    """
    matcomp = ''
    for elem in ELEMENTS:
        if row[elem] != '':
            matcomp += f'{elem}({row[elem]})'
    return matcomp



def read_parameter_file(filename):
    with open(filename) as f:
        all_parameters = json.load(f)
    
    # Unpack the phantom objects (might not be desirable depending on what to loop over)
    phantom_type = all_parameters['phantom']['phantom_type']
    if phantom_type == 'two_objects':
        phantom_keys = list(all_parameters['phantom']['object1'].keys())
        for key in phantom_keys:
            all_parameters[key+'1'] = all_parameters['phantom']['object1'][key]
            all_parameters[key+'2'] = all_parameters['phantom']['object2'][key]
    elif phantom_type == 'voxel':
         phantom_keys = list(all_parameters['phantom']['object'].keys())
         for key in phantom_keys:
             all_parameters[key] = all_parameters['phantom']['object'][key]
    else:
        print('Only support for `two_objects` or `voxel` type phantoms')  # !!!
        
    del all_parameters['phantom']  # clean up
    
    # Make dictionaries for each set of parameter combinations
    param_keys = list(all_parameters.keys())
    param_value_combos = make_combos(list(all_parameters.values()))
    param_dicts = [dict(zip(param_keys, values)) for values in param_value_combos]
    
    # Package the parameters into objects for each run
    param_list = []
    for p in param_dicts:
        # The wave
        num_pixels = int(p['grid_size'] * p['upsample_multiple'])  
        px_size = p['pixel_size'] / p['upsample_multiple']
        grid_shape = [num_pixels, num_pixels]
        pixel_scale = [px_size, px_size]
        wave = Wave(p['wave_amplitude'], p['wave_energy'], grid_shape, pixel_scale)
        
        # The object, probably a better way to initialize this stuff
        # TODO: add error catching, try/except + define errors
        if phantom_type == 'two_objects':
            if p['shape1'] == 'sphere':
                object1 = SpherePhantom(p['radius1'], p['material_name1'], 
                                        p['material_composition1'], p['density1'], 
                                        grid_shape, pixel_scale)
            elif p['shape1'] == 'cylinder':
                object1 = CylinderPhantom(p['radius1'], p['material_name1'], 
                                        p['material_composition1'], p['density1'], 
                                        grid_shape, pixel_scale, p['length1'])            
            if p['shape2'] == 'sphere':
                object2 = SpherePhantom(p['radius2'], p['material_name2'], 
                                        p['material_composition2'], p['density2'], 
                                        grid_shape, pixel_scale)
            elif p['shape2'] == 'cylinder':
                object2 = CylinderPhantom(p['radius2'], p['material_name2'], 
                                        p['material_composition2'], p['density2'], 
                                        grid_shape, pixel_scale, p['length2'])      
            phantom = [object1, object2]
        elif phantom_type == 'voxel':
            phantom_filename = os.path.join(p['phantom_filepath'], p['phantom_filename'])
            material_filename = os.path.join(p['phantom_filepath'], p['material_filename'])
            shape = [p['phantom_Nx'], p['phantom_Ny'], p['phantom_Nz']]
            voxel_size = [p['voxel_dx'], p['voxel_dy'], p['voxel_dz']]
            phantom = VoxelPhantom(p['name'], phantom_filename, material_filename,
                         shape, voxel_size, dtype=np.uint8)
        
        param_list.append([p['RUN_ID'],
                           p['propagation_distance'],
                           p['number_of_projection_slices'],
                           p['upsample_multiple'], 
                           wave, phantom])
    return param_list


def read_parameter_file_proj(filename):
    with open(filename) as f:
        all_parameters = json.load(f)

    # Make dictionaries for each set of parameter combinations
    param_keys = list(all_parameters.keys())
    param_value_combos = make_combos(list(all_parameters.values()))
    param_dicts = [dict(zip(param_keys, values)) for values in param_value_combos]
    
    # Package the parameters into objects for each run
    param_list = []
    for p in param_dicts:
        # The wave
        num_pixels = int(p['grid_size'] * p['upsample_multiple'])  
        px_size = p['pixel_size'] / p['upsample_multiple']
        grid_shape = [num_pixels, num_pixels]
        pixel_scale = [px_size, px_size]
        wave = Wave(p['wave_amplitude'], p['wave_energy'], grid_shape, pixel_scale)
        
        # Voxel phantom
        if p['phantom_type'] != 'voxel':
            print('Projection parameters only defined for voxel phantom!')
            return -1
        phantom_filename = os.path.join(p['phantom_filepath'], p['phantom_filename'])
        material_filename = os.path.join(p['phantom_filepath'], p['material_filename'])
        shape = [p['phantom_Nx'], p['phantom_Ny'], p['phantom_Nz']]
        voxel_size = [p['voxel_dx'], p['voxel_dy'], p['voxel_dz']]
        phantom = VoxelPhantom(p['phantom_name'], phantom_filename, material_filename,
                     shape, voxel_size, dtype=np.uint8)
        
        param_list.append([p['RUN_ID'],
                           p['propagation_distance'],
                           p['number_of_projection_slices'],
                           p['upsample_multiple'], 
                           wave, phantom])
    return param_list


def read_parameter_file_ct(filename):  ## different formatting for CT sim
    """
    All lengths should be in units [m]
    """
    with open(filename) as f:
        all_parameters = json.load(f)

    # Make dictionaries for each set of parameter combinations
    param_keys = list(all_parameters.keys())
    param_value_combos = make_combos(list(all_parameters.values()))
    param_dicts = [dict(zip(param_keys, values)) for values in param_value_combos]
    
    # Package the parameters into objects for each run
    param_list = []
    for p in param_dicts:

        # Wave
        num_pixels = int(p['N_channels'])
        px_size = p['beam_width'] / p['N_channels']
        grid_shape = [num_pixels, 1]  # 1D! 
        pixel_scale = [px_size, p['detector_px_height']]
        wave = Wave(p['wave_amplitude'], p['wave_energy'], grid_shape, pixel_scale)
        
        # Voxel phantom
        if p['phantom_type'] != 'voxel':
            print('CT parameters only defined for voxel phantom!')
            return -1
        phantom_filename = os.path.join(p['phantom_filepath'], p['phantom_filename'])
        material_filename = os.path.join(p['phantom_filepath'], p['material_filename'])
        shape = [p['phantom_Nx'], p['phantom_Ny'], p['phantom_Nz']]
        voxel_size = [p['voxel_dx'], p['voxel_dy'], p['voxel_dz']]
        phantom = VoxelPhantom(p['phantom_name'], phantom_filename, material_filename,
                     shape, voxel_size, dtype=np.uint8)
        
        # Scanner geometry
        if p['scanner_geometry'] != 'parallel_beam':
            print('CT parameters only defined for parallel_beam geometry!')
            return -1
        eid = p['detector_mode'] == 'eid'  # convert to bool
        ct = ParallelBeamGeometry(N_channels=p['N_channels'], 
                                  N_proj=p['N_projections'],
                                  beam_width=p['beam_width'], 
                                  theta_tot=p['rotation_angle_total'],
                                  SID=p['SID'], 
                                  SDD=p['SDD'],  # for propagation-based imaging, SDD is the object exit plane location.
                                  eid=eid, 
                                  h_iso=p['detector_px_height'],
                                  detector_file=p['detector_filename'],
                                  detector_std_electronic=p['detector_std_electronic'])
    
        param_list.append([p['RUN_ID'],
                           p['propagation_distance'],
                           p['number_of_projection_slices'],
                           p['upsample_multiple'], 
                           wave, phantom, ct,
                           p['N_recon_matrix'], p['FOV_recon'], p['ramp_filter_percent_Nyquist']
                           ])
    return param_list




if __name__=='__main__':
    # only for testing the parameter file
    parameter_sets = read_parameter_file_ct('input/params/params_voxel.txt')



