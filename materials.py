#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 10 11:21:11 2023

@author: giavanna

scratch!!! for creating phantoms
"""

import csv
import numpy as np


ELEMENTS =  ['H','He','Li','Be','B','C','N','O','F','Ne','Na','Mg','Al','Si',\
    'P','S','Cl','Ar','K','Ca','Sc','Ti','V','Cr','Mn','Fe','Co','Ni','Cu','Zn',\
    'Ga','Ge','As','Se','Br','Kr','Rb','Sr','Y','Zr','Nb','Mo','Tc','Ru','Rh',\
    'Pd','Ag','Cd','In','Sn','Sb','Te','I','Xe','Cs','Ba','La','Ce','Pr','Nd',\
    'Pm','Sm','Eu','Gd','Tb','Dy','Ho','Er','Tm','Yb','Lu','Hf','Ta','W','Re',\
    'Os','Ir','Pt','Au','Hg','Tl','Pb','Bi','Po','At','Rn','Fr','Ra','Ac','Th',\
    'Pa','U','Np','Pu','Am','Cm','Bk','Cf','Es','Fm']


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



## Copyright (C) 2010  Alex Opie  <lx_op@orcon.net.nz>
##
## This program is free software; you can redistribute it and/or modify it
## under the terms of the GNU General Public License as published by
## the Free Software Foundation; either version 3 of the License, or (at
## your option) any later version.
##
## This program is distributed in the hope that it will be useful, but
## WITHOUT ANY WARRANTY; without even the implied warranty of
## MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.	See the GNU
## General Public License for more details.
##
## You should have received a copy of the GNU General Public License
## along with this program; see the file COPYING.  If not, see
## <http://www.gnu.org/licenses/>.

def phantom (n = 256, p_type = 'Modified Shepp-Logan', ellipses = None):
	"""
	 phantom (n = 256, p_type = 'Modified Shepp-Logan', ellipses = None)
	
	Create a Shepp-Logan or modified Shepp-Logan phantom.

	A phantom is a known object (either real or purely mathematical) 
	that is used for testing image reconstruction algorithms.  The 
	Shepp-Logan phantom is a popular mathematical model of a cranial
	slice, made up of a set of ellipses.  This allows rigorous 
	testing of computed tomography (CT) algorithms as it can be 
	analytically transformed with the radon transform (see the 
	function `radon').
	
	Inputs
	------
	n : The edge length of the square image to be produced.
	
	p_type : The type of phantom to produce. Either 
	  "Modified Shepp-Logan" or "Shepp-Logan".  This is overridden
	  if `ellipses' is also specified.
	
	ellipses : Custom set of ellipses to use.  These should be in 
	  the form
	  	[[I, a, b, x0, y0, phi],
	  	 [I, a, b, x0, y0, phi],
	  	 ...]
	  where each row defines an ellipse.
	  I : Additive intensity of the ellipse.
	  a : Length of the major axis.
	  b : Length of the minor axis.
	  x0 : Horizontal offset of the centre of the ellipse.
	  y0 : Vertical offset of the centre of the ellipse.
	  phi : Counterclockwise rotation of the ellipse in degrees,
	        measured as the angle between the horizontal axis and 
	        the ellipse major axis.
	  The image bounding box in the algorithm is [-1, -1], [1, 1], 
	  so the values of a, b, x0, y0 should all be specified with
	  respect to this box.
	
	Output
	------
	P : A phantom image.
	
	Usage example
	-------------
	  import matplotlib.pyplot as pl
	  P = phantom ()
	  pl.imshow (P)
	
	References
	----------
	Shepp, L. A.; Logan, B. F.; Reconstructing Interior Head Tissue 
	from X-Ray Transmissions, IEEE Transactions on Nuclear Science,
	Feb. 1974, p. 232.
	
	Toft, P.; "The Radon Transform - Theory and Implementation", 
	Ph.D. thesis, Department of Mathematical Modelling, Technical 
	University of Denmark, June 1996.
	
	"""
	
	if (ellipses is None):
		ellipses = _select_phantom (p_type)
	elif (np.size (ellipses, 1) != 6):
		raise AssertionError ("Wrong number of columns in user phantom")
	
	# Blank image
	p = np.zeros ((n, n))

	# Create the pixel grid
	ygrid, xgrid = np.mgrid[-1:1:(1j*n), -1:1:(1j*n)]

	for ellip in ellipses:
		I   = ellip [0]
		a2  = ellip [1]**2
		b2  = ellip [2]**2
		x0  = ellip [3]
		y0  = ellip [4]
		phi = ellip [5] * np.pi / 180  # Rotation angle in radians
		
		# Create the offset x and y values for the grid
		x = xgrid - x0
		y = ygrid - y0
		
		cos_p = np.cos (phi) 
		sin_p = np.sin (phi)
		
		# Find the pixels within the ellipse
		locs = (((x * cos_p + y * sin_p)**2) / a2 
              + ((y * cos_p - x * sin_p)**2) / b2) <= 1
		
		# Add the ellipse intensity to those pixels
		p [locs] += I

	return p


def _select_phantom (name):
	if (name.lower () == 'shepp-logan'):
		e = _shepp_logan ()
	elif (name.lower () == 'modified shepp-logan'):
		e = _mod_shepp_logan ()
	else:
		raise ValueError ("Unknown phantom type: %s" % name)
	
	return e


def _shepp_logan ():
	#  Standard head phantom, taken from Shepp & Logan
	return [[   2,   .69,   .92,    0,      0,   0],
	        [-.98, .6624, .8740,    0, -.0184,   0],
	        [-.02, .1100, .3100,  .22,      0, -18],
	        [-.02, .1600, .4100, -.22,      0,  18],
	        [ .01, .2100, .2500,    0,    .35,   0],
	        [ .01, .0460, .0460,    0,     .1,   0],
	        [ .02, .0460, .0460,    0,    -.1,   0],
	        [ .01, .0460, .0230, -.08,  -.605,   0],
	        [ .01, .0230, .0230,    0,  -.606,   0],
	        [ .01, .0230, .0460,  .06,  -.605,   0]]

def _mod_shepp_logan ():
	#  Modified version of Shepp & Logan's head phantom, 
	#  adjusted to improve contrast.  Taken from Toft.
	return [[   1,   .69,   .92,    0,      0,   0],
	        [-.80, .6624, .8740,    0, -.0184,   0],
	        [-.20, .1100, .3100,  .22,      0, -18],
	        [-.20, .1600, .4100, -.22,      0,  18],
	        [ .10, .2100, .2500,    0,    .35,   0],
	        [ .10, .0460, .0460,    0,     .1,   0],
	        [ .10, .0460, .0460,    0,    -.1,   0],
	        [ .10, .0460, .0230, -.08,  -.605,   0],
	        [ .10, .0230, .0230,    0,  -.606,   0],
	        [ .10, .0230, .0460,  .06,  -.605,   0]]

#def ?? ():
#	# Add any further phantoms of interest here
#	return np.array (
#	 [[ 0, 0, 0, 0, 0, 0],
#	  [ 0, 0, 0, 0, 0, 0]])







# make a 2D multi-cell phantom
sz = 0.5e-6 # pixel size [m]
N = 256  # matrix size
upsample_multiple = 8

FOV = N*sz   # 0.5 um * 8192 = 4.096 mm
sz /= upsample_multiple
N *= upsample_multiple

N_cells = 7
cell_diameter_x = 10e-6/FOV  # 10um cells
cell_diameter_y = 10e-6/FOV  # 10um cells
nucleus_diameter_x = 2.5e-6/FOV
nucleus_diameter_y = 2.5e-6/FOV
cell_spacing = 6e-6/FOV  # 20um between cells

cells_FOV_x = N_cells * (cell_diameter_x + cell_spacing)
cells_FOV_y = N_cells * (cell_diameter_y + cell_spacing)


N_nucleus_angles = 8
nucleus_dtheta = 2*np.pi / N_nucleus_angles
nucleus_dist_from_center = 0.5*cell_diameter_x

cell_count = 0
multi_cell_ellipses = []     
for i in np.arange(-N_cells/2 + 0.5, N_cells/2 + 0.5):
    for j in np.arange(-N_cells/2 + 0.5, N_cells/2 + 0.5):
        
        # cell center
        x0 = 2*i*(cell_spacing + cell_diameter_x) 
        y0 = 2*j*(cell_spacing + cell_diameter_y)
        
        cell_ellipse = [9, cell_diameter_x, cell_diameter_y, x0, y0, 0]
        
        # nuclei deviation
        i_angle = cell_count % N_nucleus_angles
        cell_count += 1
        dx = nucleus_dist_from_center * np.cos(i_angle*nucleus_dtheta)
        dy = nucleus_dist_from_center * np.sin(i_angle*nucleus_dtheta)

        nucleus_ellipse = [1, nucleus_diameter_x, nucleus_diameter_y, x0+dx, y0+dy, 0]
        multi_cell_ellipses.append(cell_ellipse)
        multi_cell_ellipses.append(nucleus_ellipse)


#multi_cell_ellipses = [[1, .005, .01, 0, 0, 0]] 

#test = phantom(n=N, ellipses=multi_cell_ellipses) + 1mg

#%%




# fig,ax=plt.subplots(1,1,dpi=600)
# ax.axis('off')
# plt.imshow(test)
# plt.colorbar()
# plt.show()




