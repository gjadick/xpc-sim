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
##
##
##
## 10/10/2023 - Gia Jadick  
## I modified this script to make multi-cell phantom.
## The ellipses are saved in a text file, or can be generated
## using the code snippet below.
##


import numpy as np
import os

rootpath = os.path.dirname(os.path.abspath(__file__)) + '/'


def phantom (n, ellipses):
	"""
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
	
	if (np.size (ellipses, 1) != 6):
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


if __name__ == '__main__':

    ID_background = 1  # water
    ID_cytoplasm = 10  # hematoxylin
    ID_nucleus = 11    # eosin

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
        
            cell_ellipse = [ID_cytoplasm-ID_background, 
                            cell_diameter_x, cell_diameter_y, x0, y0, 0]
        
            # nuclei deviation
            i_angle = cell_count % N_nucleus_angles
            cell_count += 1
            dx = nucleus_dist_from_center * np.cos(i_angle*nucleus_dtheta)
            dy = nucleus_dist_from_center * np.sin(i_angle*nucleus_dtheta)

            nucleus_ellipse = [ID_nucleus-ID_cytoplasm-ID_background, 
                               nucleus_diameter_x, nucleus_diameter_y, x0+dx, y0+dy, 0]
            multi_cell_ellipses.append(cell_ellipse)
            multi_cell_ellipses.append(nucleus_ellipse)

    multi_cell_phantom = phantom(N, multi_cell_ellipses) + ID_background
    multi_cell_phantom.astype(np.uint8).tofile(rootpath + f'/voxels_{N}_{sz*1e6}um_uint8.bin')





