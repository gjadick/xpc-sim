# xpc-sim

Simulate x-ray phase-contrast (XPC) imaging. Supports both planar and
tomographic imaging geometries, voxelized and analytical phantoms, and 
either the projection approximation or multislice forward model to modulate
the incident wave through the phantom. Currently only for monochromatic plane waves.

Xpc-sim uses CUDA to efficiently forward project through voxelized phantoms. 
For very large phantoms and/or sinogram shapes, a batching technique is used
so that only a maximum of 10 GB of GPU memory is used at any one time. This
allows one to simulate very high detector resolutions, sufficient to observe
the difference in simulation output when using the multislice vs. projection
approximation forward models.

There is also an option to apply phase retrieval using Paganin's technique.


## Quick start

After cloning the xpc-sim repository, you will also need to initialize 
the xtomosim submodule. For example:

```
git clone https://github.com/gjadick/xpc-sim.git
cd xpc-sim/xtomosim
git submodule init
git submodule update
```

This should pull all the xtomosim scripts necessary for XPC-CT simulations.
The required Python packages are listed in the xtomosim submodule. I recommend
installing the requirements in a virtual environment. From the base
xpc-sim directory:

```
python -m venv xpc-env
source xpc-env/bin/activate
python -m pip install --upgrade pip
python -m pip install -r xtomosim/requirements.txt
```

Now you should have everything for running simulations! 
An example XPC-CT parameter file **input/params/params_voxel.txt**
hase been provided, along with the necessary input files, so **main.py** is
ready to run:

```
python main.py
```


## Simulations

Currently the main loop only includes a function for XPC-CT simulation, but
scripts for planar radiographic imaging can be found in **xpc.py**. A function
for this geometry will be added to the main loop soon.

### XPC-CT 

XPC-CT simulations use a voxelized phantom and a parameter file similar to the 
one described in the README of the xtomosim/main branch.
However, there are some additional parameters specifying phase-specific 
information, and other xtomosim parameters are not yet supported. Some 
parameter names have not yet been updated to match xtomosim. Please note
these differences to ensure your simulations run smoothly.


XPC-specific parameters:

| **Parameter name** | **Type** | **Description** |
| ---------------- | --------| ----- |
| *propagation_distance* | float | Distance from phantom exit plane to detector [m]. |
| *number_of_projection_slices* | int | Number of "slices" for the forward model. 0 == projection approximation, else use multislice. |
| *wave_amplitude* | float | Total photon counts in the incidient x-ray beam. |
| *wave_energy* | float | Monochromatic beam energy [keV]. |
| *grid_size* | int | Number of detector elements. |
| *pixel_size* | float | Size of each detector element [m] |
| *upsample_multiple* | int | Factor by which to upsample wave front during the forward simulation. Downsampling occurs during the detection stage. |



Xtomosim-like parameters:

| **Parameter name** | **Type** | **Description** |
| ---------------- | --------| ----- |
| *RUN_ID* | str | Unique identifier for the simulation. This will become the directory any outputs. If it is not changed for different simulations, old data will be overwritten. |
| *phantom_type* | str | "voxel" (other analytical options coming soon) |
| *phantom_name* | str | Unique identifier for the phantom. | 
| *phantom_filepath* | str | Path to the phantom file directory. |
| *phantom_filename* | str | Name of the raw uint8 file *within the filepath above* with the phantom material indices. Should be raveled from the shape [Nx, Ny, Nz]. |
| *material_filename* | str | Path to the csv file with the density and atomic composition corresponding to each material index in the phantom file. (see example) |
| *phantom_Nx* | int >= 1 | Number of pixels in the phantom x-direction (lateral). |
| *phantom_Ny* | int >= 1| Number of pixels in the phantom y-direction (anterio-posterior). |
| *phantom_Nz* | int == 1| Currently only 1 z-direction (axial) layer is supported. |
| *voxel_dx* | float | Size of phantom pixels in x-direction [m] |
| *voxel_dy* | float | Size of phantom pixels in y-direction [m] |
| *voxel_dz* | float | Phantom thickness [m]. |
| *scanner_geometry* | "parallel_beam" | Simulation geometry. Currently only supports parallel beam (synchrotron). |
| *SID* | float > 0 | source-to-isocenter distance [m] |
| *SDD* | float > *SID* | source-to-detector distance [m] |
| *N_channels* | int >= 1 | Number of detector channels. These will be equally spaced within *beam_width* |
| *N_projections* | int >= 1| Number of projection views. These will be equally spaced within *rotation_angle_total*. |
| *beam_width* | float > 0 | Total detector width [m]. |
| *rotation_angle_total* | float > 0 | Total rotation angle of the x-ray source. |
| *detector_px_height* | float > 0 | Height of detector pixels [cm] (used for determining total x-ray flux from the spectrum file). |
| *detector_mode* | str, "eid" OR "pcd" | Detection scheme, either energy-integrating or photon-counting. |
| *detector_std_electronic* | int | Standard deviation of additive Poisson noise in units of photon counts. This emulates electronic noise. |
| *detector_filename* | "ideal" OR str | Path to raw float32 file with the detector efficiency-vs-energy [keV] data. Set to "ideal" to simply use efficiency = 1.0 for all energies. |
| *N_recon_matrix* | int | Number of pixels in the x- and y-direction of the reconstructed image. |
| *FOV_recon* | float | Field-of-view of the reconstructed image [cm]. |
| *ramp_filter_percent_Nyquist* | float > 0.0 AND <= 1.0 | Reconstruction filter cutoff frequency. |


### XPC-radiography 
Wrapper script with parameter file coming soon. You can check out **xpc.py** 
for the building blocks.



## Misc.
Tested in Python 3.12.1.




