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