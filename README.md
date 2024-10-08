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