# Optim-CGH

Optimisation of Computer-Generated Hologram (CGH) for holographic projections.

To keep things simple, all functions are placed within cgh_toolbox.py and the main callers are multi_depth_holo_optim.py (https://doi.org/10.1364/JOSAA.478430), multi_frame_holo_optim.py (https://doi.org/10.1038/s41598-024-70428-0) and target_image_phase_optim.py (@TODO).
It is not a commercialized project therefore documentation may be unclear, and codes may include errors and mistakes, if you see anything that is unclear or dodgy, please contact me at jinze.sha@cantab.net.

## Module dependencies
- pytorch (https://pytorch.org/)
- torchvision (https://pypi.org/project/torchvision/)
- numpy (https://numpy.org/install/)

## File description
- multi_depth_holo_optim.py: Multi-depth phase-only hologram optimization using the L-BFGS algorithm with sequential slicing (https://doi.org/10.1364/JOSAA.478430)
- multi_frame_holo_optim.py: Multi-Frame Binary-Phase Holograms Batched Optimization for High Contrast Holographic Projections and Photolithography (https://doi.org/10.1038/s41598-024-70428-0)
- target_image_phase_optim.py: Target Image Phase Optimisation Hologram Generation Method for Phase-Only Spatial Light Modulators (@TODO)
- cgh_toolbox.py: toolbox for CGH related calculations
- Target_images: directory containing some sample target images (recommended sources: https://sipi.usc.edu/database/, https://data.vision.ee.ethz.ch/cvl/DIV2K/)
