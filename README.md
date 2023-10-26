# Optim-CGH

Optimisation of Computer-Generated Hologram (CGH) for holographic projections.

To keep things simple, all functions are placed within cgh_toolbox.py and the main function is in optim_cgh.py.
It is not a commercialized project therefore documentation may be lacking or unclear, and codes may include errors and mistakes,
if you see anything that is unclear or dodgy, please contact me at js2294@cam.ac.uk.

## Module dependencies
- pytorch (https://pytorch.org/)
- torchvision (https://pypi.org/project/torchvision/)
- numpy (https://numpy.org/install/)

## File description
- optim_cgh.py: Main file for hologram optimisation, including comparisons against GS with Sequential Slicing and DCGS(https://doi.org/10.1364/OE.27.008958)
- cgh_toolbox.py: toolbox for CGH related calculations
- Target_images: directory containing some sample target images
