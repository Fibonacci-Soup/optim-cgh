#!/usr/bin/env python3
"""
Copyright(c) 2023 Jinze Sha (js2294@cam.ac.uk)
Centre for Molecular Materials, Photonics and Electronics, University of Cambridge
All Rights Reserved.

This is the python script for Limited-memory Broyden-Fletcher-Goldfarb-Shanno (L-BFGS)
optimisation of Computer-Generated Hologram (CGH) whose reconstruction is a 3D target
consisted of multiple slices of 2D images at different distances.
"""

import os
import time
import torch
import torchvision
import numpy
from PIL import Image, ImageOps
import cgh_toolbox
import scipy.stats
import matplotlib.pyplot

IMAGE_DIRECTORY = ".\\Target_images\\test_images\\"

def main():
    image_filenames = os.listdir(IMAGE_DIRECTORY)
    entropy_scatter = []
    nmse_scatter = []
    for image_filename in image_filenames:
        target_image = Image.open(IMAGE_DIRECTORY + image_filename)
        target_image = ImageOps.grayscale(target_image)
        target_image = numpy.array(target_image)

        # Compute entropy
        value,counts = numpy.unique(target_image, return_counts=True)
        image_entropy = scipy.stats.entropy(counts, base=None)

        for manual_seed_i in range(3):
            target_field = torch.from_numpy(target_image)
            target_field = target_field.expand(1, -1, -1)
            hologram, nmse_list = cgh_toolbox.gerchberg_saxton_fraunhofer(target_field, iteration_number=100, manual_seed_value=manual_seed_i**10)
            entropy_scatter.append(image_entropy)
            nmse_scatter.append(nmse_list[-1])
            print(image_filename, entropy_scatter[-1], nmse_scatter[-1])

    matplotlib.pyplot.scatter(entropy_scatter, nmse_scatter)
    matplotlib.pyplot.show()



if __name__ == "__main__":
    main()