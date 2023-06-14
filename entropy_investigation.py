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
import matplotlib.pyplot as plt

IMAGE_DIRECTORY = ".\\Target_images\\test_images\\"

def main():
    image_filenames = os.listdir(IMAGE_DIRECTORY)
    entropy_scatter = {}
    nmse_scatter = {}
    for image_filename in image_filenames:
        target_image = Image.open(IMAGE_DIRECTORY + image_filename)
        target_image = ImageOps.grayscale(target_image)
        target_image = numpy.array(target_image) / 255.0

        # target_image = cgh_toolbox.generate_checkerboard_image(vertical_size=1024, horizontal_size=1024, size=128)


        # Compute entropy
        value,counts = numpy.unique(target_image, return_counts=True)
        image_entropy = scipy.stats.entropy(counts, base=None)
        print(image_entropy)
        # plt.imshow(target_image[0])
        # plt.show()


        for bit_depth in range(1, 9):
            if bit_depth not in entropy_scatter.keys():
                entropy_scatter[bit_depth] = []
                nmse_scatter[bit_depth] = []
            for manual_seed_i in range(10):
                target_field = torch.from_numpy(target_image)
                target_field = target_field.expand(1, -1, -1)
                # cgh_toolbox.save_image('.\\Output\\Target_field', target_field)
                hologram, nmse_list = cgh_toolbox.gerchberg_saxton_fraunhofer(target_field, iteration_number=100, manual_seed_value=manual_seed_i**10, hologram_quantization_bit_depth=bit_depth)
                entropy_scatter[bit_depth].append(image_entropy)
                nmse_scatter[bit_depth].append(nmse_list[-1])
                print(image_filename, entropy_scatter[bit_depth][-1], nmse_scatter[bit_depth][-1])


    fig, ax = plt.subplots()
    for i in entropy_scatter.keys():
        print(i)
        ax.scatter(entropy_scatter[i], nmse_scatter[i], label="holo bit depth = " + str(i))
        print("bit depth: ", i, "\tmean: ", numpy.mean(nmse_scatter[i]), "\tstd", numpy.std(nmse_scatter[i]))
    plt.xlabel("Target image entropy")
    plt.ylabel("NMSE")
    ax.legend()
    plt.show()




if __name__ == "__main__":
    main()