#!/usr/bin/env python3
"""
Copyright(c) 2023 Jinze Sha (js2294@cam.ac.uk)
Centre for Molecular Materials, Photonics and Electronics, University of Cambridge
All Rights Reserved.

This is the python script for entropy investigation on CGH
"""

import os
import torch
import numpy
from PIL import Image, ImageOps
import cgh_toolbox
import scipy.stats
import matplotlib.pyplot as plt
import pickle
import csv

IMAGE_DIRECTORY = os.path.join('Target_images', 'test_images')

def main():
    image_filenames = os.listdir(IMAGE_DIRECTORY)
    entropy_scatter = {}
    nmse_scatter = {}
    with open('entropy_investigation.csv', 'w', newline='') as output_file:
        file_writer = csv.writer(output_file)
        file_writer.writerow(['image_filename', 'image_entropy', 'bit_depth', 'NMSE', 'holo_entropy'])
        for image_filename in image_filenames:
            target_image = Image.open(os.path.join(IMAGE_DIRECTORY, image_filename))
            target_image = ImageOps.grayscale(target_image)
            target_image = numpy.array(target_image) / 255.0

            # Compute entropy of target image
            value, counts = numpy.unique(target_image, return_counts=True)
            image_entropy = scipy.stats.entropy(counts, base=2)
            # plt.imshow(target_image[0])
            # plt.show()

            # Generate hologram and get NSME
            for bit_depth in range(1, 9):
                if bit_depth not in entropy_scatter.keys():
                    entropy_scatter[bit_depth] = []
                    nmse_scatter[bit_depth] = []
                for manual_seed_i in range(5):
                    target_field = torch.from_numpy(target_image)
                    target_field = target_field.expand(1, -1, -1)
                    _, nmse_list, phase_hologram = cgh_toolbox.gerchberg_saxton_fraunhofer(target_field, iteration_number=100, manual_seed_value=manual_seed_i**10, hologram_quantization_bit_depth=bit_depth)
                    entropy_scatter[bit_depth].append(image_entropy)
                    nmse_scatter[bit_depth].append(nmse_list[-1])

                    # Compute entropy of the hologram
                    value, counts = numpy.unique(phase_hologram.cpu().numpy(), return_counts=True)
                    holo_entropy = scipy.stats.entropy(counts, base=2)

                    # Output result for each iteration
                    print(image_filename, 'image entropy:', image_entropy, 'bit depth:', bit_depth, 'NMSE:', nmse_scatter[bit_depth][-1], 'holo_entropy', holo_entropy)
                    file_writer.writerow([image_filename, image_entropy, bit_depth, nmse_scatter[bit_depth][-1], holo_entropy])

    with open('variables_entropy_investigation.pkl', 'wb') as f:  # Python 3: open(..., 'wb')
        pickle.dump([image_filenames, entropy_scatter, nmse_scatter], f)
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
