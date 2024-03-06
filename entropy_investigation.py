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

# IMAGE_DIRECTORY = os.path.join('Target_images', 'test_images')
IMAGE_DIRECTORY = os.path.join('Target_images', 'DIV2K_train_HR')

def compute_delentropy(input_image):
    image = numpy.array(ImageOps.grayscale(input_image))
    # ImageOps.grayscale(input_image).save("target_image.png")
    # Using a 2x2 difference kernel [[-1,+1],[-1,+1]] results in artifacts!
    # In tests the deldensity seemed to follow a diagonal because of the
    # assymetry introduced by the backward/forward difference
    # the central difference correspond to a convolution kernel of
    # [[-1,0,1],[-1,0,1],[-1,0,1]] and its transposed, produces a symmetric
    # deldensity for random noise.
    fx = ( image[:,2:] - image[:,:-2] )[1:-1,:]
    fy = ( image[2:,:] - image[:-2,:] )[:,1:-1]

    # im = Image.fromarray(fx)
    # im = im.convert("L")
    # im.save("fx.png")

    # im = Image.fromarray(fy)
    # im = im.convert("L")
    # im.save("fy.png")

    diffRange = numpy.max([numpy.abs(fx.min()), numpy.abs(fx.max()), numpy.abs(fy.min()), numpy.abs(fy.max())])
    if diffRange >= 200   and diffRange <= 255  : diffRange = 255
    if diffRange >= 60000 and diffRange <= 65535: diffRange = 65535

    # see paper eq. (17)
    # The bin edges must be integers, that's why the number of bins and range depends on each other
    nBins = min( 1024, 2*diffRange+1 )
    if image.dtype == float:
        nBins = 1024
    # Centering the bins is necessary because else all value will lie on
    # the bin edges thereby leading to assymetric artifacts
    dbin = 0 if image.dtype == float else 0.5
    r = diffRange + dbin
    delDensity, xedges, yedges = numpy.histogram2d( fx.flatten(), fy.flatten(), bins = nBins, range = [ [-r,r], [-r,r] ] )
    if nBins == 2*diffRange+1:
        assert( xedges[1] - xedges[0] == 1.0 )
        assert( yedges[1] - yedges[0] == 1.0 )


    # Normalization for entropy calculation. np.sum( H ) should be ( imageWidth-1 )*( imageHeight-1 )
    # The -1 stems from the lost pixels when calculating the gradients with non-periodic boundary conditions
    delDensity = delDensity / numpy.sum(delDensity) # see paper eq. (17)
    delDensity = delDensity.T

    # "The entropy is a sum of terms of the form p log(p). When p=0 you instead use the limiting value (as p approaches 0 from above), which is 0."
    # The 0.5 factor is discussed in the paper chapter "4.3 Papoulis generalized sampling halves the delentropy"
    delentropy = -0.5 * numpy.sum(delDensity[delDensity.nonzero()] * numpy.log2( delDensity[delDensity.nonzero()])) # see paper eq. (16)

    return delentropy


def main():
    image_filenames = os.listdir(IMAGE_DIRECTORY)
    image_entropy_dict = {}
    image_delentropy_dict = {}

    entropy_scatter = {}
    delentropy_scatter = {}
    nmse_scatter = {}
    with open('holo_information_investigation_GS_Fresnel0.1.csv', 'w', newline='') as output_file:
        file_writer = csv.writer(output_file)
        file_writer.writerow(['image_filename', 'image_entropy', 'image_delentropy', 'holo_bit_depth', 'NMSE', 'holo_entropy'])
        for image_filename in image_filenames[68:]:
            target_image = Image.open(os.path.join(IMAGE_DIRECTORY, image_filename))


            # Get image entropy and delentropy
            if (image_filename not in image_entropy_dict) or (image_filename not in image_delentropy_dict):

                # Comput delentropy of target image
                image_delentropy = compute_delentropy(target_image)

                target_image = ImageOps.grayscale(target_image)
                target_image = numpy.array(target_image) / 255.0

                # Compute entropy of target image
                value, counts = numpy.unique(target_image, return_counts=True)
                image_entropy = scipy.stats.entropy(counts, base=2)

                # Save the result for later iterations
                image_delentropy_dict[image_filename] = image_delentropy
                image_entropy_dict[image_filename] = image_entropy
            else:
                image_delentropy = image_delentropy_dict[image_filename]
                image_entropy = image_entropy_dict[image_filename]


            # Generate hologram and get NSME
            for bit_depth in range(1, 9):
                if bit_depth not in entropy_scatter.keys():
                    entropy_scatter[bit_depth] = []
                    delentropy_scatter[bit_depth] = []
                    nmse_scatter[bit_depth] = []
                for manual_seed_i in range(1):
                    target_field = torch.from_numpy(target_image)
                    target_field = target_field.expand(1, -1, -1)
                    _, nmse_list, phase_hologram = cgh_toolbox.gerchberg_saxton_single_slice(target_field, \
                        iteration_number=100, manual_seed_value=manual_seed_i**10, hologram_quantization_bit_depth=bit_depth, distance=0.1)
                    entropy_scatter[bit_depth].append(image_entropy)
                    delentropy_scatter[bit_depth].append(image_delentropy)
                    nmse_scatter[bit_depth].append(nmse_list[-1])

                    # Compute entropy of the hologram
                    value, counts = numpy.unique(phase_hologram.cpu().numpy(), return_counts=True)
                    holo_entropy = scipy.stats.entropy(counts, base=2)

                    # Output result for each iteration
                    print(image_filename, 'image entropy:', image_entropy, 'image delentropy:', image_delentropy, 'hologram bit depth:', bit_depth, 'NMSE:', nmse_scatter[bit_depth][-1], 'holo_entropy', holo_entropy)
                    file_writer.writerow([image_filename, image_entropy, image_delentropy, bit_depth, nmse_scatter[bit_depth][-1], holo_entropy])

    with open('variables_entropy_investigation.pkl', 'wb') as f:  # Python 3: open(..., 'wb')
        pickle.dump([image_filenames, entropy_scatter, delentropy_scatter, nmse_scatter], f)
    fig, ax = plt.subplots()
    for i in entropy_scatter.keys():
        print(i)
        ax.scatter(entropy_scatter[i], nmse_scatter[i], label="holo bit depth = " + str(i))
        print("bit depth: ", i, "\tmean: ", numpy.mean(nmse_scatter[i]), "\tstd", numpy.std(nmse_scatter[i]))
    plt.xlabel("Target image entropy")
    plt.ylabel("NMSE between reconstruction and target image")
    ax.legend()
    plt.show()

    fig, ax = plt.subplots()
    for i in delentropy_scatter.keys():
        print(i)
        ax.scatter(delentropy_scatter[i], nmse_scatter[i], label="holo bit depth = " + str(i))
        print("bit depth: ", i, "\tmean: ", numpy.mean(nmse_scatter[i]), "\tstd", numpy.std(nmse_scatter[i]))
    plt.xlabel("Target image delentropy")
    plt.ylabel("NMSE between reconstruction and target image")
    ax.legend()
    plt.show()



if __name__ == "__main__":
    main()
