#!/usr/bin/env python3
"""
Copyright(c) 2023 Jinze Sha (js2294@cam.ac.uk)
Centre for Molecular Materials, Photonics and Electronics, University of Cambridge
All Rights Reserved.

This is the python script for multi frame hologram optimisation.
"""

import os
import time
import torch
import matplotlib.pyplot as plt
import cgh_toolbox
# import csv
# import pickle
# nmse_list_LBFGS_RE_all = []

def main():
    NUM_ITERATIONS = 1000
    NUM_FRAMES = 24
    ENERGY_CONSERVATION_SCALING = 1.0  # Scaling factor when conserving the energy of images across different slices

    # 1. Load target images from files (please use PNG format with zero compression, even though PNG compression is lossless)
    # images = [os.path.join('Target_images', x) for x in ['A.png', 'B.png', 'C.png', 'D.png']]
    # images = [os.path.join('Target_images', x) for x in ['512_A.png', '512_B.png', '512_C.png', '512_D.png']]
    # images = [os.path.join('Target_images', x) for x in ['IMG_5793.JPG', 'IMG_5794.JPG', 'IMG_5795.JPG', 'IMG_5796.JPG']]
    # images = [os.path.join('Target_images', x) for x in ['IMG_5798.JPG', 'IMG_5799.JPG', 'IMG_5800.JPG']]
    # images = [os.path.join('Target_images', 'holography_ambigram_smaller.png')]
    images = [os.path.join('Target_images', 'mandrill_smaller.png')]
    target_fields = cgh_toolbox.load_target_images(images, energy_conserv_scaling=ENERGY_CONSERVATION_SCALING)


    # 2. Set distances according to each slice of the target (in meters)
    distances = [999]
    # distances = [1.1, 1.2, 1.3]
    # Check for mismatch between numbers of distances and images given
    if len(distances) != len(target_fields):
        raise ValueError("Different numbers of distances and images are given!")

    print("INFO: {} target fields loaded".format(len(target_fields)))

    # Check if output folder exists, then save copies of target_fields
    if not os.path.isdir('Output'):
        os.makedirs('Output')
    for i, target_field in enumerate(target_fields):
        cgh_toolbox.save_image(os.path.join('Output', 'Target_field_{}'.format(i)), target_field)



    time_start = time.time()
    _, nmse_list_LBFGS_RE, _ = cgh_toolbox.multi_frame_cgh(
        target_fields,
        distances,
        iteration_number=NUM_ITERATIONS,
        cuda=True,
        learning_rate=0.01,
        optimise_algorithm="LBFGS",
        grad_history_size=10,
        num_frames=NUM_FRAMES,
        loss_function=torch.nn.KLDivLoss(reduction="sum")
        # loss_function=torch.nn.MSELoss(reduction="sum")
    )
    time_elapsed = time.time() - time_start
    print("MFHO:\t time elapsed = {:.3f}s".format(time_elapsed))
    # Save results to file
    # cgh_toolbox.save_hologram_and_its_recons(hologram, distances, "LBFGS_RE")
    # nmse_list_LBFGS_RE_all.append(nmse_list_LBFGS_RE)
    plt.plot(range(1, NUM_ITERATIONS + 1), nmse_list_LBFGS_RE, '-', label="Number_of_frames: {}".format(NUM_FRAMES))

    plt.xlabel("iterarions")
    plt.ylabel("NMSE")
    plt.legend()
    plt.show()

    # with open(os.path.join('Output', 'runs_statistics.csv'), 'a', newline='') as output_file:
    #     file_writer = csv.writer(output_file)
    #     print('writing to csv file:', [images, NUM_ITERATIONS, NUM_FRAMES, time_elapsed, 'NMSE:', nmse_list_LBFGS_RE[-1]])
    #     file_writer.writerow([images, 'Num_iterations:', NUM_ITERATIONS, 'Num_frames:', NUM_FRAMES, 'Time:', time_elapsed, 'NMSE:', nmse_list_LBFGS_RE[-1]])
    return


if __name__ == "__main__":
    # for i in [24, 12, 8, 6, 4, 3, 2, 1]:
    #     main(i)
    main()
    # with open('nmse_list_LBFGS_RE_all.pickle', 'wb') as handle:
    #     pickle.dump(nmse_list_LBFGS_RE_all, handle)
    # plt.xlabel("iterarions")
    # plt.ylabel("NMSE")
    # plt.legend()
    # plt.show()

