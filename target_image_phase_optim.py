



#!/usr/bin/env python3
"""
Copyright(c) 2022 Jinze Sha (js2294@cam.ac.uk)
Centre for Molecular Materials, Photonics and Electronics, University of Cambridge
All Rights Reserved.

This is the python script for Limited-memory Broyden-Fletcher-Goldfarb-Shanno (L-BFGS)
optimisation of Computer-Generated Hologram (CGH) whose reconstruction is a 3D target
consisted of multiple slices of 2D images at different distances.
"""

import os
import time
import torch
import matplotlib.pyplot as plt
import numpy as np
import cgh_toolbox

# Experimental setup - device properties
PITCH_SIZE = 0.0000136  # Pitch size of the SLM
WAVELENGTH = 0.000000532  # Wavelength of the laser
ENERGY_CONSERVATION_SCALING = 1.0  # Scaling factor when conserving the energy of images across different slices


def main():
    NUM_ITERATIONS = 100
    PLOT_EACH_SLICE = True

    # 1. Load target images from files (please use PNG format with zero compression, even although PNG compression is lossless)
    # images = [os.path.join('Target_images', x) for x in ['A.png', 'B.png', 'C.png', 'D.png']]
    # images = [os.path.join('Target_images', x) for x in ['512_A.png', '512_B.png', '512_C.png', '512_D.png']]
    # images = [os.path.join('Target_images', 'mandrill_smaller.png')]
    images = [os.path.join('Target_images', 'mandrill.png')]
    # images = [os.path.join('Target_images', 'holography_ambigram_smaller.png')]
    target_fields = cgh_toolbox.load_target_images(images, energy_conserv_scaling=ENERGY_CONSERVATION_SCALING)

    # target_fields_list = []
    # donut_pattern = cgh_toolbox.energy_conserve(cgh_toolbox.generate_donut_pattern(radius=128, line_thickness=128))
    # donut_pattern = cgh_toolbox.zero_pad_to_size(donut_pattern, target_height=512, target_width=512)
    # target_fields_list.append(donut_pattern)
    # target_fields = torch.stack(target_fields_list)

    # 2. Set distances according to each slice of the target (in meters)
    # distances = [0.01 + i*0.01 for i in range(len(target_fields))]
    distances = [999]

    # 3. Check for mismatch between numbers of distances and images given
    if len(distances) != len(target_fields):
        raise ValueError("Different numbers of distances and images are given!")
    print("INFO: {} target fields loaded".format(len(target_fields)))

    # 4. Check if output folder exists, then save copies of target_fields
    if not os.path.isdir('Output'):
        os.makedirs('Output')
    for i, target_field in enumerate(target_fields):
        cgh_toolbox.save_image(os.path.join('Output', 'Target_field_{}'.format(i)), target_field)


    time_start = time.time()
    hologram, nmse_lists_LBFGS_MSE, time_list_LBFGS_MSE = cgh_toolbox.optim_cgh_3d(
        target_fields,
        distances,
        sequential_slicing=False,
        save_progress=False,
        iteration_number=NUM_ITERATIONS,
        cuda=True,
        learning_rate=0.1,
        record_all_nmse=True,
        optimise_algorithm="sgd",
        grad_history_size=100,
        loss_function=torch.nn.KLDivLoss(reduction="sum")
    )
    time_elapsed = time.time() - time_start
    to_print = "L-BFGS with MSE:\t time elapsed = {:.3f}s".format(time_elapsed)
    cgh_toolbox.save_hologram_and_its_recons(hologram, distances, "LBFGS_MSE")
    if PLOT_EACH_SLICE:
        for index, nmse_list in enumerate(nmse_lists_LBFGS_MSE):
            plt.plot(range(1, NUM_ITERATIONS + 1), nmse_list, ':', label="Phase hologram optimisation using SGD")
            to_print += "\tNMSE_{} = {:.15e}".format(index + 1, nmse_list[-1])
    print(to_print)

    time_start = time.time()
    hologram, nmse_lists_LBFGS_MSE, time_list_LBFGS_MSE = cgh_toolbox.optim_cgh_3d(
        target_fields,
        distances,
        sequential_slicing=False,
        save_progress=False,
        iteration_number=NUM_ITERATIONS,
        cuda=True,
        learning_rate=0.1,
        record_all_nmse=True,
        optimise_algorithm="lbfgs",
        grad_history_size=100,
        loss_function=torch.nn.KLDivLoss(reduction="sum")
    )
    time_elapsed = time.time() - time_start
    to_print = "L-BFGS with MSE:\t time elapsed = {:.3f}s".format(time_elapsed)
    cgh_toolbox.save_hologram_and_its_recons(hologram, distances, "LBFGS_MSE")
    if PLOT_EACH_SLICE:
        for index, nmse_list in enumerate(nmse_lists_LBFGS_MSE):
            plt.plot(range(1, NUM_ITERATIONS + 1), nmse_list, ':', label="Phase hologram optimisation using L-BFGS")
            to_print += "\tNMSE_{} = {:.15e}".format(index + 1, nmse_list[-1])
    print(to_print)



    time_start = time.time()
    hologram, nmse_lists_SGD_RE, time_list_SGD_RE = cgh_toolbox.tipo(
        target_fields,
        distances,
        iteration_number=NUM_ITERATIONS,
        cuda=True,
        learning_rate=0.1,
        optimise_algorithm="sgd",
        loss_function=torch.nn.KLDivLoss(reduction="sum")
        # loss_function=torch.nn.MSELoss(reduction="sum")
    )

    time_elapsed = time.time() - time_start
    to_print = "SGD with RE:\t time elapsed = {:.3f}s".format(time_elapsed)
    if PLOT_EACH_SLICE:
        for index, nmse_list in enumerate(nmse_lists_SGD_RE):
            plt.plot(range(1, NUM_ITERATIONS + 1), nmse_list, '-', label="Target image phase optimisation using SGD")
            to_print += "\tNMSE_{} = {:.15e}".format(index + 1, nmse_list[-1])
    print(to_print)


    time_start = time.time()
    hologram, nmse_lists_LBFGS_RE, time_list_LBFGS_RE = cgh_toolbox.tipo(
        target_fields,
        distances,
        iteration_number=NUM_ITERATIONS,
        cuda=True,
        learning_rate=0.1,
        optimise_algorithm="LBFGS",
        grad_history_size=5,
        loss_function=torch.nn.KLDivLoss(reduction="sum")
        # loss_function=torch.nn.MSELoss(reduction="sum")
    )
    time_elapsed = time.time() - time_start
    to_print = "L-BFGS with RE:\t time elapsed = {:.3f}s".format(time_elapsed)
    if PLOT_EACH_SLICE:
        for index, nmse_list in enumerate(nmse_lists_LBFGS_RE):
            plt.plot(range(1, NUM_ITERATIONS + 1), nmse_list, '-', label="Target image phase optimisation using L-BFGS")
            to_print += "\tNMSE_{} = {:.15e}".format(index + 1, nmse_list[-1])
    print(to_print)

    plt.xlabel("iterarion(s)")
    plt.ylabel("NMSE")
    plt.legend()
    plt.show()
    return

if __name__ == "__main__":
    main()
