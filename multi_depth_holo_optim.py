#!/usr/bin/env python3
"""
Copyright(c) 2022 Jinze Sha (jinze.sha@cantab.net)
Centre for Molecular Materials, Photonics and Electronics, University of Cambridge
All Rights Reserved.

This is the python script for Limited-memory Broyden-Fletcher-Goldfarb-Shanno (L-BFGS) optimisation of Computer-Generated Hologram (CGH) whose reconstruction is a 3D target consisted of multiple slices of 2D images at different distances.
If using L-BFGS with Sequential Slicing, please make reference to: https://doi.org/10.1364/JOSAA.478430
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
    SEQUENTIAL_SLICING = False
    PLOT_EACH_SLICE = True

    # 1.A Option1: Load target images from files (please use PNG format with zero compression, although PNG compression is lossless)
    images = [os.path.join('Target_images', x) for x in ['512_A.png', '512_B.png', '512_C.png', '512_D.png']]
    # images = [os.path.join('Target_images', 'Teapot_slices', 'Teapot_720p_section_{}.png'.format(x)) for x in range(20, 601, 20)]
    target_fields = cgh_toolbox.load_target_images(images, energy_conserv_scaling=ENERGY_CONSERVATION_SCALING)


    # 1.B Option2: Generate target fields of specific patterns (e.g. grid pattern here)
    # target_fields_list = []
    # for spacing in range(10, 41, 10):
    #     target_pattern = cgh_toolbox.generate_grid_pattern(vertical_size=1024, horizontal_size=1024, vertical_spacing=spacing, horizontal_spacing=spacing)
    #     target_fields_list.append(target_pattern)
    # target_fields = torch.stack(target_fields_list)

    # square_pattern = cgh_toolbox.energy_conserve(cgh_toolbox.generate_grid_pattern(vertical_size=256, horizontal_size=256, vertical_spacing=256, horizontal_spacing=256, line_thickness=1))
    # square_pattern = cgh_toolbox.zero_pad_to_size(square_pattern, target_height=512, target_width=512)
    # square_pattern = cgh_toolbox.add_up_side_down_replica_below(square_pattern)
    # square_pattern = cgh_toolbox.zero_pad_to_size(square_pattern, target_height=1080, target_width=1920)
    # square_pattern = torch.nn.functional.interpolate(square_pattern.expand(1, -1, -1, -1), (1080, 1920))[0]
    # target_fields_list.append(square_pattern)

    # square_dot_pattern = cgh_toolbox.generate_dotted_grid_pattern(vertical_size=256, horizontal_size=256, vertical_spacing=256, horizontal_spacing=256, line_thickness=8)
    # square_dot_pattern = cgh_toolbox.zero_pad_to_size(square_dot_pattern, target_height=1080, target_width=1920)
    # target_fields_list.append(square_dot_pattern)

    # donut_pattern = cgh_toolbox.energy_conserve(cgh_toolbox.generate_donut_pattern(radius=128, line_thickness=1))
    # donut_pattern = cgh_toolbox.zero_pad_to_size(donut_pattern, target_height=512, target_width=512)
    # donut_pattern = cgh_toolbox.add_up_side_down_replica_below(donut_pattern)
    # donut_pattern = cgh_toolbox.zero_pad_to_size(donut_pattern, target_height=1080, target_width=1920)
    # donut_pattern = torch.nn.functional.interpolate(donut_pattern.expand(1, -1, -1, -1), (1080, 1920))[0]


    # target_fields_list.append(square_pattern)
    # target_fields_list.append(donut_pattern)
    # target_fields_list = [cgh_toolbox.zero_pad_to_size(donut_pattern, target_height=1080, target_width=1920, left_shift_from_centre=shift_x) for shift_x in range(0, 200, 20)]

    # for radius in [16 + 16*x for x in range(8)]:
        # target_fields_list.append(cgh_toolbox.energy_conserve(cgh_toolbox.zero_pad_to_size(cgh_toolbox.generate_circle_pattern(radius=radius), target_height=1024, target_width=1024)))

    # target_fields = torch.stack(target_fields_list)



    # 2. Set distances according to each slice of the target (in meters)
    distances = [0.01 + i*0.01 for i in range(len(target_fields))]

    # 3. Check for mismatch between numbers of distances and images given
    if len(distances) != len(target_fields):
        raise ValueError("Different numbers of distances and images are given!")
    print("INFO: {} target fields loaded".format(len(target_fields)))

    # 4. Check if output folder exists, then save copies of target_fields
    if not os.path.isdir('Output'):
        os.makedirs('Output')
    for i, target_field in enumerate(target_fields):
        cgh_toolbox.save_image(os.path.join('Output', 'Target_field_{}'.format(i)), target_field)

    # 5. Carry out GS with SS
    time_start = time.time()
    hologram, nmse_lists_GS, time_list_GS = cgh_toolbox.gerchberg_saxton_3d_sequential_slicing(
        target_fields, distances, iteration_number=NUM_ITERATIONS, weighting=0, pitch_size=PITCH_SIZE, wavelength=WAVELENGTH)
    time_elapsed = time.time() - time_start
    to_print = "GS SS reference:\t time elapsed = {:.3f}s".format(time_elapsed)
    cgh_toolbox.save_hologram_and_its_recons(hologram, distances, "GS", recon_dynamic_range=target_fields[0].max())
    if PLOT_EACH_SLICE:
        for index, nmse_list in enumerate(nmse_lists_GS):
            plt.plot(range(1, NUM_ITERATIONS + 1), nmse_list, '-', label="GS with SS (Slice {})".format(index + 1))
            to_print += "\tNMSE_{} = {:.15e}".format(index + 1, nmse_list[-1])
        plt.xlabel("iteration number")
        plt.ylabel("NMSE")
        plt.legend()
        plt.show()
    print(to_print)

    # 6. Carry out DCGS
    time_start = time.time()
    hologram, nmse_lists_DCGS, time_list_DCGS = cgh_toolbox.gerchberg_saxton_3d_sequential_slicing(
        target_fields, distances, iteration_number=NUM_ITERATIONS, weighting=0.01, pitch_size=PITCH_SIZE, wavelength=WAVELENGTH)
    time_elapsed = time.time() - time_start
    to_print = "DCGS reference:\t time elapsed = {:.3f}s".format(time_elapsed)
    cgh_toolbox.save_hologram_and_its_recons(hologram, distances, "DCGS", recon_dynamic_range=target_fields[0].max())
    if PLOT_EACH_SLICE:
        for index, nmse_list in enumerate(nmse_lists_DCGS):
            plt.plot(range(1, NUM_ITERATIONS + 1), nmse_list, '-', label="DCGS (Slice {})".format(index + 1))
            to_print += "\tNMSE_{} = {:.15e}".format(index + 1, nmse_list[-1])
        plt.xlabel("iteration number")
        plt.ylabel("NMSE")
        plt.legend()
        plt.show()
    print(to_print)

    # 7. Carry out GD with MSE
    time_start = time.time()
    hologram, nmse_lists_GD_MSE, time_list_GD_MSE = cgh_toolbox.optim_cgh_3d(
        target_fields,
        distances,
        sequential_slicing=SEQUENTIAL_SLICING,
        save_progress=False,
        iteration_number=NUM_ITERATIONS,
        cuda=True,
        learning_rate=0.01,
        record_all_nmse=True,
        optimise_algorithm="GD",
        loss_function=torch.nn.MSELoss(reduction="sum")
    )
    time_elapsed = time.time() - time_start
    to_print = "GD with MSE:\t time elapsed = {:.3f}s".format(time_elapsed)
    cgh_toolbox.save_hologram_and_its_recons(hologram, distances, "GD_MSE", recon_dynamic_range=target_fields[0].max())
    if PLOT_EACH_SLICE:
        for index, nmse_list in enumerate(nmse_lists_GD_MSE):
            plt.plot(range(1, NUM_ITERATIONS + 1), nmse_list, '-', label="GD with MSE (Slice {})".format(index + 1))
            to_print += "\tNMSE_{} = {:.15e}".format(index + 1, nmse_list[-1])
        plt.xlabel("iteration number")
        plt.ylabel("NMSE")
        plt.legend()
        plt.show()
    print(to_print)


    # 8. Carry out LBFGS with RE
    time_start = time.time()
    hologram, nmse_lists_LBFGS_RE, time_list_LBFGS_RE = cgh_toolbox.optim_cgh_3d(
        target_fields,
        distances,
        sequential_slicing=SEQUENTIAL_SLICING,
        save_progress=False,
        iteration_number=NUM_ITERATIONS,
        cuda=True,
        learning_rate=0.1,
        record_all_nmse=True,
        optimise_algorithm="LBFGS",
        grad_history_size=10,
        loss_function=torch.nn.KLDivLoss(reduction="sum")
    )
    time_elapsed = time.time() - time_start
    to_print = "L-BFGS with RE:\t time elapsed = {:.3f}s".format(time_elapsed)
    cgh_toolbox.save_hologram_and_its_recons(hologram, distances, "LBFGS_RE", recon_dynamic_range=target_fields[0].max())
    if PLOT_EACH_SLICE:
        for index, nmse_list in enumerate(nmse_lists_LBFGS_RE):
            plt.plot(range(1, NUM_ITERATIONS + 1), nmse_list, '-', label="L-BFGS with RE (Slice {})".format(index + 1))
            to_print += "\tNMSE_{} = {:.15e}".format(index + 1, nmse_list[-1])
        plt.xlabel("iteration number")
        plt.ylabel("NMSE")
        plt.legend()
        plt.show()
    print(to_print)

    # 9. Compare maximum difference across slices
    plt.plot(range(1, NUM_ITERATIONS + 1), np.amax(nmse_lists_GS, axis=0) - np.amin(nmse_lists_GS, axis=0), label="GS")
    plt.plot(range(1, NUM_ITERATIONS + 1), np.amax(nmse_lists_DCGS, axis=0) - np.amin(nmse_lists_DCGS, axis=0), label="DCGS")
    plt.plot(range(1, NUM_ITERATIONS + 1), np.amax(nmse_lists_GD_MSE, axis=0) - np.amin(nmse_lists_GD_MSE, axis=0), label="GD_MSE")
    plt.plot(range(1, NUM_ITERATIONS + 1), np.amax(nmse_lists_LBFGS_RE, axis=0) - np.amin(nmse_lists_LBFGS_RE, axis=0), label="LBFGS_RE")
    plt.xlabel("iteration number")
    plt.ylabel("Maximum difference of NMSE")
    plt.legend()
    plt.show()

    # 10. Compare the average among slices
    plt.plot(range(1, NUM_ITERATIONS + 1), np.mean(nmse_lists_GS, axis=0), label="GS")
    plt.plot(range(1, NUM_ITERATIONS + 1), np.mean(nmse_lists_DCGS, axis=0), label="DCGS")
    plt.plot(range(1, NUM_ITERATIONS + 1), np.mean(nmse_lists_GD_MSE, axis=0), label="GD_MSE")
    plt.plot(range(1, NUM_ITERATIONS + 1), np.mean(nmse_lists_LBFGS_RE, axis=0), label="LBFGS_RE")
    plt.xlabel("iteration number")
    plt.ylabel("Average NMSE")
    plt.legend()
    plt.show()

    return


if __name__ == "__main__":
    main()
