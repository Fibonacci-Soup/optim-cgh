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
import math
import torch
import torchvision
import matplotlib.pyplot as plt
import numpy as np
# from itertools import cycle
from cycler import cycler

from cgh_toolbox import save_image, fresnel_propergation, energy_match, gerchberg_saxton_3d_sequential_slicing, energy_conserve, lbfgs_cgh_3d

# Experimental setup - device properties


def main():
    """
    Main function of lbfgs_cgh_3d
    """

    NUM_SLICES = 858
    SEQUENTIAL_SLICING = True
    ENERGY_CONSERVATION_SCALING = 1.0
    PLOT_EACH_SLICE = False
    LEARNING_RATE = 0.05
    RECORD_ALL_NMSE = True
    GRAD_HISTORY = 900
    PITCH_SIZE = 0.00000425
    WAVELENGTH = 0.0000006607

    # Set target images
    # images = [r".\Target_images\512_A.png", r".\Target_images\512_B.png", r".\Target_images\512_C.png", r".\Target_images\512_D.png"]
    images = [r".\Target_images\1080p_A.png"]#, r".\Target_images\1080p_B.png", r".\Target_images\1080p_C.png", r".\Target_images\1080p_D.png"]

    # images = [r".\Target_images\mandrill.png", r".\Target_images\512_B.png", r".\Target_images\512_szzx.png", r".\Target_images\512_D.png"]
    # images = [r".\Target_images\512_A.png", r".\Target_images\512_B.png", r".\Target_images\512_C.png", r".\Target_images\512_D.png", r".\Target_images\512_E.png", r".\Target_images\512_F.png", r".\Target_images\512_G.png"]
    # images = [r".\Target_images\Teapot_slices\Teapot_section_{}.png".format(i) for i in range(NUM_SLICES - 1, 358, -17)]

    NUM_ITERATIONS = 100

    # Set distances for each target image
    distances = [.08]#, .09, .1, .11]
    # distances = [0.02 + i*SLM_PITCH_SIZE for i in range(0, NUM_SLICES, 10)]
    # distances = [0.08 + i*0.0005 for i in range(len(images))] #0.0000136/4.0

    # Check for mismatch between numbers of distances and images given
    if len(distances) != len(images):
        raise Exception("Different numbers of distances and images are given!")

    # Load target images
    target_fields_list = []
    for image_name in images:
        target_field = torchvision.io.read_image(image_name, torchvision.io.ImageReadMode.GRAY).to(torch.float64)
        target_field_normalised = energy_conserve(target_field, ENERGY_CONSERVATION_SCALING)
        target_fields_list.append(target_field_normalised)
    target_fields = torch.stack(target_fields_list)
    print("Data loaded, number of fields is {}".format(len(target_fields)))

    # Check if output folder exists, then save copies of target_fields
    if not os.path.isdir('Output_3D_iter'):
        os.makedirs('Output_3D_iter')
    # for i, target_field in enumerate(target_fields):
    #     save_image(r'.\Output_3D_iter\Target_field_{}'.format(i), target_field, target_fields.max())

    default_cycler = (cycler(color=['r', 'g', 'b', 'k']) + cycler(linestyle=["--", "-.", ":", "-"]))
    plt.rc('axes', prop_cycle=default_cycler)
    # Carry out GS with SS for reference
    time_start = time.time()
    nmse_lists_GS = gerchberg_saxton_3d_sequential_slicing(target_fields, distances, iteration_number=NUM_ITERATIONS, weighting=0, quickest=not RECORD_ALL_NMSE)
    time_elapsed = time.time() - time_start
    to_print = "GS reference:\t time elapsed = {:.3f}s".format(time_elapsed)

    if PLOT_EACH_SLICE:
        for index, nmse_list in enumerate(nmse_lists_GS):
            plt.plot(range(1, len(nmse_list) + 1), nmse_list, label="GS (Slice {})".format(index + 1))
            # to_print += "\tNMSE_{} = {:.15e}".format(index + 1, nmse_list[-1])
        plt.xlabel("iterarion(s)")
        plt.ylabel("NMSE of each slice")
        plt.legend()
        plt.show()
    print(to_print)

    # Carry out DCGS for reference
    time_start = time.time()
    nmse_lists_DCGS = gerchberg_saxton_3d_sequential_slicing(target_fields, distances, iteration_number=NUM_ITERATIONS, weighting=0.01, quickest=not RECORD_ALL_NMSE)
    time_elapsed = time.time() - time_start
    to_print = "DCGS reference:\t time elapsed = {:.3f}s".format(time_elapsed)

    if PLOT_EACH_SLICE:
        for index, nmse_list in enumerate(nmse_lists_DCGS):
            plt.plot(range(1, len(nmse_list) + 1), nmse_list, label="DCGS (Slice {})".format(index + 1))
            # to_print += "\tNMSE_{} = {:.15e}".format(index + 1, nmse_list[-1])
        plt.xlabel("iterarion(s)")
        plt.ylabel("NMSE of each slice")
        plt.legend()
        plt.show()
    print(to_print)

    '''
    ## Carry out GD with MSE
    time_start = time.time()
    hologram, nmse_lists_GD_MSE = lbfgs_cgh_3d(
        target_fields,
        distances,
        sequential_slicing=SEQUENTIAL_SLICING,
        save_progress=False,
        iteration_number=NUM_ITERATIONS,
        cuda=True,
        learning_rate=LEARNING_RATE,
        record_all_nmse=RECORD_ALL_NMSE,
        optimise_algorithm="GD",
        loss_function = torch.nn.MSELoss(reduction="sum")
    )
    time_elapsed = time.time() - time_start
    to_print = "GD with MSE:\t time elapsed = {:.3f}s".format(time_elapsed)

    if PLOT_EACH_SLICE:
        for index, nmse_list in enumerate(nmse_lists_GD_MSE):
            plt.plot(range(1, len(nmse_list) + 1), nmse_list, label="GD with MSE (Slice {})".format(index + 1))
            # to_print += "\tNMSE_{} = {:.15e}".format(index + 1, nmse_list[-1])
        plt.xlabel("iterarion(s)")
        plt.ylabel("NMSE of each slice")
        plt.legend()
        plt.show()
    print(to_print)



    ## Carry out GD with RE
    time_start = time.time()
    hologram, nmse_lists_GD_RE = lbfgs_cgh_3d(
        target_fields,
        distances,
        sequential_slicing=SEQUENTIAL_SLICING,
        save_progress=False,
        iteration_number=NUM_ITERATIONS,
        cuda=True,
        learning_rate=LEARNING_RATE,
        record_all_nmse=RECORD_ALL_NMSE,
        optimise_algorithm="GD",
        loss_function=torch.nn.KLDivLoss(reduction="sum")
    )
    time_elapsed = time.time() - time_start
    to_print = "GD with RE:\t time elapsed = {:.3f}s".format(time_elapsed)


    if PLOT_EACH_SLICE:
        for index, nmse_list in enumerate(nmse_lists_GD_RE):
            plt.plot(range(1, len(nmse_list) + 1), nmse_list, label="GD with RE (Slice {})".format(index + 1))
            # to_print += "\tNMSE_{} = {:.15e}".format(index + 1, nmse_list[-1])
        plt.xlabel("iterarion(s)")
        plt.ylabel("NMSE of each slice")
        plt.legend()
        plt.show()
    print(to_print)



    ## Carry out LBFGS with MSE
    time_start = time.time()
    hologram, nmse_lists_LBFGS_MSE = lbfgs_cgh_3d(
        target_fields,
        distances,
        sequential_slicing=SEQUENTIAL_SLICING,
        save_progress=False,
        iteration_number=NUM_ITERATIONS,
        cuda=True,
        learning_rate=LEARNING_RATE,
        record_all_nmse=RECORD_ALL_NMSE,
        optimise_algorithm="LBFGS",
        grad_history_size=GRAD_HISTORY,
        # loss_function=torch.nn.KLDivLoss(reduction="sum")
        loss_function=torch.nn.MSELoss(reduction="sum")
    )
    time_elapsed = time.time() - time_start
    to_print = "L-BFGS with MSE:\t time elapsed = {:.3f}s".format(time_elapsed)

    if PLOT_EACH_SLICE:
        for index, nmse_list in enumerate(nmse_lists_LBFGS_MSE):
            plt.plot(range(1, len(nmse_list) + 1), nmse_list, label="L-BFGS with MSE (Slice {})".format(index + 1))
            # to_print += "\tNMSE_{} = {:.15e}".format(index + 1, nmse_list[-1])
        plt.xlabel("iterarion(s)")
        plt.ylabel("NMSE of each slice")
        plt.legend()
        plt.show()
    print(to_print)
    '''

    # Carry out LBFGS with RE
    time_start = time.time()
    hologram, nmse_lists_LBFGS_RE = lbfgs_cgh_3d(
        target_fields,
        distances,
        sequential_slicing=SEQUENTIAL_SLICING,
        save_progress=True,
        iteration_number=NUM_ITERATIONS,
        cuda=True,
        pitch_size=PITCH_SIZE,
        wavelength=WAVELENGTH,
        learning_rate=LEARNING_RATE,
        record_all_nmse=RECORD_ALL_NMSE,
        optimise_algorithm="LBFGS",
        grad_history_size=GRAD_HISTORY,
        loss_function=torch.nn.KLDivLoss(reduction="sum")
        # loss_function=torch.nn.MSELoss(reduction="sum")
    )
    time_elapsed = time.time() - time_start
    to_print = "L-BFGS with RE:\t time elapsed = {:.3f}s".format(time_elapsed)

    if PLOT_EACH_SLICE:
        for index, nmse_list in enumerate(nmse_lists_LBFGS_RE):
            plt.plot(range(1, len(nmse_list) + 1), nmse_list, label="L-BFGS with RE (Slice {})".format(index + 1))
            # to_print += "\tNMSE_{} = {:.15e}".format(index + 1, nmse_list[-1])
        plt.xlabel("iterarion(s)")
        plt.ylabel("NMSE of each slice")
        plt.legend()
        plt.show()
    print(to_print)

    return
    default_cycler = (cycler(color=['y', 'm', 'k', 'b', 'g', 'r']) + cycler(linestyle=["--", "-", "-.", ":", ":", "-"]) + cycler(marker=[None, "+", None, None, "x", None]))
    plt.rc('axes', prop_cycle=default_cycler)
    # Compare the average among slices
    plt.plot(np.mean(nmse_lists_GS, axis=0), label="GS")
    plt.plot(np.mean(nmse_lists_DCGS, axis=0), label="DCGS")
    plt.plot(np.mean(nmse_lists_GD_MSE, axis=0), label="GD_MSE")
    plt.plot(np.mean(nmse_lists_GD_RE, axis=0), label="GD_RE")
    plt.plot(np.mean(nmse_lists_LBFGS_MSE, axis=0), label="LBFGS_MSE")
    plt.plot(np.mean(nmse_lists_LBFGS_RE, axis=0), label="LBFGS_RE")
    print(np.mean(nmse_lists_LBFGS_RE, axis=0)[-1])
    plt.xlabel("iterarion(s)")
    plt.ylabel("Average NMSE")
    plt.legend()
    plt.show()

    # Compare maximum difference across slices
    plt.plot(np.amax(nmse_lists_GS, axis=0) - np.amin(nmse_lists_GS, axis=0), label="GS")
    plt.plot(np.amax(nmse_lists_DCGS, axis=0) - np.amin(nmse_lists_DCGS, axis=0), label="DCGS")
    plt.plot(np.amax(nmse_lists_GD_MSE, axis=0) - np.amin(nmse_lists_GD_MSE, axis=0), label="GD_MSE")
    plt.plot(np.amax(nmse_lists_GD_RE, axis=0) - np.amin(nmse_lists_GD_RE, axis=0), label="GD_RE")
    plt.plot(np.amax(nmse_lists_LBFGS_MSE, axis=0) - np.amin(nmse_lists_LBFGS_MSE, axis=0), label="LBFGS_MSE")
    plt.plot(np.amax(nmse_lists_LBFGS_RE, axis=0) - np.amin(nmse_lists_LBFGS_RE, axis=0), label="LBFGS_RE")
    plt.xlabel("iterarion(s)")
    plt.ylabel("Maximum difference of NMSE")
    plt.legend()
    plt.show()

    return

    if SEQUENTIAL_SLICING:
        plt.title("Iterations during optimisation of CGH with sequantial slicing technique")
    else:
        plt.title("Iterations during optimisation of CGH without sequantial slicing technique")

    plt.xlabel("iterarion(s)")
    plt.ylabel("NMSE")
    plt.legend()
    # plt.grid()
    # plt.subplots_adjust(left=0.03, right=0.99, top=0.97, bottom=0.05)
    plt.show()


if __name__ == "__main__":
    main()
