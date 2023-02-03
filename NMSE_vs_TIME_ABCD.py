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
    LEARNING_RATE = 0.1
    RECORD_ALL_NMSE = True
    GRAD_HISTORY = 900
    TIME_LIMIT = 2.0

    ## Set target images
    # images = [r".\Target_images\512_A.png", r".\Target_images\512_B.png", r".\Target_images\512_C.png", r".\Target_images\512_D.png"]
    # images = [r".\Target_images\mandrill.png", r".\Target_images\512_B.png", r".\Target_images\512_szzx.png", r".\Target_images\512_D.png"]
    # images = [r".\Target_images\512_A.png", r".\Target_images\512_B.png", r".\Target_images\512_C.png", r".\Target_images\512_D.png", r".\Target_images\512_E.png", r".\Target_images\512_F.png", r".\Target_images\512_G.png"]
    images = [r".\Target_images\Teapot_slices\Teapot_720p_section_{}.png".format(i) for i in range(NUM_SLICES - 1, 358, -17)]


    NUM_ITERATIONS = 1000

    ## Set distances for each target image
    # distances = [.01, .02, .03, .04]
    # distances = [0.02 + i*SLM_PITCH_SIZE for i in range(0, NUM_SLICES, 10)]
    distances = [0.02 + i*0.0000136/4.0 for i in range(len(images))]

    ## Check for mismatch between numbers of distances and images given
    if len(distances) != len(images):
        raise Exception("Different numbers of distances and images are given!")

    ## Load target images
    target_fields_list = []
    for image_name in images:
        target_field = torchvision.io.read_image(image_name, torchvision.io.ImageReadMode.GRAY).to(torch.float64)
        target_field_normalised = energy_conserve(target_field, ENERGY_CONSERVATION_SCALING)
        target_fields_list.append(target_field_normalised)
    target_fields = torch.stack(target_fields_list)
    print("Data loaded, number of fields is {}".format(len(target_fields)))

    ## Check if output folder exists, then save copies of target_fields
    if not os.path.isdir('Output_3D_iter'):
        os.makedirs('Output_3D_iter')
    for i, target_field in enumerate(target_fields):
        save_image(r'.\Output_3D_iter\Target_field_{}'.format(i), target_field, target_fields.max())

    ## Warm up GS
    gerchberg_saxton_3d_sequential_slicing(target_fields, distances, iteration_number=NUM_ITERATIONS, weighting=0, quickest=True, time_limit=TIME_LIMIT)

    ## Carry out GS with SS for reference
    _, time_list_GS = gerchberg_saxton_3d_sequential_slicing(target_fields, distances, iteration_number=NUM_ITERATIONS, weighting=0, quickest=True, time_limit=TIME_LIMIT)
    nmse_lists_GS, _ = gerchberg_saxton_3d_sequential_slicing(target_fields, distances, iteration_number=len(time_list_GS), weighting=0, quickest=False)

    ## Carry out DCGS for reference
    _, time_list_DCGS = gerchberg_saxton_3d_sequential_slicing(target_fields, distances, iteration_number=NUM_ITERATIONS, weighting=0.01, quickest=True, time_limit=TIME_LIMIT)
    nmse_lists_DCGS, _ = gerchberg_saxton_3d_sequential_slicing(target_fields, distances, iteration_number=len(time_list_DCGS), weighting=0.01, quickest=False)

    ## Warm up GD
    lbfgs_cgh_3d(target_fields, distances, sequential_slicing=True, save_progress=False, iteration_number=NUM_ITERATIONS, cuda=True, learning_rate=LEARNING_RATE, record_all_nmse=False, optimise_algorithm="GD", loss_function = torch.nn.KLDivLoss(reduction="sum"), time_limit = TIME_LIMIT)

    ## Carry out GD SS RE
    _, _, time_list_GD_SS_RE = lbfgs_cgh_3d(target_fields, distances, sequential_slicing=True, save_progress=False, iteration_number=NUM_ITERATIONS, cuda=True, learning_rate=LEARNING_RATE, record_all_nmse=False, optimise_algorithm="GD", loss_function = torch.nn.KLDivLoss(reduction="sum"), time_limit = TIME_LIMIT)
    _, nmse_lists_GD_SS_RE, _ = lbfgs_cgh_3d(target_fields, distances, sequential_slicing=True, save_progress=False, iteration_number=len(time_list_GD_SS_RE), cuda=True, learning_rate=LEARNING_RATE, record_all_nmse=True, optimise_algorithm="GD", loss_function = torch.nn.KLDivLoss(reduction="sum"))

    ## Carry out GD SS MSE
    _, _, time_list_GD_SS_MSE = lbfgs_cgh_3d(target_fields, distances, sequential_slicing=True, save_progress=False, iteration_number=NUM_ITERATIONS, cuda=True, learning_rate=LEARNING_RATE, record_all_nmse=False, optimise_algorithm="GD", loss_function = torch.nn.MSELoss(reduction="sum"), time_limit = TIME_LIMIT)
    _, nmse_lists_GD_SS_MSE, _ = lbfgs_cgh_3d(target_fields, distances, sequential_slicing=True, save_progress=False, iteration_number=len(time_list_GD_SS_MSE), cuda=True, learning_rate=LEARNING_RATE, record_all_nmse=True, optimise_algorithm="GD", loss_function = torch.nn.MSELoss(reduction="sum"))

    ## Carry out GD SoL RE
    _, _, time_list_GD_SoL_RE = lbfgs_cgh_3d(target_fields, distances, sequential_slicing=False, save_progress=False, iteration_number=NUM_ITERATIONS, cuda=True, learning_rate=LEARNING_RATE, record_all_nmse=False, optimise_algorithm="GD", loss_function = torch.nn.KLDivLoss(reduction="sum"), time_limit = TIME_LIMIT)
    _, nmse_lists_GD_SoL_RE, _ = lbfgs_cgh_3d(target_fields, distances, sequential_slicing=False, save_progress=False, iteration_number=len(time_list_GD_SoL_RE), cuda=True, learning_rate=LEARNING_RATE, record_all_nmse=True, optimise_algorithm="GD", loss_function = torch.nn.KLDivLoss(reduction="sum"))

    ## Carry out GD SoL MSE
    _, _, time_list_GD_SoL_MSE = lbfgs_cgh_3d(target_fields, distances, sequential_slicing=False, save_progress=False, iteration_number=NUM_ITERATIONS, cuda=True, learning_rate=LEARNING_RATE, record_all_nmse=False, optimise_algorithm="GD", loss_function = torch.nn.MSELoss(reduction="sum"), time_limit = TIME_LIMIT)
    _, nmse_lists_GD_SoL_MSE, _ = lbfgs_cgh_3d(target_fields, distances, sequential_slicing=False, save_progress=False, iteration_number=len(time_list_GD_SoL_MSE), cuda=True, learning_rate=LEARNING_RATE, record_all_nmse=True, optimise_algorithm="GD", loss_function = torch.nn.MSELoss(reduction="sum"))

    ## Warm up LBFGS
    lbfgs_cgh_3d(target_fields, distances, sequential_slicing=True, save_progress=False, iteration_number=NUM_ITERATIONS, cuda=True, learning_rate=LEARNING_RATE, record_all_nmse=False, optimise_algorithm="LBFGS", loss_function = torch.nn.KLDivLoss(reduction="sum"), time_limit = TIME_LIMIT)
    ## Carry out LBFGS SS RE
    _, _, time_list_LBFGS_SS_RE = lbfgs_cgh_3d(target_fields, distances, sequential_slicing=True, save_progress=False, iteration_number=NUM_ITERATIONS, cuda=True, learning_rate=LEARNING_RATE, record_all_nmse=False, optimise_algorithm="LBFGS", loss_function = torch.nn.KLDivLoss(reduction="sum"), time_limit = TIME_LIMIT)
    _, nmse_lists_LBFGS_SS_RE, _ = lbfgs_cgh_3d(target_fields, distances, sequential_slicing=True, save_progress=False, iteration_number=len(time_list_LBFGS_SS_RE), cuda=True, learning_rate=LEARNING_RATE, record_all_nmse=True, optimise_algorithm="LBFGS", grad_history_size=GRAD_HISTORY,loss_function = torch.nn.KLDivLoss(reduction="sum"))

    ## Carry out LBFGS SS MSE
    _, _, time_list_LBFGS_SS_MSE = lbfgs_cgh_3d(target_fields, distances, sequential_slicing=True, save_progress=False, iteration_number=NUM_ITERATIONS, cuda=True, learning_rate=LEARNING_RATE, record_all_nmse=False, optimise_algorithm="LBFGS", loss_function = torch.nn.MSELoss(reduction="sum"), time_limit = TIME_LIMIT)
    _, nmse_lists_LBFGS_SS_MSE, _ = lbfgs_cgh_3d(target_fields, distances, sequential_slicing=True, save_progress=False, iteration_number=len(time_list_LBFGS_SS_MSE), cuda=True, learning_rate=LEARNING_RATE, record_all_nmse=True, optimise_algorithm="LBFGS", grad_history_size=GRAD_HISTORY,loss_function = torch.nn.MSELoss(reduction="sum"))

    ## Carry out LBFGS SoL RE
    _, _, time_list_LBFGS_SoL_RE = lbfgs_cgh_3d(target_fields, distances, sequential_slicing=False, save_progress=False, iteration_number=NUM_ITERATIONS, cuda=True, learning_rate=LEARNING_RATE, record_all_nmse=False, optimise_algorithm="LBFGS", loss_function = torch.nn.KLDivLoss(reduction="sum"), time_limit = TIME_LIMIT)
    _, nmse_lists_LBFGS_SoL_RE, _ = lbfgs_cgh_3d(target_fields, distances, sequential_slicing=False, save_progress=False, iteration_number=len(time_list_LBFGS_SoL_RE), cuda=True, learning_rate=LEARNING_RATE, record_all_nmse=True, optimise_algorithm="LBFGS", grad_history_size=GRAD_HISTORY,loss_function = torch.nn.KLDivLoss(reduction="sum"))

    ## Carry out LBFGS SoL MSE
    _, _, time_list_LBFGS_SoL_MSE = lbfgs_cgh_3d(target_fields, distances, sequential_slicing=False, save_progress=False, iteration_number=NUM_ITERATIONS, cuda=True, learning_rate=LEARNING_RATE, record_all_nmse=False, optimise_algorithm="LBFGS", loss_function = torch.nn.MSELoss(reduction="sum"), time_limit = TIME_LIMIT)
    _, nmse_lists_LBFGS_SoL_MSE, _ = lbfgs_cgh_3d(target_fields, distances, sequential_slicing=False, save_progress=False, iteration_number=len(time_list_LBFGS_SoL_MSE), cuda=True, learning_rate=LEARNING_RATE, record_all_nmse=True, optimise_algorithm="LBFGS", grad_history_size=GRAD_HISTORY,loss_function = torch.nn.MSELoss(reduction="sum"))



    default_cycler = (cycler(color=['0.5', '0.5', 'r', 'g', 'b', 'm', 'r', 'g', 'b', 'm']) + cycler(linestyle=["--",":","--",":","-.","-","--",":","-.","-"]) + cycler(marker=[None,None,"|","|","|","|","x","x","x","x"]))
    plt.rc('axes', prop_cycle=default_cycler)

    plt.plot(time_list_GS, np.mean(nmse_lists_GS, axis=0), label="GS_SS")
    plt.plot(time_list_DCGS, np.mean(nmse_lists_DCGS, axis=0), label="DCGS_SS")
    plt.plot(time_list_GD_SS_MSE, np.mean(nmse_lists_GD_SS_MSE, axis=0), label="GD_SS_MSE")
    plt.plot(time_list_GD_SS_RE, np.mean(nmse_lists_GD_SS_RE, axis=0), label="GD_SS_RE")
    plt.plot(time_list_GD_SoL_MSE, np.mean(nmse_lists_GD_SoL_MSE, axis=0), label="GD_SoL_MSE")
    plt.plot(time_list_GD_SoL_RE, np.mean(nmse_lists_GD_SoL_RE, axis=0), label="GD_SoL_RE")
    plt.plot(time_list_LBFGS_SS_MSE, np.mean(nmse_lists_LBFGS_SS_MSE, axis=0), label="LBFGS_SS_MSE")
    plt.plot(time_list_LBFGS_SS_RE, np.mean(nmse_lists_LBFGS_SS_RE, axis=0), label="LBFGS_SS_RE")
    plt.plot(time_list_LBFGS_SoL_MSE, np.mean(nmse_lists_LBFGS_SoL_MSE, axis=0), label="LBFGS_SoL_MSE")
    plt.plot(time_list_LBFGS_SoL_RE, np.mean(nmse_lists_LBFGS_SoL_RE, axis=0), label="LBFGS_SoL_RE")
    plt.xlim(0, TIME_LIMIT)
    plt.xlabel("time (s)")
    plt.ylabel("Average NMSE")
    plt.legend()
    plt.subplots_adjust(left=0.1, right=0.9, top=0.9, bottom=0.1)
    plt.show()


    plt.plot(time_list_GS, np.amax(nmse_lists_GS, axis=0) - np.amin(nmse_lists_GS, axis=0), label="GS_SS")
    plt.plot(time_list_DCGS, np.amax(nmse_lists_DCGS, axis=0) - np.amin(nmse_lists_DCGS, axis=0), label="DCGS_SS")
    plt.plot(time_list_GD_SS_MSE, np.amax(nmse_lists_GD_SS_MSE, axis=0) - np.amin(nmse_lists_GD_SS_MSE, axis=0), label="GD_SS_MSE")
    plt.plot(time_list_GD_SS_RE, np.amax(nmse_lists_GD_SS_RE, axis=0) - np.amin(nmse_lists_GD_SS_RE, axis=0), label="GD_SS_RE")
    plt.plot(time_list_GD_SoL_MSE, np.amax(nmse_lists_GD_SoL_MSE, axis=0) - np.amin(nmse_lists_GD_SoL_MSE, axis=0), label="GD_SoL_MSE")
    plt.plot(time_list_GD_SoL_RE, np.amax(nmse_lists_GD_SoL_RE, axis=0) - np.amin(nmse_lists_GD_SoL_RE, axis=0), label="GD_SoL_RE")
    plt.plot(time_list_LBFGS_SS_MSE, np.amax(nmse_lists_LBFGS_SS_MSE, axis=0) - np.amin(nmse_lists_LBFGS_SS_MSE, axis=0), label="LBFGS_SS_MSE")
    plt.plot(time_list_LBFGS_SS_RE, np.amax(nmse_lists_LBFGS_SS_RE, axis=0) - np.amin(nmse_lists_LBFGS_SS_RE, axis=0), label="LBFGS_SS_RE")
    plt.plot(time_list_LBFGS_SoL_MSE, np.amax(nmse_lists_LBFGS_SoL_MSE, axis=0) - np.amin(nmse_lists_LBFGS_SoL_MSE, axis=0), label="LBFGS_SoL_MSE")
    plt.plot(time_list_LBFGS_SoL_RE, np.amax(nmse_lists_LBFGS_SoL_RE, axis=0) - np.amin(nmse_lists_LBFGS_SoL_RE, axis=0), label="LBFGS_SoL_RE")
    plt.xlim(0, TIME_LIMIT)
    plt.xlabel("time (s)")
    plt.ylabel("Maximum difference of NMSE")
    plt.legend()
    plt.subplots_adjust(left=0.1, right=0.9, top=0.9, bottom=0.1)
    plt.show()


    plt.plot(time_list_GS, np.std(nmse_lists_GS, axis=0), label="GS_SS")
    plt.plot(time_list_DCGS, np.std(nmse_lists_DCGS, axis=0), label="DCGS_SS")
    plt.plot(time_list_GD_SS_MSE, np.std(nmse_lists_GD_SS_MSE, axis=0), label="GD_SS_MSE")
    plt.plot(time_list_GD_SS_RE, np.std(nmse_lists_GD_SS_RE, axis=0), label="GD_SS_RE")
    plt.plot(time_list_GD_SoL_MSE, np.std(nmse_lists_GD_SoL_MSE, axis=0), label="GD_SoL_MSE")
    plt.plot(time_list_GD_SoL_RE, np.std(nmse_lists_GD_SoL_RE, axis=0), label="GD_SoL_RE")
    plt.plot(time_list_LBFGS_SS_MSE, np.std(nmse_lists_LBFGS_SS_MSE, axis=0), label="LBFGS_SS_MSE")
    plt.plot(time_list_LBFGS_SS_RE, np.std(nmse_lists_LBFGS_SS_RE, axis=0), label="LBFGS_SS_RE")
    plt.plot(time_list_LBFGS_SoL_MSE, np.std(nmse_lists_LBFGS_SoL_MSE, axis=0), label="LBFGS_SoL_MSE")
    plt.plot(time_list_LBFGS_SoL_RE, np.std(nmse_lists_LBFGS_SoL_RE, axis=0), label="LBFGS_SoL_RE")
    plt.xlim(0, TIME_LIMIT)
    plt.xlabel("time (s)")
    plt.ylabel("Standard deviation of NMSE across all slices")
    plt.legend()
    plt.subplots_adjust(left=0.1, right=0.9, top=0.9, bottom=0.1)
    plt.show()
    return




if __name__ == "__main__":
    main()
