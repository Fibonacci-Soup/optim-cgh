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
import torchvision
import matplotlib.pyplot as plt
import cgh_toolbox


def main():

    NUM_ITERATIONS = 100
    ENERGY_CONSERVATION_SCALING = 1.0
    SAVE_PROGRESS = False

    target_fields_list = []

    # Load target images
    images = [os.path.join('Target_images', 'mandrill_smaller.png')]
    for image_name in images:
        target_field = torchvision.io.read_image(image_name, torchvision.io.ImageReadMode.GRAY).to(torch.float32)
        target_field = torch.nn.functional.interpolate(target_field.expand(1, -1, -1, -1), (1024, 1280), mode='bilinear')[0]  # 1536, 2048
        target_field = torch.sqrt(target_field)
        # target_field = torchvision.transforms.functional.gaussian_blur(target_field, kernel_size=3)
        # target_field = cgh_toolbox.zero_pad_to_size(target_field, target_height=1080, target_width=1920)
        target_field_normalised = cgh_toolbox.energy_conserve(target_field, ENERGY_CONSERVATION_SCALING)
        target_fields_list.append(target_field_normalised)

    # # Generate patterned target image
    # target_field = torch.from_numpy(cgh_toolbox.generate_grid_image(vertical_size=1080, horizontal_size=1920, vertical_spacing=20, horizontal_spacing=20))
    # # target_field = torch.nn.functional.interpolate(target_field.expand(1, -1, -1, -1), (1080, 1920))[0]
    # # target_field = cgh_toolbox.zero_pad_to_size(target_field, target_height=1920, target_width=1920)
    # target_field_normalised = cgh_toolbox.energy_conserve(target_field, ENERGY_CONSERVATION_SCALING)
    # target_fields_list.append(target_field_normalised)

    target_fields = torch.stack(target_fields_list)

    # Set distances for each target image
    distances = [999999]
    # distances = [0.01 + i*SLM_PITCH_SIZE for i in range(0, NUM_SLICES, 10)]
    # distances = [0.09 + i*0.0001 for i in range(len(images))]

    # Check for mismatch between numbers of distances and images given
    if len(distances) != len(target_fields):
        raise Exception("Different numbers of distances and images are given!")

    print("INFO: {} target fields loaded".format(len(target_fields)))

    # Check if output folder exists, then save copies of target_fields
    if not os.path.isdir('Output'):
        os.makedirs('Output')
    for i, target_field in enumerate(target_fields):
        cgh_toolbox.save_image(os.path.join('Output', 'Target_field_{}'.format(i)), target_field)

    time_start = time.time()
    hologram, nmse_lists_LBFGS_RE, time_list_LBFGS_RE = cgh_toolbox.multi_frame_cgh(
        target_fields,
        distances,
        iteration_number=NUM_ITERATIONS,
        cuda=True,
        learning_rate=0.01,
        save_progress=SAVE_PROGRESS,
        optimise_algorithm="LBFGS",
        grad_history_size=10,
        num_frames=24,
        loss_function=torch.nn.KLDivLoss(reduction="sum")
        # loss_function=torch.nn.MSELoss(reduction="sum")
    )
    time_elapsed = time.time() - time_start
    to_print = "L-BFGS with RE:\t time elapsed = {:.3f}s".format(time_elapsed)
    # Save results to file
    # cgh_toolbox.save_hologram_and_its_recons(hologram, distances, "LBFGS_RE")
    if SAVE_PROGRESS:
        for index, nmse_list in enumerate(nmse_lists_LBFGS_RE):
            plt.plot(range(1, NUM_ITERATIONS + 1), nmse_list, '-', label="L-BFGS with RE (Slice {})".format(index + 1))
            to_print += "\tNMSE_{} = {:.15e}".format(index + 1, nmse_list[-1])
        plt.xlabel("iterarion(s)")
        plt.ylabel("NMSE")
        plt.legend()
        plt.show()
    print(to_print)

    return


if __name__ == "__main__":
    main()
