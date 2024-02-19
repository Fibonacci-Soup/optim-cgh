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


def main():
    NUM_ITERATIONS = 100
    SAVE_PROGRESS = False

    # 1.A Option1: Load target images from files (please use PNG format with zero compression, even although PNG compression is lossless)
    # images = [os.path.join('Target_images', x) for x in ['A.png', 'B.png', 'C.png', 'D.png']]
    # target_fields = cgh_toolbox.load_target_images(images, energy_conserv_scaling=ENERGY_CONSERVATION_SCALING)


    # 1.B Option2: Generate target fields of specific patterns (e.g. grid pattern here)
    target_fields_list = []

    # square_pattern = cgh_toolbox.energy_conserve(cgh_toolbox.generate_grid_pattern(vertical_size=256, horizontal_size=256, vertical_spacing=256, horizontal_spacing=256, line_thickness=1))
    # square_pattern = cgh_toolbox.zero_pad_to_size(square_pattern, target_height=512, target_width=512)
    # square_pattern = cgh_toolbox.add_up_side_down_replica_below(square_pattern)
    # square_pattern = cgh_toolbox.zero_pad_to_size(square_pattern, target_height=1024, target_width=1024)

    # square_dot_pattern = cgh_toolbox.generate_dotted_grid_pattern(vertical_size=256, horizontal_size=256, vertical_spacing=256, horizontal_spacing=256, line_thickness=8)
    # square_dot_pattern = cgh_toolbox.zero_pad_to_size(square_dot_pattern, target_height=1024, target_width=1024)
    # target_fields_list.append(square_dot_pattern)

    # donut_pattern = cgh_toolbox.energy_conserve(cgh_toolbox.generate_donut_pattern(radius=128, line_thickness=1))
    # donut_pattern = cgh_toolbox.zero_pad_to_size(donut_pattern, target_height=512, target_width=512)
    # donut_pattern = cgh_toolbox.add_up_side_down_replica_below(donut_pattern)
    # donut_pattern = cgh_toolbox.zero_pad_to_size(donut_pattern, target_height=1024, target_width=1024)

    # target_fields_list.append(square_pattern)
    # target_fields_list.append(donut_pattern)

    for radius in [16 + 16*x for x in range(8)]:
        target_fields_list.append(cgh_toolbox.energy_conserve(cgh_toolbox.zero_pad_to_size(cgh_toolbox.generate_circle_pattern(radius=radius), target_height=1024, target_width=1024)))

    target_fields = torch.stack(target_fields_list)


    # 2. Set distances according to each slice of the target (in meters)
    distances = [.19 - 0.01*x for x in range(8)]
    # distances = [0.1, 0.2, 0.3, 0.4]
    # distances = [0.15 * x / (0.15 - x) for x in distances]
    # distances = [0.01 + i*SLM_PITCH_SIZE for i in range(0, NUM_SLICES, 10)]
    # distances = [0.09 + i*0.0001 for i in range(len(images))]

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
    _, nmse_lists_LBFGS_RE, _ = cgh_toolbox.multi_frame_cgh(
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
