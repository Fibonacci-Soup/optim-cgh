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
from cgh_toolbox import save_image, fresnel_propergation, energy_conserve

# Experimental setup - device properties
SLM_PHASE_RANGE = 2 * math.pi
SLM_PITCH_SIZE = 0.0000136
LASER_WAVELENGTH = 0.000000532
ENERGY_CONSERVATION_SCALING = 1.0

def lbfgs_cgh_3d(target_fields, distances, each_slice_in_turn=False,
                 wavelength=0.000000532, pitch_size=0.0000136, save_progress=False, iteration_number=20,
                 cuda=False, learning_rate=0.1, loss_function=torch.nn.MSELoss(reduction="sum")):
    """
    Carry out L-BFGS optimisation of CGH for a 3D target consisted of multiple slices of 2D images at different distances.
    If each_slice_in_turn is True, Loss is calculated for reconstructions in all distances.
    If each_slice_in_turn is False, Loss is calculated for reconstruction in each distance in turn for each iteration.

    :param target_fields: tensor for target images
    :param distances: image distances
    :param each_slice_in_turn: decide whether to calculate loss function for each slice in turn instead of all slices
    :param wavelength: wavelength of the light source
    :param pitch_size: pitch size of the spatial light modulator (SLM)
    :param save_progress: decide whether to save progress of every iteration to files
    :param iteration_number: number of iterations
    :param cuda: decide whether to use CUDA, use CPU otherwise
    :param learning_rate: set the parameter 'lr' of torch.optim
    :param loss_function: the objective function to minimise
    :returns: resultant hologram
    """
    torch.cuda.empty_cache()
    torch.manual_seed(0)
    device = torch.device("cuda" if cuda else "cpu")
    target_fields = target_fields.to(device)
    # Fixed unit amplitude
    amplitude = torch.ones(target_fields[0].size(), requires_grad=False).to(torch.float64).to(device)
    # Random initial phase within [-pi, pi]
    phase = ((torch.rand(target_fields[0].size()) * 2 - 1) * math.pi).to(torch.float64).detach().to(device).requires_grad_()
    optimizer = torch.optim.LBFGS([{'params': [phase]}], lr=learning_rate, max_iter=200)

    for i in range(iteration_number):
        optimizer.zero_grad()
        hologram = amplitude * torch.exp(1j * phase)

        if each_slice_in_turn:
            # Propagate hologram for one distance only
            slice_number = i % len(target_fields)
            reconstruction_abs = fresnel_propergation(hologram, distances[slice_number], pitch_size, wavelength).abs()
            reconstruction_normalised = energy_conserve(reconstruction_abs, ENERGY_CONSERVATION_SCALING)

            # Calculate loss for the single slice
            loss = loss_function(torch.flatten(reconstruction_normalised).expand(1, -1),
                                 torch.flatten(target_fields[slice_number]).expand(1, -1))
        else:
            # Propagate hologram for all distances
            reconstructions_list = []
            for distance in distances:
                reconstruction_abs = fresnel_propergation(hologram, distance=distance, pitch_size=pitch_size, wavelength=wavelength).abs()
                reconstruction_normalised = energy_conserve(reconstruction_abs, ENERGY_CONSERVATION_SCALING)
                reconstructions_list.append(reconstruction_normalised)
            reconstructions = torch.stack(reconstructions_list)

            # Calculate loss for all slices (stacked in reconstructions)
            loss = loss_function(torch.flatten(reconstructions).expand(1, -1),
                                 torch.flatten(target_fields).expand(1, -1))

        loss.backward(retain_graph=True)

        def closure():
            return loss
        optimizer.step(closure)

        # Save hologram and reconstruction every iteration, if save_progress is True
        if save_progress:
            # Save hologram
            multi_phase_hologram = phase % SLM_PHASE_RANGE / SLM_PHASE_RANGE
            save_image(r'.\Output_3D_iter\holo_i_{}'.format(i), multi_phase_hologram.detach().cpu(), 1.0)
            # Save reconstructions at all distances
            for index, distance in enumerate(distances):
                reconstruction_abs = fresnel_propergation(hologram, distance, SLM_PITCH_SIZE, LASER_WAVELENGTH).abs()
                reconstruction_normalised = energy_conserve(reconstruction_abs, ENERGY_CONSERVATION_SCALING)
                save_image(r'.\Output_3D_iter\recon_i_{}_d_{}'.format(i, index), reconstruction_normalised.detach().cpu(), target_fields.detach().cpu().max())

    torch.no_grad()
    hologram = amplitude * torch.exp(1j * phase)
    return hologram.detach()


def main():
    """
    Main function of lbfgs_cgh_3d
    """

    # Set distances for each target image
    distances = [0.1, 0.2, 0.3, 0.4]

    # Set target images
    images = [r".\Target_images\A.png", r".\Target_images\B.png", r".\Target_images\C.png", r".\Target_images\D.png"]
    # images = [r".\Target_images\grey-scale-test.png", r".\Target_images\szzx1.png", r".\Target_images\guang.png", r".\Target_images\mandrill1.png"]

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
    print("save_image_dynamic_range", target_fields.max().tolist())

    # Check if output folder exists, then save copies of target_fields
    if not os.path.isdir('Output_3D_iter'):
        os.makedirs('Output_3D_iter')
    for i, target_field in enumerate(target_fields):
        save_image(r'.\Output_3D_iter\Target_field_d_{}'.format(distances[i]), target_field, target_fields.max())

    # Carry out optimisation
    time_start = time.time()
    hologram = lbfgs_cgh_3d(
        target_fields,
        distances,
        each_slice_in_turn=True,
        wavelength=LASER_WAVELENGTH,
        pitch_size=SLM_PITCH_SIZE,
        save_progress=False,
        iteration_number=100,
        cuda=True,
        learning_rate=0.1,
        # loss_function = torch.nn.MSELoss(reduction="sum") # Uncomment to choose MSE loss
        loss_function=torch.nn.CrossEntropyLoss(label_smoothing=0.0, reduction="sum")  # Uncommnet to choos CE loss
    )
    print("time = ", time.time() - time_start)

    # Save final hologram
    phase_hologram = hologram.angle()
    multi_phase_hologram = phase_hologram % SLM_PHASE_RANGE / SLM_PHASE_RANGE
    save_image(r'.\Output_3D_iter\LBFGS_holo_distances_{}'.format(distances), multi_phase_hologram.detach().cpu(), 1.0)

    # Save reconstructions at all distances
    for index, distance in enumerate(distances):
        reconstruction_abs = fresnel_propergation(hologram, distance, SLM_PITCH_SIZE, LASER_WAVELENGTH).abs()
        reconstruction_normalised = energy_conserve(reconstruction_abs, ENERGY_CONSERVATION_SCALING)
        save_image(r'.\Output_3D_iter\LBFGS_recon_defocused_at_{}'.format(distance), reconstruction_normalised.detach().cpu())


if __name__ == "__main__":
    main()
