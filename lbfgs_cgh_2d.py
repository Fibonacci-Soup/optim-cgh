#!/usr/bin/env python3
"""
Copyright(c) 2022 Jinze Sha (js2294@cam.ac.uk)
Centre for Molecular Materials, Photonics and Electronics, University of Cambridge
All Rights Reserved.

This is the python script for Limited-memory Broyden-Fletcher-Goldfarb-Shanno (L-BFGS)
optimisation of Computer-Generated Hologram (CGH) whose reconstruction is a 2D target image.
"""

import os
import time
import math
import torch
import torchvision
from cgh_toolbox import save_image, fraunhofer_propergation, fresnel_propergation, energy_conserve

# Experimental setup - device properties
SLM_PHASE_RANGE = 2 * math.pi
SLM_PITCH_SIZE = 0.0000136
LASER_WAVELENGTH = 0.000000532
ENERGY_CONSERVATION_SCALING = 1.0

def lbfgs_cgh_2d(target_field, distance=1, wavelength=0.000000532, pitch_size=0.0000136,
                 save_progress=False, iteration_number=20, cuda=False, learning_rate=0.1,
                 propagation_function=fraunhofer_propergation,
                 loss_function=torch.nn.MSELoss(reduction="sum")):
    """
    Carry out L-BFGS optimisation of CGH for a 2D target image

    :param target_field: tensor for target image
    :param distance: image distance (for Fresnel propagation only)
    :param wavelength: wavelength of the light source
    :param pitch_size: pitch size of the spatial light modulator (SLM)
    :param save_progress: decide whether to save progress of every iteration to files
    :param iteration_number: number of iterations
    :param cuda: decide whether to use CUDA, use CPU otherwise
    :param learning_rate: set the parameter 'lr' of torch.optim
    :param propagation_function: propagation function to reconstruct from hologram
    :param loss_function: the objective function to minimise
    :returns: resultant hologram
    """
    torch.cuda.empty_cache()
    torch.manual_seed(0)
    device = torch.device("cuda" if cuda else "cpu")

    target_field = target_field.to(device)

    # Fixed unit amplitude
    amplitude = torch.ones(target_field.size(), requires_grad=False).to(torch.float64).to(device)
    # Random initial phase within [-pi, pi]
    phase = ((torch.rand(target_field.size()) * 2 - 1) * math.pi).to(torch.float64).detach().to(device).requires_grad_()
    optimizer = torch.optim.LBFGS([{'params': [phase]}], lr=learning_rate, max_iter=200)

    if save_progress:
        nmse_list = []

    for i in range(iteration_number):
        optimizer.zero_grad()

        # Propagate hologram to reconstruction plane
        hologram = amplitude * torch.exp(1j * phase)
        reconstruction_abs = propagation_function(hologram, distance, pitch_size, wavelength).abs()
        reconstruction_normalised = energy_conserve(reconstruction_abs, ENERGY_CONSERVATION_SCALING)

        # Save hologram and reconstruction every iteration, if needed
        if save_progress:
            # binary_phase_hologram = torch.where(phase > 0, 1, 0)
            multi_phase_hologram = phase % SLM_PHASE_RANGE / SLM_PHASE_RANGE
            save_image(r'.\Output_2D_iter\holo_i_{}'.format(i), multi_phase_hologram.detach().cpu(), 1.0)
            save_image(r'.\Output_2D_iter\recon_i_{}'.format(i), reconstruction_normalised.detach().cpu(), target_field.detach().cpu().max())
            nmse_value = (torch.nn.MSELoss(reduction="mean")(reconstruction_normalised, target_field)).item() / (target_field**2).sum()
            nmse_list.append(nmse_value.data.tolist())

        # Calculate loss and step optimizer
        loss = loss_function(torch.flatten(reconstruction_normalised).expand(1, -1), torch.flatten(target_field).expand(1, -1))  # flatten 2D images into 1D array
        loss.backward(retain_graph=True)

        def closure():
            return loss
        optimizer.step(closure)

    if save_progress:
        with open(r'Output_2D_iter\NMSE_History.txt', 'w') as nmse_file:
            for each_nmse in nmse_list:
                nmse_file.write(str(each_nmse) + '\n')

    torch.no_grad()
    hologram = amplitude * torch.exp(1j * phase)
    return hologram.detach()


def main():
    """
    Main function of lbfgs_cgh_2d
    """
    target_field = torchvision.io.read_image(r".\Target_images\mandrill.png", torchvision.io.ImageReadMode.GRAY).to(torch.float64)
    target_field_normalised = energy_conserve(target_field, ENERGY_CONSERVATION_SCALING)
    print("save_image_dynamic_range", target_field_normalised.max().tolist())

    # Check if output folder exists, then save a copy of target_field
    if not os.path.isdir('Output_2D_iter'):
        os.makedirs('Output_2D_iter')
    save_image(r'.\Output_2D_iter\Target_field_normalised', target_field_normalised)

    # Carry out optimisation
    time_start = time.time()
    hologram = lbfgs_cgh_2d(
        target_field_normalised,
        distance=0.5,
        wavelength=LASER_WAVELENGTH,
        pitch_size=SLM_PITCH_SIZE,
        save_progress=False,
        iteration_number=20,
        cuda=True,
        learning_rate=0.1,
        propagation_function=fraunhofer_propergation,  # Uncomment to choose Fraunhofer propagation
        # propagation_function=fresnel_propergation, # Uncomment to choose Fresnel Propagation
        # loss_function=torch.nn.MSELoss(reduction="sum")  # Uncomment to choose MSE loss
        loss_function=torch.nn.CrossEntropyLoss(label_smoothing=0.0, reduction="sum")  # Uncommnet to choose CE loss
    )
    print("time = ", time.time() - time_start)

    # Save final hologram
    multi_phase_hologram = hologram.angle() % SLM_PHASE_RANGE / SLM_PHASE_RANGE
    save_image(r'.\Output_2D_iter\LBFGS_holo', multi_phase_hologram.detach().cpu(), 1.0)

    # Save reconstructions at different distances to check defocus
    for distance in [0.1, 0.5, 1, 5, 10, 100, 10**9]:
        reconstruction_abs = fresnel_propergation(hologram, distance, SLM_PITCH_SIZE, LASER_WAVELENGTH).abs()
        reconstruction_normalised = energy_conserve(reconstruction_abs, ENERGY_CONSERVATION_SCALING)
        save_image(r'.\Output_2D_iter\LBFGS_recon_defocused_at_{}'.format(distance), reconstruction_normalised.detach().cpu(), target_field_normalised.max())


if __name__ == "__main__":
    main()
