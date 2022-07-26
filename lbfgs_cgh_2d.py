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
import matplotlib.pyplot as plt
from cgh_toolbox import save_image, fraunhofer_propergation, fresnel_propergation, energy_conserve, gerchberg_saxton

# Experimental setup - device properties
SLM_PHASE_RANGE = 2 * math.pi
SLM_PITCH_SIZE = 0.0000136
LASER_WAVELENGTH = 0.000000532

ENERGY_CONSERVATION_SCALING = 1.0
NUM_ITERATIONS = 50
LEARNING_RATE = 0.1


def optim_cgh_2d(target_field, distance=1, wavelength=0.000000532, pitch_size=0.0000136,
                 save_progress=False, iteration_number=20, cuda=False, learning_rate=0.1,
                 propagation_function=fraunhofer_propergation, optimise_algorithm="LBFGS",
                 loss_function=torch.nn.MSELoss(reduction="sum")):
    """
    Carry out optimisation of CGH for a 2D target image

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
    if optimise_algorithm.lower() == "lbfgs":
        optimiser = torch.optim.LBFGS([{'params': [phase]}], lr=learning_rate, max_iter=1000)
    elif optimise_algorithm.lower() == "sgd":
        optimiser = torch.optim.SGD([phase], lr=learning_rate)
    elif optimise_algorithm.lower() == "adam":
        optimiser = torch.optim.Adam([phase], lr=learning_rate)
    else:
        raise Exception("Optimiser is not recognised!")


    nmse_list = []

    for i in range(iteration_number):
        optimiser.zero_grad()

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

        # Calculate loss and step optimiser
        nmse_value = (torch.nn.MSELoss(reduction="mean")(reconstruction_normalised, target_field)).item() / (target_field**2).sum()
        nmse_list.append(nmse_value.data.tolist())
        # loss = loss_function(torch.flatten(reconstruction_normalised).expand(1, -1), torch.flatten(target_field).expand(1, -1))  # flatten 2D images into 1D array
        loss = loss_function(torch.flatten(reconstruction_normalised/target_field.max()).expand(1, -1), torch.flatten(target_field/target_field.max()).expand(1, -1))
        loss.backward(retain_graph=True)

        def closure():
            return loss
        optimiser.step(closure)

    if save_progress:
        with open(r'Output_2D_iter\NMSE_History.txt', 'w') as nmse_file:
            for each_nmse in nmse_list:
                nmse_file.write(str(each_nmse) + '\n')

        # Save final hologram
        multi_phase_hologram = hologram.angle() % SLM_PHASE_RANGE / SLM_PHASE_RANGE
        save_image(r'.\Output_2D_iter\LBFGS_holo', multi_phase_hologram.detach().cpu(), 1.0)

        # Save reconstructions at different distances to check defocus
        for distance in [0.1, 0.5, 1, 5, 10, 100, 10**9]:
            reconstruction_abs = fresnel_propergation(hologram, distance, SLM_PITCH_SIZE, LASER_WAVELENGTH).abs()
            reconstruction_normalised = energy_conserve(reconstruction_abs, ENERGY_CONSERVATION_SCALING)
            save_image(r'.\Output_2D_iter\LBFGS_recon_defocused_at_{}'.format(distance), reconstruction_normalised.detach().cpu(), target_field.detach().cpu().max())

    torch.no_grad()
    hologram = amplitude * torch.exp(1j * phase)
    return hologram.detach(), nmse_list


def main():
    """
    Main function of lbfgs_cgh_2d
    """
    target_field = torchvision.io.read_image(r".\Target_images\mandrill.png", torchvision.io.ImageReadMode.GRAY).to(torch.float64)
    target_field_normalised = energy_conserve(target_field, ENERGY_CONSERVATION_SCALING)

    # Check if output folder exists, then save a copy of target_field
    if not os.path.isdir('Output_2D_iter'):
        os.makedirs('Output_2D_iter')
    save_image(r'.\Output_2D_iter\Target_field_normalised', target_field_normalised)



    hologram, nmse_list_SGD_MSE = optim_cgh_2d(
        target_field_normalised,
        wavelength=LASER_WAVELENGTH,
        pitch_size=SLM_PITCH_SIZE,
        save_progress=False,
        iteration_number=NUM_ITERATIONS,
        cuda=True,
        learning_rate=LEARNING_RATE,
        propagation_function=fraunhofer_propergation,  # Uncomment to choose Fraunhofer propagation
        # propagation_function=fresnel_propergation, # Uncomment to choose Fresnel Propagation
        optimise_algorithm="SGD",
        loss_function=torch.nn.MSELoss(reduction="sum")
    )

    # Carry out SGD optimisation with MSE as loss function
    time_start = time.time()
    hologram, nmse_list_SGD_MSE = optim_cgh_2d(
        target_field_normalised,
        wavelength=LASER_WAVELENGTH,
        pitch_size=SLM_PITCH_SIZE,
        save_progress=False,
        iteration_number=NUM_ITERATIONS,
        cuda=True,
        learning_rate=LEARNING_RATE,
        propagation_function=fraunhofer_propergation,  # Uncomment to choose Fraunhofer propagation
        # propagation_function=fresnel_propergation, # Uncomment to choose Fresnel Propagation
        optimise_algorithm="SGD",
        loss_function=torch.nn.MSELoss(reduction="sum")
    )
    time_elapsed = time.time() - time_start
    print("GD with MSE:\t time elapsed = {:.3f}s,\t final NMSE = {:.15e}".format(time_elapsed, nmse_list_SGD_MSE[-1]))

    # Carry out SGD optimisation with CE as loss function
    time_start = time.time()
    hologram, nmse_list_SGD_CE = optim_cgh_2d(
        target_field_normalised,
        wavelength=LASER_WAVELENGTH,
        pitch_size=SLM_PITCH_SIZE,
        save_progress=False,
        iteration_number=NUM_ITERATIONS,
        cuda=True,
        learning_rate=LEARNING_RATE,
        propagation_function=fraunhofer_propergation,  # Uncomment to choose Fraunhofer propagation
        # propagation_function=fresnel_propergation, # Uncomment to choose Fresnel Propagation
        optimise_algorithm="SGD",
        loss_function=torch.nn.CrossEntropyLoss(label_smoothing=0.0, reduction="sum")
    )
    time_elapsed = time.time() - time_start
    print("GD with CE:\t time elapsed = {:.3f}s,\t final NMSE = {:.15e}".format(time_elapsed, nmse_list_SGD_CE[-1]))

    # Carry out SGD optimisation with RE as loss function
    time_start = time.time()
    hologram, nmse_list_SGD_RE = optim_cgh_2d(
        target_field_normalised,
        wavelength=LASER_WAVELENGTH,
        pitch_size=SLM_PITCH_SIZE,
        save_progress=False,
        iteration_number=NUM_ITERATIONS,
        cuda=True,
        learning_rate=LEARNING_RATE,
        propagation_function=fraunhofer_propergation,  # Uncomment to choose Fraunhofer propagation
        # propagation_function=fresnel_propergation, # Uncomment to choose Fresnel Propagation
        optimise_algorithm="SGD",
        loss_function=torch.nn.KLDivLoss(reduction="sum")
    )
    time_elapsed = time.time() - time_start
    print("GD with RE:\t time elapsed = {:.3f}s,\t final NMSE = {:.15e}".format(time_elapsed, nmse_list_SGD_RE[-1]))



    # Carry out Adam optimisation with MSE as loss function
    time_start = time.time()
    hologram, nmse_list_Adam_MSE = optim_cgh_2d(
        target_field_normalised,
        wavelength=LASER_WAVELENGTH,
        pitch_size=SLM_PITCH_SIZE,
        save_progress=False,
        iteration_number=NUM_ITERATIONS,
        cuda=True,
        learning_rate=LEARNING_RATE,
        propagation_function=fraunhofer_propergation,  # Uncomment to choose Fraunhofer propagation
        # propagation_function=fresnel_propergation, # Uncomment to choose Fresnel Propagation
        optimise_algorithm="Adam",
        loss_function=torch.nn.MSELoss(reduction="sum")
    )
    time_elapsed = time.time() - time_start
    print("Adam with MSE:\t time elapsed = {:.3f}s,\t final NMSE = {:.15e}".format(time_elapsed, nmse_list_Adam_MSE[-1]))

    # Carry out Adam optimisation with CE as loss function
    time_start = time.time()
    hologram, nmse_list_Adam_CE = optim_cgh_2d(
        target_field_normalised,
        wavelength=LASER_WAVELENGTH,
        pitch_size=SLM_PITCH_SIZE,
        save_progress=False,
        iteration_number=NUM_ITERATIONS,
        cuda=True,
        learning_rate=LEARNING_RATE,
        propagation_function=fraunhofer_propergation,  # Uncomment to choose Fraunhofer propagation
        # propagation_function=fresnel_propergation, # Uncomment to choose Fresnel Propagation
        optimise_algorithm="Adam",
        loss_function=torch.nn.CrossEntropyLoss(label_smoothing=0.0, reduction="sum")
    )
    time_elapsed = time.time() - time_start
    print("Adam with CE:\t time elapsed = {:.3f}s,\t final NMSE = {:.15e}".format(time_elapsed, nmse_list_Adam_CE[-1]))

    # Carry out Adam optimisation with RE as loss function
    time_start = time.time()
    hologram, nmse_list_Adam_RE = optim_cgh_2d(
        target_field_normalised,
        wavelength=LASER_WAVELENGTH,
        pitch_size=SLM_PITCH_SIZE,
        save_progress=False,
        iteration_number=NUM_ITERATIONS,
        cuda=True,
        learning_rate=LEARNING_RATE,
        propagation_function=fraunhofer_propergation,  # Uncomment to choose Fraunhofer propagation
        # propagation_function=fresnel_propergation, # Uncomment to choose Fresnel Propagation
        optimise_algorithm="Adam",
        loss_function=torch.nn.KLDivLoss(reduction="sum")
    )
    time_elapsed = time.time() - time_start
    print("Adam with RE:\t time elapsed = {:.3f}s,\t final NMSE = {:.15e}".format(time_elapsed, nmse_list_Adam_RE[-1]))



    # Carry out LBFGS optimisation with MSE as loss function
    time_start = time.time()
    hologram, nmse_list_LBFGS_MSE = optim_cgh_2d(
        target_field_normalised,
        wavelength=LASER_WAVELENGTH,
        pitch_size=SLM_PITCH_SIZE,
        save_progress=False,
        iteration_number=NUM_ITERATIONS,
        cuda=True,
        learning_rate=LEARNING_RATE,
        propagation_function=fraunhofer_propergation,  # Uncomment to choose Fraunhofer propagation
        # propagation_function=fresnel_propergation, # Uncomment to choose Fresnel Propagation
        optimise_algorithm="LBFGS",
        loss_function=torch.nn.MSELoss(reduction="sum")
    )
    time_elapsed = time.time() - time_start
    print("LBFGS with MSE:\t time elapsed = {:.3f}s,\t final NMSE = {:.15e}".format(time_elapsed, nmse_list_LBFGS_MSE[-1]))

    # Carry out LBFGS optimisation with CE as loss function
    time_start = time.time()
    hologram, nmse_list_LBFGS_CE = optim_cgh_2d(
        target_field_normalised,
        wavelength=LASER_WAVELENGTH,
        pitch_size=SLM_PITCH_SIZE,
        save_progress=False,
        iteration_number=NUM_ITERATIONS,
        cuda=True,
        learning_rate=LEARNING_RATE,
        propagation_function=fraunhofer_propergation,  # Uncomment to choose Fraunhofer propagation
        # propagation_function=fresnel_propergation, # Uncomment to choose Fresnel Propagation
        optimise_algorithm="LBFGS",
        loss_function=torch.nn.CrossEntropyLoss(label_smoothing=0.0, reduction="sum")
    )
    time_elapsed = time.time() - time_start
    print("LBFGS with CE:\t time elapsed = {:.3f}s,\t final NMSE = {:.15e}".format(time_elapsed, nmse_list_LBFGS_CE[-1]))

    # Carry out LBFGS optimisation with RE as loss function
    time_start = time.time()
    hologram, nmse_list_LBFGS_RE = optim_cgh_2d(
        target_field_normalised,
        wavelength=LASER_WAVELENGTH,
        pitch_size=SLM_PITCH_SIZE,
        save_progress=False,
        iteration_number=NUM_ITERATIONS,
        cuda=True,
        learning_rate=LEARNING_RATE,
        propagation_function=fraunhofer_propergation,  # Uncomment to choose Fraunhofer propagation
        # propagation_function=fresnel_propergation, # Uncomment to choose Fresnel Propagation
        optimise_algorithm="LBFGS",
        loss_function=torch.nn.KLDivLoss(reduction="sum")
    )
    time_elapsed = time.time() - time_start
    print("LBFGS with RE:\t time elapsed = {:.3f}s,\t final NMSE = {:.15e}".format(time_elapsed, nmse_list_LBFGS_RE[-1]))



    # Carry out Gerchberg Saxton for reference
    time_start = time.time()
    nmse_list_GS = gerchberg_saxton(target_field_normalised, iteration_number=NUM_ITERATIONS)
    time_elapsed = time.time() - time_start
    print("GS reference:\t time elapsed = {:.3f}s,\t final NMSE = {:.15e}".format(time_elapsed, nmse_list_GS[-1]))


    # Plot NMSE
    x_list = range(1, NUM_ITERATIONS + 1)
    plt.plot(x_list, nmse_list_SGD_MSE, 's--', label="GD optimiser with MSE loss")
    plt.plot(x_list, nmse_list_SGD_CE, 'x--', label="GD optimiser with CE loss")
    plt.plot(x_list, nmse_list_SGD_RE, '^--', label="GD optimiser with RE loss")
    plt.plot(x_list, nmse_list_Adam_MSE, 's-.', label="Adam optimiser with MSE loss")
    plt.plot(x_list, nmse_list_Adam_CE, 'x-.', label="Adam optimiser with CE loss")
    plt.plot(x_list, nmse_list_Adam_RE, '^-.', label="Adam optimiser with RE loss")
    plt.plot(x_list, nmse_list_LBFGS_MSE, 's:', label="L-BFGS optimiser with MSE loss")
    plt.plot(x_list, nmse_list_LBFGS_CE, 'x:', label="L-BFGS optimiser with CE loss")
    plt.plot(x_list, nmse_list_LBFGS_RE, '^:', label="L-BFGS optimiser with RE loss")
    plt.plot(x_list, nmse_list_GS, 'o--', label="Gerchberg Saxton Reference")

    plt.xticks(x_list)
    plt.title("Optimisation of CGH for Fraunhofer diffraction")
    plt.xlabel("iterarion(s)")
    plt.ylabel("NMSE")
    plt.legend()
    plt.show()


if __name__ == "__main__":
    main()
