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
from cgh_toolbox import save_image, fraunhofer_propergation, fresnel_propergation, energy_conserve, gerchberg_saxton, generate_quadradic_phase, generate_linear_phase

# Experimental setup - device properties
SLM_PHASE_RANGE = 2 * math.pi
SLM_PITCH_SIZE = 0.0000136
LASER_WAVELENGTH = 0.000000532

ENERGY_CONSERVATION_SCALING = 1.0
NUM_ITERATIONS = 100
LEARNING_RATE = 0.1

SMOOTH_HOLOGRAM = True


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

    if SMOOTH_HOLOGRAM:
        phase = (torch.ones(target_field.size()) * generate_linear_phase([target_field.shape[-2], target_field.shape[-1]], 0.5)).to(torch.float64).detach().to(device).requires_grad_()
        save_image(r'.\Output_2D_iter\initial_phase', phase.detach().cpu() % SLM_PHASE_RANGE)
    else:
        # Random initial phase within [-pi, pi]
        phase = ((torch.rand(target_field.size()) * 2 - 1) * math.pi).to(torch.float64).detach().to(device).requires_grad_()

    if optimise_algorithm.lower() == "lbfgs":
        optimiser = torch.optim.LBFGS([phase], lr=learning_rate)
    elif optimise_algorithm.lower() == "sgd":
        optimiser = torch.optim.SGD([phase], lr=learning_rate)
    elif optimise_algorithm.lower() == "adam":
        optimiser = torch.optim.Adam([phase], lr=learning_rate)
    else:
        raise Exception("Optimiser is not recognised!")

    nmse_list = []

    for i in range(iteration_number):
        optimiser.zero_grad()

        ## Smooth the phase hologram
        if SMOOTH_HOLOGRAM:
            blurrerd_phase = torchvision.transforms.functional.gaussian_blur(phase, kernel_size=23)
            # save_image(r'.\Output_2D_iter\blurred_phase_i_{}'.format(i), blurrerd_phase.detach().cpu())
            hologram = amplitude * torch.exp(1j * blurrerd_phase)
        else:
            hologram = amplitude * torch.exp(1j * phase)

        # Propagate hologram to reconstruction plane
        reconstruction_abs = propagation_function(hologram, distance, pitch_size, wavelength).abs()
        reconstruction_normalised = energy_conserve(reconstruction_abs, ENERGY_CONSERVATION_SCALING)

        # Save hologram and reconstruction every iteration, if needed
        if save_progress:
            # binary_phase_hologram = torch.where(phase > 0, 1, 0)
            multi_phase_hologram = hologram.angle() % SLM_PHASE_RANGE / SLM_PHASE_RANGE
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

    if save_progress or True:
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


def summation_3d_hologram():
    # images = [r".\Target_images\512_A.png", r".\Target_images\512_B.png", r".\Target_images\512_C.png", r".\Target_images\512_D.png"]
    images = [r".\Target_images\mandrill.png", r".\Target_images\512_B.png", r".\Target_images\512_szzx.png", r".\Target_images\512_D.png"]

    distances = [.01, .02, .03, .04]

    target_fields_list = []
    for image_name in images:
        target_field = torchvision.io.read_image(image_name, torchvision.io.ImageReadMode.GRAY).to(torch.float64)
        target_field_normalised = energy_conserve(target_field, ENERGY_CONSERVATION_SCALING)
        target_fields_list.append(target_field_normalised)
    target_fields = torch.stack(target_fields_list)

    total_hologram = torch.zeros(torchvision.io.read_image(images[0], torchvision.io.ImageReadMode.GRAY).size()).to(torch.float64)
    hologram, nmse_list = optim_cgh_2d(
        target_fields[0],
        distance=distances[0],
        wavelength=LASER_WAVELENGTH,
        pitch_size=SLM_PITCH_SIZE,
        save_progress=False,
            iteration_number=NUM_ITERATIONS,
        cuda=True,
        learning_rate=LEARNING_RATE,
        propagation_function=fresnel_propergation,  # Uncomment to choose Fresnel Propagation
            optimise_algorithm="LBFGS",
        # loss_function=torch.nn.MSELoss(reduction="sum")
        loss_function=torch.nn.KLDivLoss(reduction="sum")
    )
    total_time_start = time.time()
    for i, distance in enumerate(distances):
        time_start = time.time()
        hologram, nmse_list = optim_cgh_2d(
            target_fields[i],
            distance=distance,
            wavelength=LASER_WAVELENGTH,
            pitch_size=SLM_PITCH_SIZE,
            save_progress=False,
            iteration_number=NUM_ITERATIONS,
            cuda=True,
            learning_rate=LEARNING_RATE,
            propagation_function=fresnel_propergation,  # Uncomment to choose Fresnel Propagation
            optimise_algorithm="LBFGS",
            # loss_function=torch.nn.MSELoss(reduction="sum")
            loss_function=torch.nn.KLDivLoss(reduction="sum")
        )
        total_hologram = total_hologram + hologram.detach().cpu()
        time_elapsed = time.time() - time_start
        print("Optimise hologram for Slice {}:\t time elapsed = {:.3f}s,\t this slice's NMSE = {:.15e}".format(i+1, time_elapsed, nmse_list[-1]))

    print("Total time taken for a summed hologram = {:.3f}s".format(time.time() - total_time_start))
    # Save reconstructions at different distances to check defocus
    total_phase_hologram = total_hologram.angle()
    save_image(r'.\Output_3D_iter\LBFGS_holo', total_phase_hologram.detach().cpu())
    for i, distance in enumerate(distances):
        reconstruction_abs = fresnel_propergation(torch.exp(1j*total_phase_hologram), distance, SLM_PITCH_SIZE, LASER_WAVELENGTH).abs()
        reconstruction_normalised = energy_conserve(reconstruction_abs, ENERGY_CONSERVATION_SCALING)
        print("Propagate the summed hologram to slice {}: NMSE = {:.15e}".format(i+1, (torch.nn.MSELoss(reduction="mean")(reconstruction_normalised, target_fields[i])).item() / (target_fields[i]**2).sum()))
        save_image(r'.\Output_3D_iter\LBFGS_NB_SoH_recon_{}'.format(distance), reconstruction_normalised, target_fields.max())


def main():
    """
    Main function of lbfgs_cgh_2d
    """
    target_field = torchvision.io.read_image(r".\Target_images\LB.png", torchvision.io.ImageReadMode.GRAY).to(torch.float64)
    target_field_normalised = energy_conserve(target_field, ENERGY_CONSERVATION_SCALING)

    # Check if output folder exists, then save a copy of target_field
    if not os.path.isdir('Output_2D_iter'):
        os.makedirs('Output_2D_iter')
    save_image(r'.\Output_2D_iter\Target_field_normalised', target_field_normalised)

    if SMOOTH_HOLOGRAM:
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
            # loss_function=torch.nn.MSELoss(reduction="sum")
        )
        time_elapsed = time.time() - time_start
        print("LBFGS with RE:\t time elapsed = {:.3f}s,\t final NMSE = {:.15e}".format(time_elapsed, nmse_list_LBFGS_RE[-1]))
        return


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
    plt.plot(x_list, nmse_list_SGD_MSE, 'b-', label="GD optimiser with MSE loss")
    plt.plot(x_list, nmse_list_SGD_CE, 'bx--', label="GD optimiser with CE loss")
    plt.plot(x_list, nmse_list_SGD_RE, 'b^:', label="GD optimiser with RE loss")
    plt.plot(x_list, nmse_list_Adam_MSE, 'g-', label="Adam optimiser with MSE loss")
    plt.plot(x_list, nmse_list_Adam_CE, 'gx--', label="Adam optimiser with CE loss")
    plt.plot(x_list, nmse_list_Adam_RE, 'g^:', label="Adam optimiser with RE loss")
    plt.plot(x_list, nmse_list_LBFGS_MSE, 'r-', label="L-BFGS optimiser with MSE loss")
    plt.plot(x_list, nmse_list_LBFGS_CE, 'rx--', label="L-BFGS optimiser with CE loss")
    plt.plot(x_list, nmse_list_LBFGS_RE, 'r^:', label="L-BFGS optimiser with RE loss")
    plt.plot(x_list, nmse_list_GS, 'o--', label="Gerchberg Saxton Reference")

    plt.xticks(x_list)
    plt.title("Optimisation of CGH for Fraunhofer diffraction")
    plt.xlabel("iterarion(s)")
    plt.ylabel("NMSE")
    plt.legend()
    plt.show()


if __name__ == "__main__":
    main()
    # summation_3d_hologram()
