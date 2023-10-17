#!/usr/bin/env python3
"""
Copyright(c) 2022 Jinze Sha (js2294@cam.ac.uk)
Centre for Molecular Materials, Photonics and Electronics, University of Cambridge
All Rights Reserved.

This is the toolbox for Computer-Generated Hologram (CGH) related functions.
"""

import os
import time
import math
import torch
import torchvision
import scipy.interpolate
import numpy as np

DEFAULT_PITCH_SIZE = 0.00000425
DEFAULT_WAVELENGTH = 0.0000006607


def generate_checkerboard_image(vertical_size=512, horizontal_size=512, size=2):
    checkerboard_array = np.zeros((vertical_size, horizontal_size))
    for v_i in range(0, vertical_size):
        for h_i in range(0, horizontal_size):
            if v_i % size == 0:
                if h_i % size == 0:
                    checkerboard_array[v_i][h_i] = 255
                else:
                    checkerboard_array[v_i][h_i] = 0
            else:
                if h_i % size == 0:
                    checkerboard_array[v_i][h_i] = 0
                else:
                    checkerboard_array[v_i][h_i] = 255
    return np.array([checkerboard_array])


def generate_grid_image(vertical_size=512, horizontal_size=512, vertical_spacing=2, horizontal_spacing=2):
    grid_array = np.zeros((vertical_size, horizontal_size))
    for v_i in range(0, vertical_size):
        for h_i in range(0, horizontal_size):
            if (v_i % vertical_spacing == 0) or (h_i % horizontal_spacing == 0) or \
                    (v_i % vertical_spacing == vertical_spacing-1) or (h_i % horizontal_spacing == horizontal_spacing-1):
                grid_array[v_i][h_i] = 1
    return np.array([grid_array])


def zero_pad_to_size(input_tensor, target_height=5120, target_width=5120):
    zero_pad_left = int((target_width - input_tensor.shape[-1]) / 2)
    zero_pad_right = target_width - input_tensor.shape[-1] - zero_pad_left
    zero_pad_top = int((target_height - input_tensor.shape[-2]) / 2)
    zero_pad_bottom = target_height - input_tensor.shape[-2] - zero_pad_top
    padded_image = torch.nn.ZeroPad2d((zero_pad_left, zero_pad_right, zero_pad_top, zero_pad_bottom))(input_tensor)
    return padded_image


def save_image(filename, image_tensor, tensor_dynamic_range=None):
    if tensor_dynamic_range is None:
        tensor_dynamic_range = image_tensor.max()
        print("save_image_dynamic_range: " + filename, tensor_dynamic_range.tolist())
    torchvision.io.write_png((image_tensor / tensor_dynamic_range * 255.0).to(torch.uint8), filename + ".png", compression_level=0)


def hologram_encoding_gamma_correct_linear(gamma_correct_me, pre_gamma_grey_values=None, post_gamma_grey_values=None):
    if pre_gamma_grey_values is None:
        pre_gamma_grey_values = [0, 15, 31, 47, 63, 79, 95, 111, 127, 143, 159, 175, 191, 207, 223, 239, 255]
    if post_gamma_grey_values is None:
        post_gamma_grey_values = [0, 10, 15, 19, 24, 29, 39, 54,  72,  97,  124, 150, 172, 193, 213, 232, 241]

    # Numpy Option - limited to linear
    # Lookup gamma-corrected value using numpy's linear interpolation function
    # interpolated_values = np.interp(gamma_correct_me, pre_gamma_grey_values, post_gamma_grey_values)

    # Scipy Option - can do cubic
    inter_func = scipy.interpolate.interp1d(pre_gamma_grey_values, post_gamma_grey_values, kind='quadratic')
    interpolated_values = inter_func(gamma_correct_me)

    # round-off to nearest int - need to do a type conversion as per https://stackoverflow.com/questions/55146871/can-numpy-rint-to-return-an-int32
    # gamma_corrected_value = np.rint(interpolated_values).astype(np.uint8)
    return torch.tensor(interpolated_values)


def save_hologram_and_its_recons(hologram, distances, alg_name, pitch_size=DEFAULT_PITCH_SIZE, wavelength=DEFAULT_WAVELENGTH, filename_note=''):
    print("phase mean: ", hologram.angle().mean().item(), "max: ", hologram.angle().max().item(), "min: ", hologram.angle().min().item())
    phase_hologram = hologram.detach().cpu().angle() % (2*math.pi) / (2*math.pi) * 255.0
    # gamma_corrected_phase_hologram = hologram_encoding_gamma_correct_linear(phase_hologram)

    # print("encoded holo mean: ", phase_hologram.mean().item(), "max: ", phase_hologram.max().item(), "min: ", phase_hologram.min().item())
    if not os.path.isdir(os.path.join('Output', alg_name)):
        os.makedirs(os.path.join('Output', alg_name))
    save_image(os.path.join('Output', alg_name, '{}_holo_{:.2f}m{}'.format(alg_name, distances[0], filename_note)), phase_hologram, 255.0)

    for distance in distances:
        reconstruction_abs = fresnel_propergation(hologram, distance, pitch_size=pitch_size, wavelength=wavelength).abs()
        reconstruction_normalised = energy_conserve(reconstruction_abs)
        save_image(os.path.join('Output', alg_name, '{}_recon_{:.2f}m{}'.format(alg_name, distance, filename_note)), reconstruction_normalised.detach().cpu())


def add_zeros_below(target_field):
    """
    Add zeros below target image (useful for binary SLM).

    :param torch.tensor target_field: input tensor representing an image
    :returns: the image being added zeros of same dimensions below
    """
    return torch.cat((target_field, torch.zeros([target_field.size(0), target_field.size(1)])), 0)


def add_up_side_down_replica_below(target_field):
    """
    Add an up side down replica below target image (useful for binary SLM)

    :param torch.tensor target_field: input tensor representing an image
    :returns: the image being added an up side down replica below
    """
    target_up_side_down = torch.rot90(target_field, 2, [0, 1])
    return torch.cat((target_field, target_up_side_down), 0)


def flip_left_right(target_field):
    """
    Flip the image left and right

    :param torch.tensor target_field: input tensor representing an image
    :returns: the image flipped left and right
    """
    return torch.fliplr(target_field)


def fraunhofer_propergation(hologram, *_):
    """
    Fraunhofer propergation

    :param torch.tensor hologram: input tensor representing the hologram
    :param *_: stash away unused variables (e.g. distance, pitch_size, wavelength)
    :returns: the reconstruction of the hologram at far field
    """
    return torch.fft.fftshift(torch.fft.fft2(torch.fft.fftshift(hologram)))


def fresnel_propergation(hologram, distance=2, pitch_size=DEFAULT_PITCH_SIZE, wavelength=DEFAULT_WAVELENGTH):
    """
    Fresnel propergation

    Equivalent MatLab code written by Fan Yang (fy255@cam.ac.uk) in 2019:
        function [u2] = ForwardPropergation(u1,pitch,lambda,z)
            holo_width = size(u1,2);
            holo_height = size(u1,1);

            [xx, yy] = meshgrid(1:holo_width, 1:holo_height);
            xMeters = pitch*(xx-(holo_width+1)/2);
            yMeters = pitch*(yy-(holo_height+1)/2);

            zern3 = pi*(xMeters.^2 + yMeters.^2);
            h = (-1i)*exp(-1i/(lambda.*z).*zern3);
            U2=h.*u1;
            u2=fftshift(fft2(U2));
        end

    :param torch.tensor hologram: input tensor representing the hologram
    :param distance: distance of Fresnel propergation
    :param pitch_size: pitch size of the spatial light modulator (SLM) which displays the hologram
    :param wavelength: wavelength of the light source
    :returns: the reconstruction of the hologram at the requested distance
    """
    distance = torch.tensor([distance]).to(hologram.device)
    holo_height = hologram.shape[-2]
    holo_width = hologram.shape[-1]

    x_lin = torch.linspace(-pitch_size * (holo_width - 1) / 2,
                           pitch_size * (holo_width - 1) / 2,
                           holo_width,
                           dtype=torch.float32).to(hologram.device)
    y_lin = torch.linspace(-pitch_size * (holo_height - 1) / 2,
                           pitch_size * (holo_height - 1) / 2,
                           holo_height,
                           dtype=torch.float32).to(hologram.device)
    y_meters, x_meters = torch.meshgrid(y_lin, x_lin, indexing='ij')

    h = (-1j)*torch.exp(-1j/(wavelength*distance) * math.pi * (x_meters**2 + y_meters**2))
    U2 = h * hologram
    u2 = torch.fft.fftshift(torch.fft.fft2(torch.fft.fftshift(U2)))
    return u2


def fresnel_backward_propergation(field, distance=2, pitch_size=DEFAULT_PITCH_SIZE, wavelength=DEFAULT_WAVELENGTH):
    """
    Fresnel backward propergation

    :param torch.tensor hologram: input tensor representing the replay field
    :param distance: distance of Fresnel propergation
    :param pitch_size: pitch size of the spatial light modulator (SLM) which displays the hologram
    :param wavelength: wavelength of the light source
    :returns: the reconstruction of the hologram at the requested distance
    """
    distance = torch.tensor([distance]).to(field.device)
    holo_height = field.shape[-2]
    holo_width = field.shape[-1]

    x_lin = torch.linspace(-pitch_size * (holo_width - 1) / 2,
                           pitch_size * (holo_width - 1) / 2,
                           holo_width,
                           dtype=torch.float32).to(field.device)
    y_lin = torch.linspace(-pitch_size * (holo_height - 1) / 2,
                           pitch_size * (holo_height - 1) / 2,
                           holo_height,
                           dtype=torch.float32).to(field.device)
    y_meters, x_meters = torch.meshgrid(y_lin, x_lin, indexing='ij')

    h = (-1j)*torch.exp(-1j/(wavelength*distance) * math.pi * (x_meters**2 + y_meters**2))
    U2 = torch.fft.ifftshift(torch.fft.ifft2(torch.fft.ifftshift(field)))
    u1 = U2 / h
    return u1


def generate_quadradic_phase(size, scaling=0.00002):
    holo_height = size[0]
    holo_width = size[1]

    x_lin = torch.linspace(-(holo_width - 1) / 2, (holo_width - 1) / 2, holo_width)
    y_lin = torch.linspace(-(holo_height - 1) / 2, (holo_height - 1) / 2, holo_height)
    y_meters, x_meters = torch.meshgrid(y_lin, x_lin, indexing='ij')

    h = (x_meters**2 + y_meters**2) * scaling
    return h


def generate_linear_phase(size, scaling=0.1):
    holo_height = size[0]
    holo_width = size[1]

    x_lin = torch.linspace(-(holo_width - 1) / 2, (holo_width - 1) / 2, holo_width)
    y_lin = torch.linspace(-(holo_height - 1) / 2, (holo_height - 1) / 2, holo_height)
    y_meters, x_meters = torch.meshgrid(y_lin, x_lin, indexing='ij')

    h = torch.sqrt(x_meters**2 + y_meters**2) * scaling
    return h


def low_pass_filter_2d(img, filter_rate=1):
    fft_img = torch.fft.fft2(img)
    h = img.shape[-2]
    w = img.shape[-1]
    cy, cx = int(h/2), int(w/2)  # centerness
    rh, rw = int(filter_rate * cy), int(filter_rate * cx)  # filter_size
    fft_img[:, cy-rh:cy+rh, cx-rw:cx+rw] = 0
    # fft_img[:, 0:cy-rh, 0:cx-rw] = 0
    # fft_img[:, cy+rh:h, cx+rw:w] = 0
    return torch.fft.ifft2(fft_img).abs()


def energy_conserve(field, scaling=1.0):
    return field * torch.sqrt((scaling * field.size(-1) * field.size(-2)) / (field**2).sum())


def energy_match(field, target_field):
    return field * torch.sqrt((target_field**2).sum() / (field**2).sum())


def gerchberg_saxton_fraunhofer(target_field, iteration_number=50, manual_seed_value=0, hologram_quantization_bit_depth=None):
    torch.manual_seed(manual_seed_value)
    device = torch.device("cuda")
    target_field = target_field.to(device)

    A = torch.exp(1j * ((torch.rand(target_field.size()) * 2 - 1) * math.pi).to(torch.float64)).to(device)
    GS_NMSE_list = []
    for i in range(iteration_number):
        E = torch.fft.fftshift(torch.fft.fft2(torch.fft.fftshift(A)))
        E_norm = energy_match(E.abs(), target_field)
        # save_image(".\\Output\\recon", E_norm.detach().cpu())
        # GS_NMSE_list.append((torch.nn.MSELoss(reduction="mean")(E_norm, target_field)).item() / (target_field**2).sum().item())
        # GS_NMSE_list.append(torch.nn.KLDivLoss(reduction="mean")(torch.flatten(E_norm / target_field.max()).expand(1, -1), torch.flatten(target_field / target_field.max()).expand(1, -1)).item())
        nmse_i = (((E_norm - target_field)**2).mean() / (target_field**2).sum()).item()
        # nmse_i = pytorch_msssim.ssim(E_norm.expand(1, -1, -1, -1), target_field.expand(1, -1, -1, -1)).item()

        GS_NMSE_list.append(nmse_i)
        E = target_field * torch.exp(1j * E.angle())
        A = torch.fft.ifftshift(torch.fft.ifft2(torch.fft.ifftshift(E)))
        phase_hologram = A.angle()
        if hologram_quantization_bit_depth:
            phase_hologram = torch.round(phase_hologram / math.pi * 2**(hologram_quantization_bit_depth-1)) / 2**(hologram_quantization_bit_depth-1) * math.pi
        A = torch.exp(1j * phase_hologram)
    # save_image('.\\Output\\recon_bit_depth_{}'.format(hologram_quantization_bit_depth), E_norm.detach().cpu(), target_field.max().cpu())
    # save_image('.\\Output\\hologram_bit_depth_{}'.format(hologram_quantization_bit_depth), phase_hologram.detach().cpu(), math.pi)
    return A, GS_NMSE_list


def gerchberg_saxton_fraunhofer_smooth(target_field, iteration_number=50):
    torch.manual_seed(0)
    A = torch.exp(1j * ((torch.rand(target_field.size()) * 2 - 1) * math.pi).to(torch.float64))
    GS_NMSE_list = []
    for i in range(iteration_number):
        E = torch.fft.fftshift(torch.fft.fft2(torch.fft.fftshift(A)))
        E_norm = energy_conserve(E.abs())
        GS_NMSE_list.append((torch.nn.MSELoss(reduction="mean")(E_norm, target_field)).item() / (target_field**2).sum())
        E = target_field * torch.exp(1j * E.angle())
        A = torch.fft.ifftshift(torch.fft.ifft2(torch.fft.ifftshift(E)))
        # Smooth the phase hologram
        A_blurred = torchvision.transforms.functional.gaussian_blur(A.angle(), kernel_size=3)
        A = torch.exp(1j * A_blurred)
    return A, GS_NMSE_list


def gerchberg_saxton_3d_sequential_slicing(target_fields, distances, iteration_number=50, weighting=0.001, time_limit=None, pitch_size=DEFAULT_PITCH_SIZE, wavelength=DEFAULT_WAVELENGTH):
    time_start = time.time()

    torch.manual_seed(0)
    device = torch.device("cuda")
    target_fields = target_fields.to(device)

    amplitude = torch.ones(target_fields[0].size(), requires_grad=False).to(torch.float64).to(device)
    phase = ((torch.rand(target_fields[0].size()) * 2 - 1) * math.pi).to(torch.float64).to(device)
    # phase = (torch.ones(target_fields[0].size()) * generate_quadradic_phase([target_fields[0].shape[-2], target_fields[0].shape[-1]], 0.00001)).to(torch.float64).detach().to(device)
    A = amplitude * torch.exp(1j * phase)

    nmse_lists = []
    for distance in distances:
        nmse_lists.append([])
    time_list = []

    for i in range(iteration_number):
        for d_index, distance in enumerate(distances):
            reconstruction_abs = fresnel_propergation(A, distance, pitch_size=pitch_size, wavelength=wavelength).abs()
            reconstruction_normalised = energy_conserve(reconstruction_abs)
            nmse_lists[d_index].append(((torch.nn.MSELoss(reduction="mean")(reconstruction_normalised, target_fields[d_index])).item() / (target_fields[d_index]**2).sum()).data.tolist())
            if weighting != 0:
                if (len(nmse_lists[d_index]) > 1) and (nmse_lists[d_index][-1] > nmse_lists[d_index][-2]):
                    weighting = weighting * 1.01
                else:
                    weighting = weighting * 0.99

        time_list.append(time.time() - time_start)
        if time_limit:
            if time_list[-1] >= time_limit:
                break

        slice_number = i % len(target_fields)
        E = fresnel_propergation(A, distances[slice_number], pitch_size=pitch_size, wavelength=wavelength)
        if weighting == 0:
            E = target_fields[slice_number] * torch.exp(1j * E.angle())
        else:
            E = (E.abs() * weighting + target_fields[slice_number] * (1-weighting)) * torch.exp(1j * E.angle())
        A = fresnel_backward_propergation(E, distances[slice_number], pitch_size=pitch_size, wavelength=wavelength)
        A = torch.exp(1j * A.angle())

    return A.detach().cpu(), nmse_lists, time_list


def optim_cgh_3d(target_fields, distances, sequential_slicing=False, wavelength=DEFAULT_WAVELENGTH, pitch_size=DEFAULT_PITCH_SIZE,
                 save_progress=False, iteration_number=20, cuda=False, learning_rate=0.1, record_all_nmse=True, optimise_algorithm="LBFGS",
                 grad_history_size=10, loss_function=torch.nn.MSELoss(reduction="sum"), energy_conserv_scaling=1.0, time_limit=None,
                 initial_phase='random', smooth_holo_kernel_size=None):
    """
    Carry out L-BFGS optimisation of CGH for a 3D target consisted of multiple slices of 2D images at different distances.
    If sequential_slicing is True, Loss is calculated for reconstructions in all distances.
    If sequential_slicing is False, Loss is calculated for reconstruction in each distance in turn for each iteration.

    :param target_fields: tensor for target images
    :param distances: image distances
    :param sequential_slicing: decide whether to calculate loss function for each slice in turn instead of all slices
    :param wavelength: wavelength of the light source
    :param pitch_size: pitch size of the spatial light modulator (SLM)
    :param save_progress: decide whether to save progress of every iteration to files
    :param iteration_number: number of iterations
    :param cuda: decide whether to use CUDA, use CPU otherwise
    :param learning_rate: set the parameter 'lr' of torch.optim
    :param loss_function: the objective function to minimise
    :returns: resultant hologram
    """
    time_start = time.time()
    torch.cuda.empty_cache()
    torch.manual_seed(0)
    device = torch.device("cuda" if cuda else "cpu")
    target_fields = target_fields.to(device)

    # Fixed unit amplitude
    amplitude = torch.ones(target_fields[0].size(), requires_grad=False).to(torch.float64).to(device)

    # Variable phase
    if initial_phase.lower() in ['random', 'rand']:
        # Random initial phase within [-pi, pi]
        phase = ((torch.rand(target_fields[0].size()) * 2 - 1) * math.pi).to(torch.float64).detach().to(device).requires_grad_()
    elif initial_phase.lower() in ['linear', 'lin']:
        # linear initial phase
        phase = (torch.ones(target_fields[0].size()) * generate_linear_phase([target_fields[0].shape[-2], target_fields[0].shape[-1]], 0.5)).to(torch.float64).detach().to(device).requires_grad_()
    elif initial_phase.lower() in ['quadratic', 'quad']:
        # linear initial phase
        phase = (torch.ones(target_fields[0].size()) * generate_quadradic_phase([target_fields[0].shape[-2],
                 target_fields[0].shape[-1]], 0.00002)).to(torch.float64).detach().to(device).requires_grad_()
    else:
        raise Exception("The required initial phase is not recognised!")

    # Decide optimisation algorithm
    if optimise_algorithm.lower() in ["lbfgs", "l-bfgs"]:
        optimiser = torch.optim.LBFGS([phase], lr=learning_rate, history_size=grad_history_size)
    elif optimise_algorithm.lower() in ["sgd", "gd"]:
        optimiser = torch.optim.SGD([phase], lr=learning_rate)
    elif optimise_algorithm.lower() == "adam":
        optimiser = torch.optim.Adam([phase], lr=learning_rate)
    else:
        raise Exception("Optimiser is not recognised!")

    time_list = []
    nmse_lists = []
    for distance in distances:
        nmse_lists.append([])

    for i in range(iteration_number):
        optimiser.zero_grad()
        if smooth_holo_kernel_size is not None:
            # Smooth the phase hologram
            blurrerd_phase = torchvision.transforms.functional.gaussian_blur(phase, kernel_size=smooth_holo_kernel_size)
            hologram = amplitude * torch.exp(1j * blurrerd_phase)
        else:
            hologram = amplitude * torch.exp(1j * phase)

        if sequential_slicing:
            # Propagate hologram for one distance only
            slice_number = i % len(target_fields)
            reconstruction_abs = fresnel_propergation(hologram, distances[slice_number], pitch_size, wavelength).abs()
            reconstruction_normalised = energy_conserve(reconstruction_abs, energy_conserv_scaling)

            # Calculate loss for the single slice
            loss = loss_function(torch.flatten(reconstruction_normalised / target_fields[slice_number].max()).expand(1, -1),
                                 torch.flatten(target_fields[slice_number] / target_fields[slice_number].max()).expand(1, -1))

        else:
            # Propagate hologram for all distances
            reconstructions_list = []
            for index, distance in enumerate(distances):
                reconstruction_abs = fresnel_propergation(hologram, distance=distance, pitch_size=pitch_size, wavelength=wavelength).abs()
                reconstruction_normalised = energy_conserve(reconstruction_abs, energy_conserv_scaling)
                reconstructions_list.append(reconstruction_normalised)
            reconstructions = torch.stack(reconstructions_list)

            # Calculate loss for all slices (stacked in reconstructions)
            loss = loss_function(torch.flatten(reconstructions / target_fields.max()).expand(1, -1),
                                 torch.flatten(target_fields / target_fields.max()).expand(1, -1))

        loss.backward(retain_graph=True)
        # Record NMSE
        if record_all_nmse:
            for index, distance in enumerate(distances):
                reconstruction_abs = fresnel_propergation(hologram, distance, pitch_size, wavelength).abs()
                reconstruction_normalised = energy_conserve(reconstruction_abs, energy_conserv_scaling)
                # reconstruction_normalised = energy_match(reconstruction_abs, target_fields[slice_number])
                nmse_value = (torch.nn.MSELoss(reduction="mean")(reconstruction_normalised, target_fields[index])).item() / (target_fields[index]**2).sum()
                nmse_lists[index].append(nmse_value.data.tolist())
                if save_progress:
                    save_hologram_and_its_recons(hologram, distances, optimise_algorithm)
        time_list.append(time.time() - time_start)
        if time_limit:
            if time_list[-1] >= time_limit:
                break

        def closure():
            return loss
        optimiser.step(closure)

    torch.no_grad()
    hologram = amplitude * torch.exp(1j * phase)
    return hologram.detach().cpu(), nmse_lists, time_list


def freeman_projector_encoding(holograms, alg_name='MultiFrame', filename_note=''):
    binary_hologram = torch.round(holograms.detach().cpu().angle() / math.pi)
    freeman_hologram = torch.zeros(3, holograms.size(-2), holograms.size(-1))
    for channel_i in range(3):
        channel_temp_sum = torch.zeros(1, holograms.size(-2), holograms.size(-1))
        for subframe_i in range(8):
            current_frame_i = (channel_i * 8 + subframe_i) % holograms.size(0)
            channel_temp_sum += binary_hologram[current_frame_i] * 2**subframe_i
        freeman_hologram[channel_i] = channel_temp_sum.detach()

    save_image(os.path.join('Output', alg_name, '{}_freeman_holo{}'.format(alg_name, filename_note)), freeman_hologram)
    return


def save_multi_frame_holograms_and_their_recons(holograms, reconstructions=None, recon_dynamic_range=None, alg_name='MultiFrame', filename_note=''):
    if not os.path.isdir(os.path.join('Output', '{}'.format(alg_name))):
        os.makedirs(os.path.join('Output', '{}'.format(alg_name)))
    freeman_projector_encoding(holograms)

    if reconstructions is None:
        print("Reconstructions not given, re-doing computations from holograms")
        for hologram_i, hologram in enumerate(holograms):
            # print("phase mean: ", hologram.angle().mean().item(), "max: ", hologram.angle().max().item(), "min: ", hologram.angle().min().item())
            phase_hologram = hologram.detach().cpu().angle() % (2*math.pi) / (2*math.pi) * 255.0
            # print("encoded holo mean: ", phase_hologram.mean().item(), "max: ", phase_hologram.max().item(), "min: ", phase_hologram.min().item())
            save_image(os.path.join('Output', alg_name, '{}_holo{}{}'.format(alg_name, hologram_i, filename_note)), phase_hologram, 255.0)

            # gamma_corrected_phase_hologram = hologram_encoding_gamma_correct_linear(phase_hologram)
            # print("Sony holo mean: ", gamma_corrected_phase_hologram.mean().item(), "max: ", gamma_corrected_phase_hologram.max().item(), "min: ", gamma_corrected_phase_hologram.min().item())
            # save_image('.\Output\{0}\{0}_sony_holo{1}'.format(alg_name, filename_note), gamma_corrected_phase_hologram, 255.0)
            reconstruction_abs = fraunhofer_propergation(hologram).abs()
            reconstruction_normalised = energy_conserve(reconstruction_abs)
            save_image(os.path.join('Output', alg_name, '{}_recon{}{}'.format(alg_name, hologram_i, filename_note)), reconstruction_normalised.detach().cpu(), recon_dynamic_range)
    else:
        for hologram_i, hologram in enumerate(holograms):
            phase_hologram = hologram.detach().cpu().angle() % (math.pi) / (math.pi) * 255.0
            save_image(os.path.join('Output', alg_name, '{}_holo{}{}'.format(alg_name, hologram_i, filename_note)), phase_hologram, 255.0)

        for reconstruction_i, reconstruction in enumerate(reconstructions):
            reconstruction_normalised = energy_conserve(reconstruction)
            save_image(os.path.join('Output', alg_name, '{}_recon{}{}'.format(alg_name, reconstruction_i, filename_note)), reconstruction_normalised.detach().cpu(), recon_dynamic_range)


def multi_frame_cgh(target_fields, distances, wavelength=DEFAULT_WAVELENGTH, pitch_size=DEFAULT_PITCH_SIZE,
                    iteration_number=20, cuda=False, learning_rate=0.1, save_progress=True, optimise_algorithm="LBFGS",
                    grad_history_size=10, loss_function=torch.nn.MSELoss(reduction="sum"), energy_conserv_scaling=1.0, time_limit=None,
                    num_frames=8):

    time_start = time.time()
    torch.cuda.empty_cache()
    torch.manual_seed(0)
    device = torch.device("cuda" if cuda else "cpu")
    target_fields = target_fields.to(device)

    # Fixed unit amplitude
    amplitude = torch.ones(target_fields[0].size(), requires_grad=False).to(torch.float32).to(device)

    # Multi-frame phases
    phases = ((torch.rand([num_frames] + list(target_fields[0].size())) * 2 - 1) * math.pi).to(torch.float32).detach().to(device).requires_grad_()

    # Decide optimisation algorithm
    if optimise_algorithm.lower() in ["lbfgs", "l-bfgs"]:
        optimiser = torch.optim.LBFGS([phases], lr=learning_rate, history_size=grad_history_size)
    elif optimise_algorithm.lower() in ["sgd", "gd"]:
        optimiser = torch.optim.SGD([phases], lr=learning_rate)
    elif optimise_algorithm.lower() == "adam":
        optimiser = torch.optim.Adam([phases], lr=learning_rate)
    else:
        raise Exception("Optimiser is not recognised!")

    time_list = []
    nmse_lists = []
    nmse_lists.append([])

    for i in range(iteration_number):
        print(i)
        optimiser.zero_grad()
        if i > iteration_number / 4:
            holograms = amplitude * torch.exp(1j * torch.nn.Sigmoid()(phases / math.pi) * math.pi)
        else:
            holograms = amplitude * torch.exp(1j * phases)

        # Propagate hologram for all distances

        reconstructions = fraunhofer_propergation(holograms).abs()
        if save_progress or (i == iteration_number - 1):
            save_multi_frame_holograms_and_their_recons(holograms, reconstructions, recon_dynamic_range=target_fields.detach().cpu().max(), alg_name='MultiFrame')
        reconstructions = reconstructions.mean(dim=0)
        reconstructions = energy_conserve(reconstructions, energy_conserv_scaling)
        if save_progress or (i == iteration_number - 1):
            save_image(os.path.join('Output', 'MultiFrame', 'MultiFrame_mean'), reconstructions.detach().cpu(), target_fields.detach().cpu().max())

        # Calculate loss for all slices (stacked in reconstructions)
        loss = loss_function(torch.flatten(reconstructions / target_fields.max()).expand(1, -1),
                             torch.flatten(target_fields / target_fields.max()).expand(1, -1))

        loss.backward(retain_graph=True)
        # Record NMSE
        if save_progress or (i == iteration_number - 1):
            reconstructions = fraunhofer_propergation(holograms).abs()
            reconstructions = reconstructions.mean(dim=0)
            reconstructions = energy_conserve(reconstructions, energy_conserv_scaling)
            nmse_value = (torch.nn.MSELoss(reduction="mean")(reconstructions, target_fields[0])).item() / (target_fields[0]**2).sum()
            nmse_lists[0].append(nmse_value.data.tolist())

        time_list.append(time.time() - time_start)
        if time_limit:
            if time_list[-1] >= time_limit:
                break

        def closure():
            return loss
        optimiser.step(closure)

    torch.no_grad()
    holograms = amplitude * torch.exp(1j * phases)
    return holograms.detach().cpu(), nmse_lists, time_list
