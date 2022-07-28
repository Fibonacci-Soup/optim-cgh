#!/usr/bin/env python3
"""
Copyright(c) 2022 Jinze Sha (js2294@cam.ac.uk)
Centre for Molecular Materials, Photonics and Electronics, University of Cambridge
All Rights Reserved.

This is the toolbox for Computer-Generated Hologram (CGH) related functions.
"""

import os
import math
import torch
import torchvision


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


def fresnel_propergation(hologram, distance=2, pitch_size=0.0000136, wavelength=0.000000532):
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


def fresnel_backward_propergation(field, distance=2, pitch_size=0.0000136, wavelength=0.000000532):
    """
    Fresnel propergation

    Equivalent MatLab code written by Fan Yang (fy255@cam.ac.uk) in 2019:
        function [u2] = ForwardPropergation(u1,pitch,lambda,z)
            holo_width = size(u2,2);
            holo_height = size(u2,1);
            [xx, yy] = meshgrid(1:holo_width, 1:holo_height);
            xMeters = pitch*(xx-(holo_width+1)/2);
            yMeters = pitch*(yy-(holo_height+1)/2);
            zern3 = pi*(xMeters.^2 + yMeters.^2);
            h = (-1i)*exp(-1i/(lambda.*z).*zern3);
            U2=fftshift(ifft2(fftshift(u2)));
            u1 = U2./h;
        end

    :param torch.tensor hologram: input tensor representing the hologram
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


def save_image(filename, image_tensor, tensor_dynamic_range=None):
    if tensor_dynamic_range is None:
        tensor_dynamic_range = image_tensor.max()
        print("save_image_dynamic_range" + filename, tensor_dynamic_range.tolist())
    torchvision.io.write_png((image_tensor / tensor_dynamic_range * 255.0).to(torch.uint8), filename + ".png", compression_level=0)


def energy_conserve(field, scaling=1.0):
    return field * torch.sqrt((scaling * field.size(-1) * field.size(-2)) / (field**2).sum())


def gerchberg_saxton(target_field, iteration_number=50):
    # if not os.path.isdir('Output_GS'):
    #     os.makedirs('Output_GS')
    torch.manual_seed(0)
    A = torch.exp(1j * ((torch.rand(target_field.size()) * 2 - 1) * math.pi).to(torch.float64))
    GS_NMSE_list = []
    for i in range(iteration_number):
        E = torch.fft.fftshift(torch.fft.fft2(torch.fft.fftshift(A)))
        E_norm = energy_conserve(E.abs())
        GS_NMSE_list.append((torch.nn.MSELoss(reduction="mean")(E_norm, target_field)).item() / (target_field**2).sum())
        # save_image(r".\Output_GS\GS_recon_i_{}".format(i), E_norm)
        E = target_field * torch.exp(1j * E.angle())
        A = torch.fft.ifftshift(torch.fft.ifft2(torch.fft.ifftshift(E)))
        A = torch.exp(1j * A.angle())
    return GS_NMSE_list


def gerchberg_saxton_3d_sequential_slicing(target_fields, distances, iteration_number=50):
    if not os.path.isdir('Output_GS'):
        os.makedirs('Output_GS')
    # torch.manual_seed(0)
    A = torch.exp(1j * ((torch.rand(target_fields[0].size()) * 2 - 1) * math.pi).to(torch.float64))
    GS_NMSE_list = []
    for i in range(iteration_number):
        slice_number = 3 - i % len(target_fields)
        print(slice_number)
        E = fresnel_propergation(A, distances[slice_number])

        E_norm = energy_conserve(E.abs())
        GS_NMSE_list.append((torch.nn.MSELoss(reduction="mean")(E_norm, target_fields[slice_number])).item() / (target_fields[slice_number]**2).sum())
        for d_index, distance in enumerate(distances):
            reconstruction_abs = fresnel_propergation(A, distance).abs()
            reconstruction_normalised = energy_conserve(reconstruction_abs)
            save_image(r'.\Output_GS\GS_recon_i_{}_d_{}'.format(i, d_index), reconstruction_normalised, target_fields[slice_number].max())

        E = target_fields[slice_number] * torch.exp(1j * E.angle())
        A = fresnel_backward_propergation(E, distances[slice_number])
        A = torch.exp(1j * A.angle())
    return GS_NMSE_list
