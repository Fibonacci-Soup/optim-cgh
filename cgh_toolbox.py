#!/usr/bin/env python3
"""
Copyright(c) 2022 Jinze Sha (js2294@cam.ac.uk)
Centre for Molecular Materials, Photonics and Electronics, University of Cambridge
All Rights Reserved.

This is the toolbox for Computer-Generated Hologram (CGH) related functions.
"""

from PIL import Image
import torch
import numpy as np

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

def save_image(file_name, image_tensor, vmin=0., vmax=1.):
    """
    Save the given image to a file

    :param str file_name (str): File name to save
    :param torch.tensor image_tensor: input tensor representing the image to save
    :param float vmin: minimum value of the data range that the colormap covers
    :param float vmax: maximum value of the data range that the colormap covers
    """
    image_numpy = image_tensor.cpu().detach().numpy()
    image_numpy[image_numpy < vmin] = vmin
    image_numpy[image_numpy > vmax] = vmax
    image_numpy /= (vmax - vmin)
    image_numpy *= 255
    result_img = Image.fromarray(image_numpy).convert("L")
    result_img.save(file_name)

def fraunhofer_propergation(hologram, *_):
    """
    Fraunhofer propergation

    :param torch.tensor hologram: input tensor representing the hologram
    :param *_: stash away unused variables (e.g. distance, pitch_size, wavelength)
    :returns: the reconstruction of the hologram at far field
    """
    return torch.fft.fftshift(torch.fft.fft2(hologram))

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
    holo_height = hologram.shape[0]
    holo_width = hologram.shape[1]

    x_lin = torch.linspace(-pitch_size * (holo_width - 1) / 2,
                           pitch_size * (holo_width - 1) / 2,
                           holo_width,
                           dtype=torch.float32).to(hologram.device)
    y_lin = torch.linspace(-pitch_size * (holo_height - 1) / 2,
                           pitch_size * (holo_height - 1) / 2,
                           holo_height,
                           dtype=torch.float32).to(hologram.device)
    y_meters, x_meters = torch.meshgrid(y_lin, x_lin, indexing='ij')

    h = (-1j)*torch.exp(-1j/(wavelength*distance) * np.pi * (x_meters**2 + y_meters**2))
    U2 = h * hologram
    u2 = torch.fft.fftshift(torch.fft.fft2(U2))
    return u2


def normalise_reconstruction(reconstruction):
    """
    Normalise the reconstruction resulting from Fraunhofer or Fresnel diffraction into the range of [0, 1]

    :param torch.tensor reconstruction: input tensor representing the reconstruction
    :returns: the normalised reconstruction
    """
    reconstruction_abs = reconstruction.abs()
    recon_ceil = reconstruction_abs.size(0) + reconstruction_abs.size(1)
    reconstruction_abs[reconstruction_abs > recon_ceil] = recon_ceil
    reconstruction_abs = reconstruction_abs / recon_ceil
    return reconstruction_abs
