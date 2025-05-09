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
import numpy as np
import scipy

DEFAULT_PITCH_SIZE = 0.0000136  # Default pitch size of the SLM 13.62e-6 for Freeman projector
DEFAULT_WAVELENGTH = 0.000000532  # Default wavelength of the laser 532e-9 for Freeman projector
DEFAULT_ENERGY_CONSERVATION_SCALING = 1.0  # Default scaling factor when conserving the energy of images across different slices
DATATYPE = torch.float32 # Switch to torch.float64 for better accuracy, but will need much more memory and computation resource

def read_image_to_tensor(filename):
    """
    Read an image file (please use PNG format with no compression) to pytorch tensor converted to floating points.
    You are welcomed to try other modules for other formats, torchvision turns out the best option for me.
    (If you use PIL, be careful with their channel index and type conversions)

    :param filename: name of file to load from
    :returns pytorch.tensor: the loaded image in a pytorch tensor in grayscale
    """
    return torchvision.io.read_image(filename, torchvision.io.ImageReadMode.GRAY).to(DATATYPE)


def load_target_images(filenames=[os.path.join('Target_images', 'A.png')], energy_conserv_scaling=DEFAULT_ENERGY_CONSERVATION_SCALING):
    """
    Load multiple image files to a pytorch tensor (please use PNG format with no compression)
    for single files, simply put a list of one element to filenames

    :param filenames: list of filenames to load from (Default is an example file)
    :param energy_conserv_scaling: all target images and reconstructions are conserved to have the same total energy (Default: 1.0)
    :returns pytorch.tensor: a stacked tensor of all loaded images
    """
    target_fields_list = []
    for filename in filenames:
        target_field = read_image_to_tensor(filename)
        # The following commented-out code might be useful for resizing and zero padding the target images
        # target_field = torch.nn.functional.interpolate(target_field.expand(1, -1, -1, -1), (1024, 1280))[0]
        # target_field = zero_pad_to_size(target_field, target_height=1024, target_width=1024, shift_downwards=256)
        target_field_normalised = energy_conserve(target_field, energy_conserv_scaling)
        target_fields_list.append(target_field_normalised)
    target_fields = torch.stack(target_fields_list)
    return target_fields


def save_image(filename, image_tensor, tensor_dynamic_range=None):
    """
    Save a tensor respresenting an image to a file.

    :param filename: name of file to save to
    :param image_tensor: tensor representing the image
    :param tensor_dynamic_range: dynamic range when saving the image (default: None, which will be assigned the maximum value of the tensor)
    """
    if tensor_dynamic_range is None:
        tensor_dynamic_range = image_tensor.max()
        print("saving image with dynamic range: " + filename, tensor_dynamic_range.tolist())
    torchvision.io.write_png((image_tensor / tensor_dynamic_range * 255.0).to(torch.uint8), filename + ".png", compression_level=0)


def generate_checkerboard_pattern(vertical_size=512, horizontal_size=512, square_size=2):
    """
    Generate a checkerboard pattern of given dimensions.

    :param vertical_size: vertical size of the overall pattern
    :param horizontal_size: horizontal size of the overall pattern
    :param square_size: the size of the squares in the checkerboard pattern
    :returns: tensor having the checkerboard patthern
    """
    checkerboard_array = np.zeros((vertical_size, horizontal_size))
    for v_i in range(0, vertical_size):
        for h_i in range(0, horizontal_size):
            if v_i % square_size == 0:
                if h_i % square_size == 0:
                    checkerboard_array[v_i][h_i] = 255
                else:
                    checkerboard_array[v_i][h_i] = 0
            else:
                if h_i % square_size == 0:
                    checkerboard_array[v_i][h_i] = 0
                else:
                    checkerboard_array[v_i][h_i] = 255
    return torch.from_numpy(np.array([checkerboard_array]))


def generate_grid_pattern(vertical_size=512, horizontal_size=512, vertical_spacing=2, horizontal_spacing=2, line_thickness=1):
    """
    Generate a grid pattern of given dimensions.

    :param vertical_size: vertical size of the overall pattern (Default: 512)
    :param horizontal_size: horizontal size of the overall pattern (Default: 512)
    :param vertical_spacing: vertical spacing between lines (Default: 2)
    :param horizontal_spacing: horizontal spacing between lines (Default: 2)
    :param line_thickness: thickness of the lines (Default: 1)
    :returns: tensor having the grid pattern
    """
    grid_array = np.zeros((vertical_size, horizontal_size))
    for v_i in range(0, vertical_size):
        for h_i in range(0, horizontal_size):
            if (v_i % vertical_spacing < line_thickness) or (h_i % horizontal_spacing < line_thickness) or \
                    (v_i % vertical_spacing >= vertical_spacing-line_thickness) or (h_i % horizontal_spacing >= horizontal_spacing-line_thickness):
                grid_array[v_i][h_i] = 1
    return torch.from_numpy(np.array([grid_array]))

def generate_dotted_grid_pattern(vertical_size=512, horizontal_size=512, vertical_spacing=2, horizontal_spacing=2, line_thickness=1):
    """
    Generate a grid pattern of given dimensions.

    :param vertical_size: vertical size of the overall pattern (Default: 512)
    :param horizontal_size: horizontal size of the overall pattern (Default: 512)
    :param vertical_spacing: vertical spacing between lines (Default: 2)
    :param horizontal_spacing: horizontal spacing between lines (Default: 2)
    :param line_thickness: thickness of the lines (Default: 1)
    :returns: tensor having the grid pattern
    """
    grid_array = np.zeros((vertical_size, horizontal_size))
    for v_i in range(0, vertical_size):
        for h_i in range(0, horizontal_size):
            if (v_i % vertical_spacing < line_thickness) and (h_i % horizontal_spacing < line_thickness) or \
               (v_i % vertical_spacing < line_thickness) and (h_i % horizontal_spacing >= horizontal_spacing-line_thickness) or \
               (v_i % vertical_spacing >= vertical_spacing-line_thickness) and (h_i % horizontal_spacing < line_thickness) or \
               (v_i % vertical_spacing >= vertical_spacing-line_thickness) and (h_i % horizontal_spacing >= horizontal_spacing-line_thickness):
                grid_array[v_i][h_i] = 1
    return torch.from_numpy(np.array([grid_array]))

def generate_circle_pattern(radius=512):
    xx, yy = np.mgrid[:2*radius, :2*radius]
    circle = (xx - radius + 0.5) ** 2 + (yy - radius + 0.5) ** 2
    solid_circle = circle <= (radius)**2
    return torch.from_numpy(np.array([solid_circle]))

def generate_donut_pattern(radius=512, line_thickness=1):
    xx, yy = np.mgrid[:2*radius, :2*radius]
    circle = (xx - radius + 0.5) ** 2 + (yy - radius + 0.5) ** 2
    donut = (circle <= (radius)**2) & (circle > (radius - line_thickness)**2)
    return torch.from_numpy(np.array([donut]))


def zero_pad_to_size(input_tensor, target_height=5120, target_width=5120, shift_downwards=0):
    """
    Zero pad a tensor to target height and width.

    :param input_tensor: input tensor to be zero padded
    :param target_height: target total height after zero padding (Default: 5120)
    :param target_width: target total width after zero padding (Default: 5120)
    :param shift_downwards: an option to shift the content downwards when zero padding (Default: 0)
    :returns pytorch.tensor: the zero padded tensor
    """
    zero_pad_left = int((target_width - input_tensor.shape[-1]) / 2)
    zero_pad_right = target_width - input_tensor.shape[-1] - zero_pad_left
    zero_pad_top = int((target_height - input_tensor.shape[-2]) / 2) + shift_downwards
    zero_pad_bottom = target_height - input_tensor.shape[-2] - zero_pad_top
    padded_image = torch.nn.ZeroPad2d((zero_pad_left, zero_pad_right, zero_pad_top, zero_pad_bottom))(input_tensor)
    return padded_image.to(DATATYPE)


def hologram_encoding_gamma_correct_linear(gamma_correct_me, pre_gamma_grey_values=None, post_gamma_grey_values=None):
    """
    (Deprecated, for reference only)
    Originally written by Andrew Kadis
    """
    import scipy.interpolate
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


def save_hologram_and_its_recons(hologram, distances=[9999999], alg_name="unspecified_algorithm", pitch_size=DEFAULT_PITCH_SIZE, wavelength=DEFAULT_WAVELENGTH, filename_note='', recon_dynamic_range=None):
    """
    Save a hologram and its reconstructions at specified distances to the Output folder.

    :param hologram: the hologram tensor of complex values
    :param distances: list of distances in meters (Default: [9999999])
    :param alg_name: string specifying the algorithm name, for clearer file naming (Default: "unspecified_algorithm")
    :param pitch_size: pitch size of the spatial light modulator (SLM) which displays the hologram (Default: DEFAULT_PITCH_SIZE)
    :param wavelength: wavelength of the light source (Default: DEFAULT_WAVELENGTH)
    :param filename_note: option to add extra note to the end of the file (Default: "")
    :param recon_dynamic_range: the dynamic range when saving image files (default: None, which will be assigned to the maximum value automatically)
    """
    # print("phase mean: ", hologram.angle().mean().item(), "max: ", hologram.angle().max().item(), "min: ", hologram.angle().min().item())
    phase_hologram = hologram.detach().cpu().angle() % (2*math.pi) / (2*math.pi) * 255.0
    # gamma_corrected_phase_hologram = hologram_encoding_gamma_correct_linear(phase_hologram)
    # print("encoded holo mean: ", phase_hologram.mean().item(), "max: ", phase_hologram.max().item(), "min: ", phase_hologram.min().item())
    if not os.path.isdir(os.path.join('Output', alg_name)):
        os.makedirs(os.path.join('Output', alg_name))
    save_image(os.path.join('Output', alg_name, '{}_holo_{:.2f}m{}'.format(alg_name, distances[0], filename_note)), phase_hologram, 255.0)

    for distance in distances:
        reconstruction_abs = fresnel_propagation(hologram, distance, pitch_size=pitch_size, wavelength=wavelength).abs()
        reconstruction_normalised = energy_conserve(reconstruction_abs)
        save_image(os.path.join('Output', alg_name, '{}_recon_{:.2f}m{}'.format(alg_name, distance, filename_note)), reconstruction_normalised.detach().cpu(), recon_dynamic_range)


def add_zeros_below(target_field):
    """
    Add zeros below target image (useful for binary SLM).

    :param torch.tensor target_field: input tensor representing an image
    :returns: the image being added zeros of same dimensions below
    """
    return torch.cat((target_field, torch.zeros(target_field.size())), -2)


def add_up_side_down_replica_below(target_field):
    """
    Add an up side down replica below target image (useful for binary SLM)

    :param torch.tensor target_field: input tensor representing an image
    :returns: the image being added an up side down replica below
    """
    target_up_side_down = torch.rot90(target_field, 2, [-2, -1])
    return torch.cat((target_field, target_up_side_down), -2)


def flip_left_right(target_field):
    """
    Flip the image left and right

    :param torch.tensor target_field: input tensor representing an image
    :returns: the image flipped left and right
    """
    return torch.fliplr(target_field)


def fraunhofer_propagation(hologram, *_):
    """
    Fraunhofer propagation

    :param torch.tensor hologram: input tensor representing the hologram
    :param *_: stash away unused variables (e.g. distance, pitch_size, wavelength)
    :returns: the reconstruction of the hologram at far field
    """
    return torch.fft.fftshift(torch.fft.fft2(torch.fft.fftshift(hologram)))


def fresnel_propagation(hologram, distance=2, pitch_size=DEFAULT_PITCH_SIZE, wavelength=DEFAULT_WAVELENGTH):
    """
    Fresnel propagation

    Equivalent MatLab code written by Fan Yang (fy255@cam.ac.uk) in 2019:
        function [u2] = ForwardPropagation(u1,pitch,lambda,z)
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

    :param torch.tensor hologram: input tensor representing the hologram in complex form
    :param distance: distance of Fresnel propagation in meters (Default: 2)
    :param pitch_size: pitch size of the spatial light modulator (SLM) which displays the hologram (Default: DEFAULT_PITCH_SIZE)
    :param wavelength: wavelength of the light source (Default: DEFAULT_WAVELENGTH)
    :returns: the reconstruction of the hologram at the requested distance (complex valued)
    """
    distance = torch.tensor([distance]).to(hologram.device)
    holo_height = hologram.shape[-2]
    holo_width = hologram.shape[-1]

    x_lin = torch.linspace(-pitch_size * (holo_width - 1) / 2,
                           pitch_size * (holo_width - 1) / 2,
                           holo_width,
                           dtype=DATATYPE).to(hologram.device)
    y_lin = torch.linspace(-pitch_size * (holo_height - 1) / 2,
                           pitch_size * (holo_height - 1) / 2,
                           holo_height,
                           dtype=DATATYPE).to(hologram.device)
    y_meters, x_meters = torch.meshgrid(y_lin, x_lin, indexing='ij')

    h = (-1j)*torch.exp(-1j/(wavelength*distance) * math.pi * (x_meters**2 + y_meters**2))
    return torch.fft.fftshift(torch.fft.fft2(torch.fft.fftshift(h * hologram)))



def fresnel_backward_propagation(field, distance=2, pitch_size=DEFAULT_PITCH_SIZE, wavelength=DEFAULT_WAVELENGTH):
    """
    Fresnel backward propagation

    :param torch.tensor field: input tensor representing the replay field
    :param distance: distance of Fresnel propagation in meters (Default: 2)
    :param pitch_size: pitch size of the spatial light modulator (SLM) which displays the hologram (Default: DEFAULT_PITCH_SIZE)
    :param wavelength: wavelength of the light source (Default: DEFAULT_WAVELENGTH)
    :returns: the hologram of the reconstruction at the requested distance (complex valued)
    """
    distance = torch.tensor([distance]).to(field.device)
    holo_height = field.shape[-2]
    holo_width = field.shape[-1]

    x_lin = torch.linspace(-pitch_size * (holo_width - 1) / 2,
                           pitch_size * (holo_width - 1) / 2,
                           holo_width,
                           dtype=DATATYPE).to(field.device)
    y_lin = torch.linspace(-pitch_size * (holo_height - 1) / 2,
                           pitch_size * (holo_height - 1) / 2,
                           holo_height,
                           dtype=DATATYPE).to(field.device)
    y_meters, x_meters = torch.meshgrid(y_lin, x_lin, indexing='ij')

    h = (-1j)*torch.exp(-1j/(wavelength*distance) * math.pi * (x_meters**2 + y_meters**2))
    U2 = torch.fft.ifftshift(torch.fft.ifft2(torch.fft.ifftshift(field)))
    u1 = U2 / h
    return u1


def generate_quadratic_phase(size=[1280, 1024], scaling=None):
    """
    Generate an initial phase hologram having a quadratic pattern starting from the center.

    :param size: the dimension of the hologram (Default: [1280, 1024])
    :param scaling: a tuple scaling to control the rate of the quadratic increase in two directions (e.g. [0.002, 0.0018]) (Default: None)
    """
    holo_height = size[1]
    holo_width = size[0]

    x_lin = torch.linspace(-(holo_width - 1) / 2, (holo_width - 1) / 2, holo_width)
    y_lin = torch.linspace(-(holo_height - 1) / 2, (holo_height - 1) / 2, holo_height)
    y_meters, x_meters = torch.meshgrid(y_lin, x_lin, indexing='ij')

    if scaling:
        h = scaling[0] * x_meters**2 + scaling[1] * y_meters**2
    else:
        # Given in https://doi.org/10.1364/OE.422115
        h = math.pi/2/size[0] * x_meters**2 + math.pi/2/size[1] * y_meters**2
    return h


def generate_linear_phase(size=[1024, 1024], scaling=0.1):
    """
    Generate an initial phase hologram having a linearly increasing pattern starting from the center.

    :param size: the dimension of the hologram (Default: [1024, 1024])
    :param scaling: a scaling to control the gradient of the linear increase (Default: 0.1)
    """
    holo_height = size[0]
    holo_width = size[1]

    x_lin = torch.linspace(-(holo_width - 1) / 2, (holo_width - 1) / 2, holo_width)
    y_lin = torch.linspace(-(holo_height - 1) / 2, (holo_height - 1) / 2, holo_height)
    y_meters, x_meters = torch.meshgrid(y_lin, x_lin, indexing='ij')

    h = torch.sqrt(x_meters**2 + y_meters**2) * scaling
    return h


def energy_match(field, target_field):
    """
    Match the total energy of field to target_field

    :param field: input tensor to be matched energy
    :param target_field: target tensor to match the energy towards
    :returns: tensor of field having been energy matched to the target field
    """
    return field * torch.sqrt((target_field**2).sum() / (field**2).sum())


def energy_conserve(field, scaling=DEFAULT_ENERGY_CONSERVATION_SCALING):
    """
    Conserve the energy of the tensor to a uniform amplitude tensor of the same dimension.
    The idea of this function is to speed up energy matching, as all targets and reconstructions are matched
    to the same energy, there will be no need to compute sum of squares of target fields anymore.

    :param field: input tensor to be matched energy
    :param scaling: the amplitude of the uniform amplitude tensor of the same dimension as field (Default: 1.0)
    :returns: tensor of field having been energy matched to the target field
    """
    return field * torch.sqrt((scaling * field.size(-1) * field.size(-2)) / (field**2).sum())


def quantize_to_bit_depth(phase_hologram, hologram_quantization_bit_depth):
    """
    Quantize the phase hologram to a designated bit depth.

    :param phase_hologram: input tensor representing the phase hologram
    :param hologram_quantization_bit_depth: the bit depth to quantize the hologram to
    :returns: tensor of phase hologram quantized to the designated bit depth
    """
    int_hologram = torch.round(phase_hologram / math.pi * 2**(hologram_quantization_bit_depth-1))
    # make -pi to pi (e.g. 1 bit binary hologram will change values from [-1, 0, 1] to [0, 1])
    int_hologram[int_hologram == -2**(hologram_quantization_bit_depth-1)] = 2**(hologram_quantization_bit_depth-1)
    return int_hologram / 2**(hologram_quantization_bit_depth-1) * math.pi


def gerchberg_saxton_single_slice(target_field, iteration_number=50, manual_seed_value=0, hologram_quantization_bit_depth=None, distance=None):
    """
    Traditional Gerchberg Saxton algorithm implemented using pytorch for Fraunhofer region (far field).

    :param target_field: tensor for target image
    :param iteration_number: number of iterations (Default: 50)
    :param manual_seed_value: manual seed for random hologram generation (Default: 0)
    :param hologram_quantization_bit_depth: quantize the hologram to designated bit depth if needed (Default: None)
    :param distance: distance of Fresnel propagation in meters (Default: None)
    :returns: resultant hologram, list of NMSE
    """
    torch.manual_seed(manual_seed_value)
    device = torch.device("cuda")
    target_field = target_field.to(device)

    A = torch.exp(1j * ((torch.rand(target_field.size()) * 2 - 1) * math.pi).to(DATATYPE)).to(device)
    # A = torch.exp(1j * torch.zeros(target_field.size()).to(DATATYPE)).to(device)
    GS_NMSE_list = []
    holo_entropy_list = []
    for i in range(iteration_number):
        phase_hologram = A.angle()
        if hologram_quantization_bit_depth:
            phase_hologram = quantize_to_bit_depth(phase_hologram, hologram_quantization_bit_depth)
            # Compute entropy of the hologram (see https://doi.org/10.1117/12.3021882)
            value, counts = np.unique(phase_hologram.cpu().numpy(), return_counts=True)
            holo_entropy_list.append(scipy.stats.entropy(counts, base=2))
        A = torch.exp(1j * phase_hologram)

        if distance:
            E = fresnel_propagation(A, distance)
        else:
            E = torch.fft.fftshift(torch.fft.fft2(torch.fft.fftshift(A)))
        E_norm = energy_match(E.abs(), target_field)
        # save_image(".\\Output\\recon_bit_depth_{}".format(quantize_to_bit_depth), E_norm.detach().cpu())
        # GS_NMSE_list.append((torch.nn.MSELoss(reduction="mean")(E_norm, target_field)).item() / (target_field**2).sum().item())
        # GS_NMSE_list.append(torch.nn.KLDivLoss(reduction="mean")(torch.flatten(E_norm / target_field.max()).expand(1, -1), torch.flatten(target_field / target_field.max()).expand(1, -1)).item())
        nmse_i = (((E_norm - target_field)**2).mean() / (target_field**2).sum()).item()
        # nmse_i = pytorch_msssim.ssim(E_norm.expand(1, -1, -1, -1), target_field.expand(1, -1, -1, -1)).item()

        GS_NMSE_list.append(nmse_i)
        E = target_field * torch.exp(1j * E.angle())
        if distance:
            A = fresnel_backward_propagation(E, distance)
        else:
            A = torch.fft.ifftshift(torch.fft.ifft2(torch.fft.ifftshift(E)))

    # save_image('.\\Output\\recon_bit_depth_{}'.format(hologram_quantization_bit_depth), E_norm.detach().cpu(), target_field.max().cpu())
    # save_image('.\\Output\\hologram_bit_depth_{}'.format(hologram_quantization_bit_depth), phase_hologram.detach().cpu(), math.pi)
    return A, GS_NMSE_list, holo_entropy_list


def gerchberg_saxton_multiple_slice_phase_add(target_fields, distances, iteration_number=50, manual_seed_value=0, pitch_size=DEFAULT_PITCH_SIZE, wavelength=DEFAULT_WAVELENGTH):
    """
    Generate holograms for target_fields individually and then add up holograms to form the final phase hologram.

    :param target_fields: tensor for target images (e.g. loaded by load_target_images)
    :param distances: images distances in meters (e.g. [1, 2, 3])
    :param iteration_number: number of iterations (Default: 50)
    :param manual_seed_value: manual seed for random hologram generation (Default: 0)
    :param pitch_size: pitch size of the spatial light modulator (SLM) (Default: DEFAULT_PITCH_SIZE)
    :param wavelength: wavelength of the light source (Default: DEFAULT_WAVELENGTH)
    :returns: resultant hologram, list of NMSE
    """
    torch.manual_seed(manual_seed_value)
    device = torch.device("cuda")
    target_fields = target_fields.to(DATATYPE).to(device)
    A_total = torch.zeros(target_fields[0].size()).to(DATATYPE).to(device)

    for target_i in range(len(target_fields)):
        E = target_fields[target_i] * torch.exp(1j * ((torch.rand(target_fields[target_i].size()) * 2 - 1) * math.pi).to(DATATYPE)).to(device)
        for i in range(iteration_number):
            A = fresnel_backward_propagation(E, distances[target_i], pitch_size=pitch_size, wavelength=wavelength)
            A = torch.exp(1j * A.angle())
            E = fresnel_propagation(A, distances[target_i], pitch_size=pitch_size, wavelength=wavelength)
            E = target_fields[target_i] * torch.exp(1j * E.angle())

        A_total = A_total + A
    return A_total


def gerchberg_saxton_3d_sequential_slicing(target_fields, distances, iteration_number=50, weighting=0, zero_cap=None, time_limit=None, pitch_size=DEFAULT_PITCH_SIZE, wavelength=DEFAULT_WAVELENGTH):
    """
    Implementation of Gerchberg Saxton algorithm with sequential slicing (SS) for Fresnel region.
    It propagates the hologarm to a single slice at each iteration, sequentially looping through all slices.

    :param target_fields: tensor for target images (e.g. loaded by load_target_images)
    :param distances: images distances in meters (e.g. [1, 2, 3])
    :param iteration_number: number of iterations (Default: 50)
    :param weighting: weighting of current amplitude when applying the target amplitude constraint (Default: 0.001)
        (this is an implementation of DCGS: https://doi.org/10.1364/OE.27.008958)
    :param zero_cap: Allowance of the noise for the zero values in the target image (Default: None)
    :param time_limit: time limit of the run if any (Default: None)
    :param pitch_size: pitch size of the spatial light modulator (SLM) (Default: DEFAULT_PITCH_SIZE)
    :param wavelength: wavelength of the light source (Default: DEFAULT_WAVELENGTH)
    :returns: resultant hologram, list of NMSE and the time recorded respectively
    """
    time_start = time.time()

    torch.manual_seed(0)
    device = torch.device("cuda")
    target_fields = target_fields.to(device)

    amplitude = torch.ones(target_fields[0].size(), requires_grad=False).to(DATATYPE).to(device)
    phase = ((torch.rand(target_fields[0].size()) * 2 - 1) * math.pi).to(DATATYPE).to(device)
    A = amplitude * torch.exp(1j * phase)

    nmse_lists = []
    for distance in distances:
        nmse_lists.append([])
    time_list = []

    for i in range(iteration_number):
        for d_index, distance in enumerate(distances):
            reconstruction_abs = fresnel_propagation(A, distance, pitch_size=pitch_size, wavelength=wavelength).abs()
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
        E = fresnel_propagation(A, distances[slice_number], pitch_size=pitch_size, wavelength=wavelength)
        if weighting != 0:
            E = (E.abs() * weighting + target_fields[slice_number] * (1-weighting)) * torch.exp(1j * E.angle())
        elif zero_cap:
            E_abs = E.abs()
            E_zeros = E_abs * (target_fields[slice_number].max() - target_fields[slice_number])
            E_zeros[E_zeros > zero_cap] = zero_cap
            E = (E_zeros + target_fields[slice_number]) * torch.exp(1j * E.angle())
        else:
            E = target_fields[slice_number] * torch.exp(1j * E.angle())
        A = fresnel_backward_propagation(E, distances[slice_number], pitch_size=pitch_size, wavelength=wavelength)
        A = torch.exp(1j * A.angle())

    return A.detach().cpu(), nmse_lists, time_list


def optim_cgh_3d(target_fields, distances, sequential_slicing=False, wavelength=DEFAULT_WAVELENGTH, pitch_size=DEFAULT_PITCH_SIZE,
                 save_progress=False, iteration_number=20, cuda=False, learning_rate=0.1, record_all_nmse=True, optimise_algorithm="LBFGS",
                 grad_history_size=10, loss_function=torch.nn.MSELoss(reduction="sum"), energy_conserv_scaling=DEFAULT_ENERGY_CONSERVATION_SCALING, time_limit=None,
                 initial_phase='random', smooth_holo_kernel_size=None, holo_bit_depth=None):
    """
    Carry out L-BFGS optimisation of CGH for a 3D target consisted of multiple slices of 2D images at different distances.
    Please make reference of this function to: https://doi.org/10.1364/JOSAA.478430

    :param target_fields: tensor for target images
    :param distances: a list of image distances (e.g. [1.1 2.3 3.1 4.3]), whose length matches the number of target images.
    :param sequential_slicing: switch to enable/disable sequential slicing (Default: False)
        If sequential_slicing is True, Loss is calculated summing reconstructions at all distances.
        If sequential_slicing is False, Loss is calculated for reconstruction at each distance in turn for each iteration.
    :param wavelength: wavelength of the light source (Default: DEFAULT_WAVELENGTH)
    :param pitch_size: pitch size of the spatial light modulator (SLM) (Default: DEFAULT_PITCH_SIZE)
    :param save_progress: decide whether to save progress of every iteration to files (Default: False)
    :param iteration_number: number of iterations (Default: 20)
    :param cuda: decide whether to use CUDA, use CPU otherwise (Default: False)
    :param learning_rate: set the parameter 'lr' of torch.optim (Default: 0.1)
    :param record_all_nmse: decide whether to record NSME throughout the run, set it to False for speedy runs (Default: True)
    :param optimise_algorithm: select optimisation algorithm from "LBFGS", "SGD" and "ADAM", case insensitive (Default: "LBFGS")
    :param grad_history_size: gradient history size for LBFGS algorithm only (Default: 10)
    :param loss_function: the objective function to minimise (Default: torch.nn.MSELoss(reduction="sum"))
    :param energy_conserv_scaling: all target images and reconstructions are conserved to have the same total energy (Default: 1.0)
    :param time_limit: time limit of the run if any (Default: None)
    :param initial_phase: initial phase to start from, select from: "random", "quadratic" and "linear" (Default: "random")
    :param smooth_holo_kernel_size: the hologram will be smoothed by a Gaussian filter of this kernel size (Default: None)
    :returns: resultant hologram, list of NMSE and the time recorded respectively
    """
    time_start = time.time()
    torch.cuda.empty_cache()
    torch.manual_seed(0)
    device = torch.device("cuda" if cuda else "cpu")
    target_fields = target_fields.to(device)

    # Fixed unit amplitude
    amplitude = torch.ones(target_fields[0].size(), requires_grad=False).to(DATATYPE).to(device)

    # Variable phase
    if initial_phase.lower() in ['random', 'rand']:
        # Random initial phase within [-pi, pi]
        phase = ((torch.rand(target_fields[0].size()) * 2 - 1) * math.pi).to(DATATYPE).detach().to(device).requires_grad_()
    elif initial_phase.lower() in ['linear', 'lin']:
        # linear initial phase
        phase = (torch.ones(target_fields[0].size()) * generate_linear_phase([target_fields[0].shape[-2], target_fields[0].shape[-1]], 0.5)).to(DATATYPE).detach().to(device).requires_grad_()
    elif initial_phase.lower() in ['quadratic', 'quad']:
        # linear initial phase
        phase = (torch.ones(target_fields[0].size()) * generate_quadratic_phase([target_fields[0].shape[-1],
                 target_fields[0].shape[-2]])).to(DATATYPE).detach().to(device).requires_grad_()
    else:
        raise ValueError("The required initial phase is not recognised!")

    # Decide optimisation algorithm
    if optimise_algorithm.lower() in ["lbfgs", "l-bfgs"]:
        optimiser = torch.optim.LBFGS([phase], lr=learning_rate, history_size=grad_history_size)
    elif optimise_algorithm.lower() in ["sgd", "gd"]:
        optimiser = torch.optim.SGD([phase], lr=learning_rate)
    elif optimise_algorithm.lower() == "adam":
        optimiser = torch.optim.Adam([phase], lr=learning_rate)
    else:
        raise ValueError("Optimiser is not recognised!")

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
        elif holo_bit_depth:
            # quantize the phase to certain bit depth
            hologram = amplitude * torch.exp(1j * quantize_to_bit_depth(phase, holo_bit_depth))
        else:
            hologram = amplitude * torch.exp(1j * phase)

        if sequential_slicing:
            # Propagate hologram for one distance only
            slice_number = i % len(target_fields)
            reconstruction_abs = fresnel_propagation(hologram, distances[slice_number], pitch_size, wavelength).abs()
            reconstruction_normalised = energy_conserve(reconstruction_abs, energy_conserv_scaling)

            # Calculate loss for the single slice
            loss = loss_function(torch.flatten(reconstruction_normalised / target_fields[slice_number].max()).expand(1, -1),
                                 torch.flatten(target_fields[slice_number] / target_fields[slice_number].max()).expand(1, -1))

        else:
            # Propagate hologram for all distances
            reconstructions_list = []
            for index, distance in enumerate(distances):
                reconstruction_abs = fresnel_propagation(hologram, distance=distance, pitch_size=pitch_size, wavelength=wavelength).abs()
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
                reconstruction_abs = fresnel_propagation(hologram, distance, pitch_size, wavelength).abs()
                reconstruction_normalised = energy_conserve(reconstruction_abs, energy_conserv_scaling)
                # reconstruction_normalised = energy_match(reconstruction_abs, target_fields[slice_number])
                nmse_value = (torch.nn.MSELoss(reduction="mean")(reconstruction_normalised, target_fields[index])).item() / (target_fields[index]**2).sum()
                nmse_lists[index].append(nmse_value.data.tolist())
                if save_progress:
                    save_hologram_and_its_recons(hologram, distances, optimise_algorithm, filename_note = str(i))
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
    """
    Freeman Projector Encoder for Multi Frame Holograms Batched Optimisation.
    It outputs a function that encodes the 24 binary-phase hologram sub-frames into a single 3-channel 8-bit image file.

    :param holograms: tensor for a stack of holograms
    :param alg_name: string specifying the algorithm name, for clearer file naming (Default: "MultiFrame")
    :param filename_note: option to add extra note to the end of the file (Default: "")
    """
    binary_phase_holograms = torch.round(holograms.detach().cpu().angle() / math.pi)
    # print(binary_phase_holograms.max(), binary_phase_holograms.min(), binary_phase_holograms.mean())
    reconstruction_abs = fresnel_propagation(torch.exp(1j * binary_phase_holograms * math.pi), distance=999).abs()
    reconstruction_mean = energy_conserve(reconstruction_abs.mean(dim=0))
    save_image(os.path.join('Output', 'MultiFrame', '{}_binaryholo_recon_mean_test'.format('MultiFrame')), reconstruction_mean.detach().cpu())

    freeman_hologram = torch.zeros(3, holograms.size(-2), holograms.size(-1))
    # Freeman projector takes image of 3 channel * 8 bit depth, totaling 24 bit planes
    for channel_i in range(3):
        channel_temp_sum = torch.zeros(1, holograms.size(-2), holograms.size(-1))
        for subframe_i in range(8):
            current_frame_i = (channel_i * 8 + subframe_i) % holograms.size(0)
            channel_temp_sum += binary_phase_holograms[current_frame_i] * 2**subframe_i
        freeman_hologram[channel_i] = channel_temp_sum.detach()

    save_image(os.path.join('Output', alg_name, '{}_freeman_holo{}'.format(alg_name, filename_note)), freeman_hologram)
    return


def save_multi_frame_holograms_and_their_recons(holograms, distance=None, reconstruction_subframes=None, recon_dynamic_range=None, alg_name='MultiFrame', filename_note='', pitch_size=DEFAULT_PITCH_SIZE, wavelength=DEFAULT_WAVELENGTH):
    """
    Save the multi-frame holograms and their reconstructions to files.

    :param holograms: tensor for a stack of holograms
    :param distance: distance of Fresnel propagation in meters (Default: None)
    :param reconstruction_subframes: tensor for a stack of reconstructions (Default: None)
    :param recon_dynamic_range: dynamic range of the reconstruction (Default: None)
    :param alg_name: string specifying the algorithm name, for clearer file naming (Default: "MultiFrame")
    :param filename_note: option to add extra note to the end of the file (Default: "")
    :param pitch_size: pitch size of the spatial light modulator (SLM) (Default: DEFAULT_PITCH_SIZE)
    :param wavelength: wavelength of the light source (Default: DEFAULT_WAVELENGTH)
    """
    if not os.path.isdir(os.path.join('Output', '{}'.format(alg_name))):
        os.makedirs(os.path.join('Output', '{}'.format(alg_name)))
    freeman_projector_encoding(holograms, filename_note = filename_note)

    if reconstruction_subframes is None:
        for hologram_i, hologram in enumerate(holograms):
            # print("phase mean: ", hologram.angle().mean().item(), "max: ", hologram.angle().max().item(), "min: ", hologram.angle().min().item())
            phase_hologram = hologram.detach().cpu().angle() % (2*math.pi) / (2*math.pi) * 255.0
            # print("encoded holo mean: ", phase_hologram.mean().item(), "max: ", phase_hologram.max().item(), "min: ", phase_hologram.min().item())
            save_image(os.path.join('Output', alg_name, '{}_holo{}_d{}{}'.format(alg_name, hologram_i, distance, filename_note)), phase_hologram, 255.0)

            # gamma_corrected_phase_hologram = hologram_encoding_gamma_correct_linear(phase_hologram)
            # print("Sony holo mean: ", gamma_corrected_phase_hologram.mean().item(), "max: ", gamma_corrected_phase_hologram.max().item(), "min: ", gamma_corrected_phase_hologram.min().item())
            # save_image('.\Output\{0}\{0}_sony_holo{1}'.format(alg_name, filename_note), gamma_corrected_phase_hologram, 255.0)
            reconstruction_abs = fresnel_propagation(holograms, distance=distance, pitch_size=pitch_size, wavelength=wavelength).abs()
            reconstruction_normalised = energy_conserve(reconstruction_abs)
            save_image(os.path.join('Output', alg_name, '{}_recon{}_d{}{}'.format(alg_name, hologram_i, distance, filename_note)), reconstruction_normalised.detach().cpu(), recon_dynamic_range)
    else:
        for hologram_i, hologram in enumerate(holograms):
            phase_hologram = hologram.detach().cpu().angle() % (math.pi) / (math.pi) * 255.0
            save_image(os.path.join('Output', alg_name, '{}_holo{}_d{}{}'.format(alg_name, hologram_i, distance, filename_note)), phase_hologram, 255.0)

        for reconstruction_i, reconstruction in enumerate(reconstruction_subframes):
            reconstruction_normalised = energy_conserve(reconstruction)
            save_image(os.path.join('Output', alg_name, '{}_recon{}_d{}{}'.format(alg_name, reconstruction_i, distance, filename_note)), reconstruction_normalised.detach().cpu(), recon_dynamic_range)


def multi_frame_cgh(target_fields, distances, wavelength=DEFAULT_WAVELENGTH, pitch_size=DEFAULT_PITCH_SIZE,
                    iteration_number=20, cuda=False, learning_rate=0.1, optimise_algorithm="LBFGS",
                    grad_history_size=10, loss_function=torch.nn.MSELoss(reduction="sum"), is_binary_phase=True,
                    energy_conserv_scaling=DEFAULT_ENERGY_CONSERVATION_SCALING, time_limit=None, num_frames=24):
    """
    Multi Frame Holograms Batched Optimization (MFHBO)
    Please make reference of this function to: https://doi.org/10.1038/s41598-024-70428-0

    :param target_fields: tensor for target images
    :param distances: a list of image distances (e.g. [1.1 2.3 3.1 4.3]), whose length matches the number of target images.
    :param wavelength: wavelength of the light source (Default: DEFAULT_WAVELENGTH)
    :param pitch_size: pitch size of the spatial light modulator (SLM) (Default: DEFAULT_PITCH_SIZE)
    :param iteration_number: number of iterations (Default: 20)
    :param cuda: decide whether to use CUDA, use CPU otherwise (Default: False)
    :param learning_rate: set the parameter 'lr' of torch.optim (Default: 0.1)
    :param optimise_algorithm: select optimisation algorithm from "LBFGS", "SGD" and "ADAM", case insensitive (Default: "LBFGS")
    :param grad_history_size: gradient history size for LBFGS algorithm only (Default: 10)
    :param loss_function: the objective function to minimise (Default: torch.nn.MSELoss(reduction="sum"))
    :param is_binary_phase: a boolean flag to set the binary quantisation of the phase holograms (using the sigmoid function) (Default: True)
    :param energy_conserv_scaling: all target images and reconstructions are conserved to have the same total energy (Default: 1.0)
    :param time_limit: time limit of the run if any (Default: None)
    :param num_frames: the number of multiplexing frames of phase holograms (Default: 24)
    :returns: resultant holograms batch, list of NMSE and the time recorded respectively
    """
    time_start = time.time()
    torch.cuda.empty_cache()
    torch.manual_seed(0)
    device = torch.device("cuda" if cuda else "cpu")
    target_fields = target_fields.to(device)

    # Fixed unit amplitude
    amplitude = torch.ones(target_fields[0].size(), requires_grad=False).to(DATATYPE).to(device)

    # Initial multi-frame phases
    phases = ((torch.rand([num_frames] + list(target_fields[0].size())) * 2 - 1) * math.pi).to(DATATYPE).detach().to(device).requires_grad_()
    # phases = torch.stack([fresnel_backward_propagation(target_fields[0] * torch.exp(1j*generate_quadratic_phase(target_fields[0].size())).to(DATATYPE).detach().to(device), distance=distances[0], pitch_size=pitch_size, wavelength=wavelength).angle() for i in range(num_frames)]).to(DATATYPE).detach().to(device).requires_grad_()
    # phases = fresnel_backward_propagation(target_fields[0] * torch.exp(1j * torch.rand([num_frames] + list(target_fields[0].size()))).to(device), distance=distances[0], pitch_size=pitch_size, wavelength=wavelength).angle().to(DATATYPE).detach().requires_grad_()

    # Decide optimisation algorithm
    if optimise_algorithm.lower() in ["lbfgs", "l-bfgs"]:
        optimiser = torch.optim.LBFGS([phases], lr=learning_rate, history_size=grad_history_size)
    elif optimise_algorithm.lower() in ["sgd", "gd"]:
        optimiser = torch.optim.SGD([phases], lr=learning_rate)
    elif optimise_algorithm.lower() == "adam":
        optimiser = torch.optim.Adam([phases], lr=learning_rate)
    else:
        raise NameError("Optimiser is not recognised!")

    time_list = []
    nmse_list = []

    for i in range(iteration_number):
        print('iteration {} out of {}'.format(i, iteration_number))
        optimiser.zero_grad()

        # If the SLM is binary phase, apply binary phase quantization
        if is_binary_phase:
            holograms = amplitude * torch.exp(1j * torch.nn.Sigmoid()(phases / math.pi) * math.pi)
        else:
            holograms = amplitude * torch.exp(1j * phases)

        # Propagate hologram for all distances
        reconstructions_list = []
        for index, distance in enumerate(distances):
            reconstructions = fresnel_propagation(holograms, distance=distance, pitch_size=pitch_size, wavelength=wavelength).abs() # propagate for each sub frame
            if i == iteration_number - 1: # save final subframes and subreconstructions at the last iteration
                save_multi_frame_holograms_and_their_recons(holograms, distance=distance, reconstruction_subframes=reconstructions, recon_dynamic_range=target_fields.detach().cpu().max(), alg_name='MultiFrame', filename_note='_{}frames'.format(num_frames))
            reconstructions = energy_conserve(reconstructions.mean(dim=0), energy_conserv_scaling) # take the average among all sub frames
            if i == iteration_number - 1: # save final average reconstructions at the last iteration
                save_image(os.path.join('Output', 'MultiFrame', '{}_recon_mean_d{}_{}frames'.format('MultiFrame', distance, num_frames)), reconstructions.detach().cpu(), target_fields.detach().cpu().max())
            reconstructions_list.append(reconstructions)
        reconstructions = torch.stack(reconstructions_list)

        # Calculate loss for all slices (stacked in reconstructions)
        loss = loss_function(torch.flatten(reconstructions / target_fields.max()).expand(1, -1),
                             torch.flatten(target_fields / target_fields.max()).expand(1, -1))

        loss.backward(retain_graph=True)

        # Record NMSE for quality analysis, or uncomment the code to use psnr and ssim instead
        nmse_value = (torch.nn.MSELoss(reduction="mean")(reconstructions, target_fields)).item() / (target_fields**2).sum().data.item()
        # psnr_value = 20 * torch.log10(target_fields.max() / torch.sqrt(torch.nn.MSELoss(reduction="mean")(reconstructions, target_fields))).item()
        # ssim_value = pytorch_msssim.ssim(reconstructions, target_fields).item()
        nmse_list.append(nmse_value)

        # Record the time stamp at each iteration for timing analysis
        time_list.append(time.time() - time_start)
        if time_limit:
            if time_list[-1] >= time_limit:
                break

        def closure():
            return loss
        optimiser.step(closure)

    torch.no_grad()
    holograms = amplitude * torch.exp(1j * phases)
    return holograms.detach().cpu(), nmse_list, time_list


def tipo(target_fields, distances, wavelength=DEFAULT_WAVELENGTH, pitch_size=DEFAULT_PITCH_SIZE,
                 iteration_number=20, cuda=False, learning_rate=0.1, optimise_algorithm="LBFGS",
                 grad_history_size=10, loss_function=torch.nn.MSELoss(reduction="sum"),
                 energy_conserv_scaling=DEFAULT_ENERGY_CONSERVATION_SCALING, time_limit=None,
                 initial_phase='random'):
    """
    Target Image Phase Optimization (TIPO)
    Please make reference of this function to: https://doi.org/10.1117/12.3039305

    :param target_fields: tensor for target images
    :param distances: a list of image distances (e.g. [1.1 2.3 3.1 4.3]), whose length matches the number of target images
    :param wavelength: wavelength of the light source (Default: DEFAULT_WAVELENGTH)
    :param pitch_size: pitch size of the spatial light modulator (SLM) (Default: DEFAULT_PITCH_SIZE)
    :param iteration_number: number of iterations (Default: 20)
    :param cuda: decide whether to use CUDA, use CPU otherwise (Default: False)
    :param learning_rate: set the parameter 'lr' of torch.optim (Default: 0.1)
    :param optimise_algorithm: select optimisation algorithm from "LBFGS", "SGD" and "ADAM", case insensitive (Default: "LBFGS")
    :param grad_history_size: gradient history size for LBFGS algorithm only (Default: 10)
    :param loss_function: the objective function to minimise (Default: torch.nn.MSELoss(reduction="sum"))
    :param energy_conserv_scaling: all target images and reconstructions are conserved to have the same total energy (Default: 1.0)
    :param time_limit: time limit of the run if any (Default: None)
    :param initial_phase: initial phase to start from, select from: "random", "quadratic" and "linear" (Default: "random")
    :returns: resultant hologram, list of NMSE and the time recorded respectively
    """

    time_start = time.time()
    torch.cuda.empty_cache()
    torch.manual_seed(0)
    device = torch.device("cuda" if cuda else "cpu")
    target_fields = target_fields.to(device)

    # Variable phase
    if initial_phase.lower() in ['random', 'rand']:
        # Random initial phase within [-pi, pi]
        phase = ((torch.rand(target_fields[0].size()) * 2 - 1) * math.pi).to(DATATYPE).detach().to(device).requires_grad_()
    elif initial_phase.lower() in ['linear', 'lin']:
        # linear initial phase
        phase = (torch.ones(target_fields[0].size()) * generate_linear_phase([target_fields[0].shape[-2], target_fields[0].shape[-1]], 0.5)).to(DATATYPE).detach().to(device).requires_grad_()
    elif initial_phase.lower() in ['quadratic', 'quad']:
        # linear initial phase
        phase = (torch.ones(target_fields[0].size()) * generate_quadratic_phase([target_fields[0].shape[-1],
                 target_fields[0].shape[-2]])).to(DATATYPE).detach().to(device).requires_grad_()
    else:
        raise ValueError("The required initial phase is not recognised!")

    # Decide optimisation algorithm
    if optimise_algorithm.lower() in ["lbfgs", "l-bfgs"]:
        optimiser = torch.optim.LBFGS([phase], lr=learning_rate, history_size=grad_history_size)
    elif optimise_algorithm.lower() in ["sgd", "gd"]:
        optimiser = torch.optim.SGD([phase], lr=learning_rate)
    elif optimise_algorithm.lower() == "adam":
        optimiser = torch.optim.Adam([phase], lr=learning_rate)
    else:
        raise ValueError("Optimiser is not recognised!")

    time_list = []
    nmse_lists = []
    for distance in distances:
        nmse_lists.append([])

    for i in range(iteration_number):
        optimiser.zero_grad()
        target_complex_field = target_fields * torch.exp(1j * phase)
        if not os.path.isdir(os.path.join('Output', 'TIPO')):
            os.makedirs(os.path.join('Output', 'TIPO'))

        save_image(os.path.join('Output', 'TIPO', 'TIPO_target_phase{}_d{}'.format(i, distance)), phase.detach().cpu())

        holograms = fresnel_backward_propagation(target_complex_field, distance=distances[0], pitch_size=pitch_size, wavelength=wavelength)
        phase_holograms = holograms.angle()
        save_image(os.path.join('Output', 'TIPO', 'TIPO_holo{}_d{}'.format(i, distance)), phase_holograms[0].detach().cpu())
        # reconstructions = energy_conserve(fresnel_propagation(torch.exp(1j * phase_holograms), distance=distances[0], pitch_size=pitch_size, wavelength=wavelength).abs())
        reconstructions = fresnel_propagation(torch.exp(1j * phase_holograms), distance=distances[0], pitch_size=pitch_size, wavelength=wavelength)
        save_image(os.path.join('Output', 'TIPO', 'TIPO_recon_phase{}_d{}'.format(i, distance)), reconstructions[0].angle().detach().cpu())
        reconstructions = energy_conserve(reconstructions.abs(), energy_conserv_scaling)

        if not os.path.isdir(os.path.join('Output', 'TIPO')):
            os.makedirs(os.path.join('Output', 'TIPO'))
        save_image(os.path.join('Output', 'TIPO', 'TIPO_recon{}_d{}'.format(i, distance)), reconstructions[0].detach().cpu())
        loss = loss_function(torch.flatten(reconstructions / target_fields.max()).expand(1, -1),
                             torch.flatten(target_fields / target_fields.max()).expand(1, -1))#loss_function(hologram.abs(), torch.ones(size=hologram.size()).to(device)*hologram.abs().mean())

        loss.backward(retain_graph=True)


        nmse_value = (torch.nn.MSELoss(reduction="mean")(reconstructions, target_fields)).item() / (target_fields**2).sum().data.item()
        nmse_lists[0].append(nmse_value)
        time_list.append(time.time() - time_start)
        if time_limit:
            if time_list[-1] >= time_limit:
                break

        def closure():
            return loss
        optimiser.step(closure)

    torch.no_grad()

    return holograms.detach().cpu(), nmse_lists, time_list