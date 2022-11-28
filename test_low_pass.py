import os
import time
import math
import torch
import torchvision
import matplotlib.pyplot as plt
from cgh_toolbox import save_image, fraunhofer_propergation, fresnel_propergation, energy_conserve, gerchberg_saxton, low_pass_filter_2d

target_field = torchvision.io.read_image(r".\Target_images\mandrill.png", torchvision.io.ImageReadMode.GRAY).to(torch.float64)
lpf_img = low_pass_filter_2d(target_field, 0.99)
save_image(r'.\Output_2D_iter\FUCK', lpf_img)