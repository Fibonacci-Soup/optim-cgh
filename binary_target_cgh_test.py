import os
import matplotlib.pyplot as plt
import cgh_toolbox
import torch

ENERGY_CONSERVATION_SCALING = 1.0
images = [os.path.join('Target_images', x) for x in ['USAF-1951.png']]
target_fields = cgh_toolbox.load_target_images(images, energy_conserv_scaling=ENERGY_CONSERVATION_SCALING)
target_fields = target_fields.max()-target_fields

hologram, nmse_list, time_list = cgh_toolbox.gerchberg_saxton_3d_sequential_slicing(target_fields, [9999], iteration_number=100, zero_cap=0.5)

reconstruction_abs = cgh_toolbox.energy_conserve(cgh_toolbox.fresnel_propergation(hologram, 9999).abs(), ENERGY_CONSERVATION_SCALING)
cgh_toolbox.save_image('Temp', reconstruction_abs)

plt.hist(torch.flatten(reconstruction_abs * (target_fields.max()-target_fields)), bins=100)
plt.show()