import os
import time
import math
import torch
import torchvision
import numpy as np
from PIL import Image
'''
import stl

from mpl_toolkits import mplot3d
from matplotlib import pyplot



stl_mesh = stl.mesh.Mesh.from_file("Target_images/Utah_teapot_(solid).stl")

# Create a new plot
figure = pyplot.figure()
axes = mplot3d.Axes3D(figure)

# Load the STL files and add the vectors to the plot
axes.add_collection3d(mplot3d.art3d.Poly3DCollection(stl_mesh.vectors))

# Auto scale to the mesh size
scale = stl_mesh.points.flatten()
axes.auto_scale_xyz(scale, scale, scale)

# Show the plot to the screen
pyplot.show
'''

import trimesh

mesh = trimesh.load_mesh("Target_images/Utah_teapot_(solid).stl")
# voxelized_mesh = mesh.voxelized(pitch=0.1)

z_extents = mesh.bounds[:,2]
z_levels  = np.arange(*z_extents, step=0.01)
print(len(z_levels))
sections = mesh.section_multiplane(plane_origin=mesh.bounds[0], plane_normal=[0,1,0], heights=z_levels)
# combined = np.sum(sections)

for section_i in range(len(sections)):
    # rasterized_sections = trimesh.path.Path2D.rasterize(sections[section_i], pitch=[0.02,0.02], origin=[-24,-20], resolution=[1920, 1080])
    rasterized_sections = trimesh.path.Path2D.rasterize(sections[section_i], pitch=[0.02,0.02], origin=[-11,-21], resolution=[720, 1280])
    rasterized_sections = rasterized_sections.rotate(-90, expand=1)
    file_name = 'Target_images/Teapot_slices/Teapot_720p_section_{}.png'.format(section_i)
    rasterized_sections.save(file_name)
    # img_file = Image.open(file_name)
    # img_file = img_file.rotate(-90, expand=1)
    # img_file.save(file_name)
