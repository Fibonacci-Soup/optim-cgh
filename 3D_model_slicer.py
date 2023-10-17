import numpy as np
import os
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

mesh = trimesh.load_mesh(os.path.join('Target_images','Utah_teapot_(solid).stl'))
# voxelized_mesh = mesh.voxelized(pitch=0.1)
output_filename = 'Teapot_slices'
z_extents = mesh.bounds[:,2]
z_levels  = np.arange(*z_extents, step=0.01) # 0.01
print(len(z_levels))
sections = mesh.section_multiplane(plane_origin=mesh.bounds[0], plane_normal=[0,1,0], heights=z_levels)
# combined = np.sum(sections)

for section_i in range(len(sections)):
    # rasterized_sections = trimesh.path.Path2D.rasterize(sections[section_i], pitch=[0.02,0.02], origin=[-24,-20], resolution=[1920, 1080])
    # rasterized_sections = trimesh.path.Path2D.rasterize(sections[section_i], pitch=[0.02,0.02], origin=[-11,-21], resolution=[720, 1280])
    if sections[section_i] is not None:
        rasterized_sections = trimesh.path.Path2D.rasterize(sections[section_i], pitch=[0.02,0.02], origin=[-9, -19], resolution=[512, 1024])
        rasterized_sections = rasterized_sections.rotate(-90, expand=1)
        if not os.path.isdir(os.path.join('Target_images',output_filename)):
            os.makedirs(os.path.join('Target_images',output_filename))
        rasterized_sections.save(os.path.join('Target_images',output_filename, output_filename+'section_{}.png'.format(section_i)))
        # img_file = Image.open(file_name)
        # img_file = img_file.rotate(-90, expand=1)
        # img_file.save(file_name)
