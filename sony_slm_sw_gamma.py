
# -*- coding: utf-8 -*-
"""
Python Script to handle the SLM calibration of the Sony SLM
"""

# import initExample ## Add path to library (just for examples; you do not need this)

# # from PyQt5.QtWidgets import QDesktopWidget
# # from pyqtgraph.Qt import QtGui, QtCore
# import numpy as np
# # from scipy.fft import fft2, fftshift
# # import scipy as sp
# # import pyqtgraph as pg
# # import pyqtgraph.opengl as gl
# from PIL import Image

# from tkinter.tix import MAX
# import cv2
# import glob
# import numpy as np
# from matplotlib import pyplot as plt 

# from scipy.interpolate import interp1d

import numpy as np
from scipy import interpolate

# See OneNote for where these values come from
# 5-point
# pre_gamma_grey_values  = [0, 63, 127, 191, 255]
# pre_gamma_grey_values  = [0, 63, 127, 191, 255]
# post_gamma_grey_values = [0, 25, 88,  197, 252]
# 9-point
# pre_gamma_grey_values  = [0, 31, 63, 95, 127, 159, 191, 223, 255]
pre_gamma_grey_values  = [0, 15, 31, 47, 63, 79, 95, 111, 127, 143, 159, 175, 191, 207, 223, 239, 255]
post_gamma_grey_values = [0, 10, 15, 19, 24, 29, 39, 54,  72,  97,  124, 150, 172, 193, 213, 232, 241]



def gamma_correct_linear(gamma_correct_me):

    # Numpy Option - limited to linear
    # Lookup gamma-corrected value using numpy's linear interpolation function
    # interpolated_values = np.interp(gamma_correct_me, pre_gamma_grey_values, post_gamma_grey_values)

    # Scipy Option - can do cubic
    inter_func = interpolate.interp1d(pre_gamma_grey_values, post_gamma_grey_values, kind='quadratic')
    interpolated_values = inter_func(gamma_correct_me)

    # Need to round-off to nearest int - need to do a type conversion as per https://stackoverflow.com/questions/55146871/can-numpy-rint-to-return-an-int32
    gamma_corrected_value = np.rint(interpolated_values).astype(np.uint8)
    return gamma_corrected_value
    


