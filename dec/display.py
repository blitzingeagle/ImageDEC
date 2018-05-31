from __future__ import division
from __future__ import print_function

import imageset_utils as imgutils

import cv2
import numpy as np

import os
import os.path as path
from glob import glob

# x = [np.random.random((50, 50, 3))] * 40
# y = np.hstack(x)
# y = [y for i in xrange(20)]
# z = np.vstack(y)
#
# cv2.imshow("stack", z)
# cv2.waitKey()
# cv2.destroyAllWindows()

"""
import cv2
import numpy as np

image = cv2.imread('pinocchio.png')
# I just resized the image to a quarter of its original size
image = cv2.resize(image, (0, 0), None, .25, .25)

grey = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
# Make the grey scale image have three channels
grey_3_channel = cv2.cvtColor(grey, cv2.COLOR_GRAY2BGR)

numpy_vertical = np.vstack((image, grey_3_channel))
numpy_horizontal = np.hstack((image, grey_3_channel))

numpy_vertical_concat = np.concatenate((imag
for group_dir in group_dirs:e, grey_3_channel), axis=0)
numpy_horizontal_concat = np.concatenate((image, grey_3_channel), axis=1)

cv2.imshow('Main', image)
cv2.imshow('Numpy Vertical', numpy_vertical)
cv2.imshow('Numpy Horizontal', numpy_horizontal)
cv2.imshow('Numpy Vertical Concat', numpy_vertical_concat)
cv2.imshow('Numpy Horizontal Concat', numpy_horizontal_concat)

cv2.waitKey()
"""

img_width = 50
img_height = 50

page_width = 30
page_height = 20
page_total = page_width * page_height

option = cv2.IMREAD_GRAYSCALE

blank_image = [[[0., 0., 127.]] * 50] * 50 if option == cv2.IMREAD_COLOR else [[127.] * 50] * 50 

output_dir = "output"
group_dirs = sorted(glob(path.join(output_dir, "group*")))
group_cnt = len(group_dirs)

print(group_cnt)

group_idx = 0
page_num = 0
imgsets = [None] * group_cnt

while True:
    group_dir = group_dirs[group_idx]
    if imgsets[group_idx] == None:
        imgsets[group_idx] = imgutils.resize_images(imgutils.load_imageset(group_dir, option), (img_height, img_width))
    imgset = imgsets[group_idx]
    size = len(imgset)

    full_rows = size // page_width
    rows = [np.hstack(imgset[r*page_width : (r+1)*page_width]) / 255 for r in xrange(full_rows)]
    rows.append(np.hstack(imgset[full_rows*page_width:] + [blank_image] * (page_width - size % page_width)) / 255)
    page = np.vstack(rows)

    print(group_dir)
    cv2.imshow("Output", page)
    key = cv2.waitKey()

    if key == 1048603:  # ESC
        cv2.destroyAllWindows()
        break
    elif key == 1113939:
        group_idx = min(group_idx+1, group_cnt-1)
        page_num = 0
    elif key == 1113937:
        group_idx = max(group_idx-1, 0)
        page_num = 0
