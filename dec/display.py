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

numpy_vertical_concat = np.concatenate((image, grey_3_channel), axis=0)
numpy_horizontal_concat = np.concatenate((image, grey_3_channel), axis=1)

cv2.imshow('Main', image)
cv2.imshow('Numpy Vertical', numpy_vertical)
cv2.imshow('Numpy Horizontal', numpy_horizontal)
cv2.imshow('Numpy Vertical Concat', numpy_vertical_concat)
cv2.imshow('Numpy Horizontal Concat', numpy_horizontal_concat)

cv2.waitKey()
"""

output_dir = "output"
group_dirs = sorted(glob(path.join(output_dir, "group*")))
group_cnt = len(group_dirs)
print(group_cnt)
