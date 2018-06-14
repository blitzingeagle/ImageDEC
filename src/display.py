from __future__ import division
from __future__ import print_function

import imageset_utils as imgutils

import cv2
import numpy as np

import os
import os.path as path
from glob import glob

import sys

img_width = 50
img_height = 50

page_width = 30
page_height = 20
page_total = page_width * page_height

option = cv2.IMREAD_COLOR

blank_image = [[[0., 0., 127.]] * 50] * 50 if option == cv2.IMREAD_COLOR else [[127.] * 50] * 50

output_dir = "output"
if len(sys.argv) > 1:
    output_dir = sys.argv[1]
group_dirs = sorted(glob(path.join(output_dir, "group*")))
group_cnt = len(group_dirs)

print(group_cnt)

group_idx = 0
page_num = 0
imgsets = [None] * group_cnt

while True:
    group_dir = group_dirs[group_idx]
    if imgsets[group_idx] == None:
        with open(os.path.join(group_dir, "cluster.txt")) as file:
            contents = file.readlines()
            file_list = [os.path.join(group_dir, content.split()[0]) for content in contents]
            weights = [float(content.split()[1]) for content in contents]
            imgsets[group_idx] = imgutils.resize_images(imgutils.load_images(file_list, option), (img_height, img_width))
            for idx, img in enumerate(imgsets[group_idx]):
                cv2.putText(img, "%3f"%weights[idx], (0, 10), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 255, 255))
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
