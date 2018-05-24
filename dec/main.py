from __future__ import division
from __future__ import print_function

import numpy as np
import cv2

import os.path as path
from glob import glob
from operator import mul

# === Imageset Loading ===
def image_paths(directory):
    return sorted(glob(path.join(directory, '*')))


def load_images(filepaths, option=cv2.IMREAD_COLOR):
    return [cv2.imread(filepath, option) for filepath in filepaths]


def load_imageset(directory, option=cv2.IMREAD_COLOR):
    return load_images(image_paths(directory), option)


# === Imageset Stats ===
def imageset_stats(imageset):
    stats = {}
    count = len(imageset)
    shape = {}

    height_sum, width_sum = 0, 0
    for img in imageset:
        height_sum += img.shape[0]
        width_sum += img.shape[1]
    shape["mean_height"] = height_sum / count
    shape["mean_width"] = width_sum / count

    stats["count"] = count
    stats["shape"] = shape

    return stats


# === Resize Images ===
def resize_images(images, shape=(128,128)):
    return [cv2.resize(img, dsize=shape, interpolation=cv2.INTER_CUBIC) for img in images]


def resize_to_mean(imageset):
    imgset_stats = imageset_stats(imageset)
    target_shape = (int(imgset_stats["shape"]["mean_width"]), int(imgset_stats["shape"]["mean_height"]))
    return resize_images(imageset, shape=target_shape)


# === Columnize ===
def columnize(dataset):
    return [elem.reshape(reduce(mul, elem.shape, 1)) for elem in dataset]


if __name__ == "__main__":
    input_dir = "images"
    imageset = resize_to_mean(load_imageset("images", cv2.IMREAD_GRAYSCALE))
    data = columnize(imageset)

    print(np.array(data).shape)
