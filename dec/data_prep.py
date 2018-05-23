import numpy as np
import cv2

def img_vector(filepath):
    img = cv2.imread(filepath)
    return img.reshape(reduce(lambda x,y: x*y, img.shape))
