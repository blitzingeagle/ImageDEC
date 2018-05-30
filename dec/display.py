import cv2
import numpy as np

Z = np.random.random((50, 50, 3))
cv2.imshow("LOL", Z)
cv2.waitKey()
cv2.destroyAllWindows()
