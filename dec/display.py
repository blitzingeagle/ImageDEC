import cv2
import numpy as np

x = np.random.random((50, 50, 3))
y = np.random.random((50, 50, 3))

cv2.imshow("LOL", x)
cv2.imshow("LOL2", y)
cv2.waitKey()
cv2.destroyAllWindows()
