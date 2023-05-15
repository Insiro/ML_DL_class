from matplotlib import pyplot as plt
import numpy as np
import cv2
import sys


def imshow(title, img):
    """
    cv2 and plt make conflit for frame generating/n
    it required to use cv2-headless/n
    for alternative cv2.imhow, implement imshow
    """
    plt.title(title)
    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    plt.show()


if img := cv2.imread("lena_gray.png") is None:
    sys.exit("file cannot read")

sobelx = cv2.Sobel(img, -1, 1, 0, 3)
sobely = cv2.Sobel(img, -1, 0, 1, 3)

abs_grad_x = cv2.convertScaleAbs(sobelx)
abs_grad_y = cv2.convertScaleAbs(sobely)
sobel = cv2.addWeighted(abs_grad_x, 0.5, abs_grad_y, 0.5, 0)

imshow("Original", img)
imshow("Sobel", sobel)
imshow("Sobel x", sobelx)
imshow("Sobel y", sobely)

canny = cv2.Canny(img, 400, 400)
imshow("Canny", canny)


laplacian = cv2.Laplacian(img, cv2.CV_8U, ksize=3)
mask1 = np.array([0, -1, 0], [-1, 4, -1], [0, -1, 0])
mask2 = np.array([1, 1, 1], [1, -8, 1], [1, 1, 1])
mask3 = np.array([-1, -1, -1], [-1, 8, -1], [-1, -1, -1])

laplacian1 = cv2.filter2D(img, -1, mask1)
laplacian2 = cv2.filter2D(img, -1, mask2)
laplacian3 = cv2.filter2D(img, -1, mask3)
laplacian4 = cv2.Laplacian(img, -1)

imshow("Laplacian 1", laplacian1)
imshow("Laplacian 2", laplacian2)
imshow("Laplacian 3", laplacian3)
imshow("Laplacian 4", laplacian4)
