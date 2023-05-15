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


# use 1/9 avg filter


if (img := cv2.imread("lena_color.png")) is None:
    sys.exit("file Cannot Find")
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


kernel = np.ones((5, 5), np.float32) / 25
dst = cv2.filter2D(img, -1, kernel)
dist_gray = cv2.filter2D(gray, -1, kernel)
gausian1 = cv2.GaussianBlur(gray, (5, 5), 1)  # SIgma = 1
gausian2 = cv2.GaussianBlur(gray, (5, 5), 3)  # SIgma = 3
gausian3 = cv2.GaussianBlur(gray, (0, 0), 7)  # auto kernel, sigma = 7

imshow("Original Image", img)
imshow("Average Image", dst)
imshow("Original Image Gray", gray)
imshow("Average Image Gray", gray)

imshow("Guaissian Image siagma = 1 with 5x5", gausian1)
imshow("Guaissian Image siagma = 3 with 5x5", gausian2)
imshow("Guaissian Image siagma = 7", gausian3)
