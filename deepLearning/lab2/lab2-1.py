from matplotlib import pyplot as plt
import cv2
import sys


if (img := cv2.imread("lena_color.png")) is None:
    sys.exit("file cannot find")


def imshow(title, img):
    """
    cv2 and plt make conflit for frame generating/n
    it required to use cv2-headless/n
    for alternative cv2.imhow, implement imshow
    """
    plt.title(title)
    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    plt.show()


print(f"Image Size = {img.shape}")
h, w, c = img.shape
print(f"Pixel Intensity Value = {img[100, 70]}")


gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
roi = img[200:400, 200:350]


imshow("Color Image", img)
imshow("Gray Image", gray)
imshow("Cropped Image", roi)
