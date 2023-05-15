import cv2
import sys
from matplotlib import pyplot as plt


def display(image, title=None, n=1):
    ax = plt.subplot(3, 3, n)

    if title is not None:
        ax.set_title(title)
    ax.imshow(image)


img = cv2.imread("bus.jpg")
if img is None:
    sys.exit("cannot find file")
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

thr, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

cv2.imwrite("bus.gray.jpeg", gray)
display(binary, f"BINARY OTSU with {thr}", 1)
display(cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB), "GRAY_IMG", 2)
display(img, "original img", 3)

red = img.copy()
red[:, :, 1] = 0
red[:, :, 2] = 0

blue = img.copy()
blue[:, :, 0] = 0
blue[:, :, 1] = 0

green = img.copy()
green[:, :, 0] = 0
green[:, :, 2] = 0

display(red, "Red", 4)
display(blue, "Blue", 5)
display(green, "Green", 6)


ax = plt.subplot(3, 3, 7)
ax.plot(cv2.calcHist([gray], [0], None, [256], [0, 256]), color="black")
colors = ["red", "green", "blue"]
bgr = cv2.split(img)
for idx in range(3):
    ax.plot(cv2.calcHist([bgr[idx]], [0], None, [256], [0, 256]), color=colors[idx])
ax.legend(["gray"] + colors)
plt.show()

exit()
