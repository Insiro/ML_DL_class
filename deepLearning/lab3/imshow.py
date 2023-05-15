import cv2
import matplotlib.pyplot as plt


def imshow(img, title=""):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    plt.title(title)
    plt.imshow(img)
    plt.show()
