from matplotlib import pyplot as plt
import numpy as np
import cv2


def imshow(title, img):
    """
    cv2 and plt make conflit for frame generating/n
    it required to use cv2-headless/n
    for alternative cv2.imhow, implement imshow
    """
    plt.title(title)
    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    plt.show()


apple = cv2.imread("apple.png")
orange = cv2.imread("orange.png")
dst = cv2.pyrDown(apple)
dst2 = cv2.pyrUp(apple)
imshow("apple", apple)
imshow("pyrDown", dst)
imshow("pyrUp", dst2)

apple_orange = np.hstack((apple[::, :256], orange[::, 256:]))
