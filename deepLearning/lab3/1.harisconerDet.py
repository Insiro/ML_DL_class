import cv2
import numpy as np
from matplotlib import pyplot as plt

img = cv2.imread("./house.jpg")
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# 코너 검출
corner = cv2.cornerHarris(gray, 2, 3, 0.04)
# 변화령 결과의 105이상인 좌표 구하기
coord = np.where(corner > 0.1 * corner.max())
coord = np.stack((coord[1], coord[0]), axis=-1)

# 동그라미 그리기
for x, y in coord:
    cv2.circle(img, (x, y), 5, (0, 0, 255), 1, cv2.LINE_AA)
# 변화량 정규화
corner_norm = cv2.normalize(corner, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)

corner_img = cv2.cvtColor(corner_norm, cv2.COLOR_GRAY2RGB)
merged = np.hstack((corner_img, cv2.cvtColor(img, cv2.COLOR_BGR2RGB)))

plt.imshow(merged)
plt.show()
