import cv2
import numpy as np
from imshow import imshow

img = cv2.imread("box.png")
# Initalize ORB detector
orb: cv2.ORB = cv2.ORB_create()
# find keypoints
kp = orb.detect(img, None)
# compute descriptors
ko, des = orb.compute(img, kp)
# draw only keypoints location (ignore size, orientation)
img2 = cv2.drawKeypoints(img, kp, None, color=(0, 255, 0), flags=0)
imshow(img2)


img = cv2.imread("house.jpg")
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
orb = cv2.ORB_create()
kp, desc = orb.detectAndCompute(img, None)
img_draw = cv2.drawKeypoints(
    img, kp, None, flags=cv2.DrawMatchesFlags_DRAW_RICH_KEYPOINTS
)
img_draw2 = cv2.drawKeypoints(img, kp, None, color=(0, 255, 0), flags=0)
imshow(img_draw)
imshow(img_draw2)

# Feature Matching
im1 = cv2.imread("taekwonv1.jpg")
im2 = cv2.imread("figures.jpg")
gray1 = cv2.cvtColor(im1, cv2.COLOR_BGR2GRAY)
gray2 = cv2.cvtColor(im2, cv2.COLOR_BGR2GRAY)

kp1, desc1 = orb.detectAndCompute(gray1, None)
kp2, desc2 = orb.detectAndCompute(gray2, None)

matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
matches = matcher.match(desc1, desc2)
sorted_matches = sorted(matches, key=lambda x: x.distance)

min_dist, max_dist = sorted_matches[0].distance, sorted_matches[-1].distance
ratio = 0.2
thr = (max_dist - min_dist) * ratio + min_dist
mathers_thr = [m for m in sorted_matches if m.distance < thr]

res = cv2.drawMatches(
    im1,
    kp1,
    im2,
    kp2,
    mathers_thr,
    None,
    flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS,
)
imshow(res)
matcher = cv2.BFMatcher(cv2.NORM_HAMMING2)

matches = matcher.knnMatch(desc1, desc2, 2)
ratio = 0.8
matches_thr = [fst for fst, sec in matches if fst.distance < sec.distance * ratio]
print(f"matches : {mathers_thr}, {len(matches)}")

# 좋은 매칭점의 좌표 구하기
src_pts = np.float32([kp1[m.queryIdx].pt for m in mathers_thr])
desc_pts = np.float32([kp2[m.queryIdx].pt for m in mathers_thr])
# 원근 변환 행렬
mtrx, mask = cv2.findHomography(src_pts, desc_pts)
# generate axis points by origin images
h, w, c = im1.shape
pts = np.float32(
    [[[0, 0]], [[0, h - 1]], [[w - 1, h - 1]], [[w - 1, 0]]],
)
# 원본 영상 좌표를 원근 변환
dst = cv2.perspectiveTransform(pts, mtrx)
# 변환 좌표 영역을 대상 영역에 그리기
im2 = cv2.polylines(im2, [np.int32(dst)], True, 255, 3, cv2.LINE_AA)

res = cv2.drawMatches(
    im1,
    kp1,
    im2,
    kp2,
    matches_thr,
    None,
    flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS,
)

res = cv2.rectangle(
    res,
    (int(dst[0][0][0]), int(dst[0][0][1])),
    (int(dst[3][0][0]), int(dst[3][0][1])),
    color=(0, 255, 0),
)
imshow(res)
