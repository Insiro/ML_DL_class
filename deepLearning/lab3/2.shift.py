import cv2
from imshow import imshow


img = cv2.imread("house.jpg")
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# shift extractor
sift = cv2.SIFT_create()
# key point extraction and calc description
keypoints, descriptor = sift.detectAndCompute(gray, None)
print(f"keypoint : {len(keypoints)}\tdescriptor : {descriptor.shape}")
print(descriptor)

# draw key points
img_draw = cv2.drawKeypoints(
    img, keypoints, None, flags=cv2.DrawMatchesFlags_DRAW_RICH_KEYPOINTS
)

imshow(img_draw)


im1 = cv2.imread("taekwonv1.jpg")
im2 = cv2.imread("figures.jpg")
gray1 = cv2.cvtColor(im1, cv2.COLOR_BGR2GRAY)
gray2 = cv2.cvtColor(im2, cv2.COLOR_BGR2GRAY)
# SIFG descriptor generator create
detector = cv2.SIFT_create()
kp1, desc1 = detector.detectAndCompute(gray1, None)
kp2, desc2 = detector.detectAndCompute(gray2, None)

# Generate BF Matcher, check L1 distance BrouthForce
matcher = cv2.BFMatcher(cv2.NORM_L1, crossCheck=True)
# chack matching
matches = matcher.match(desc1, desc2)
res = cv2.drawMatches(
    im1, kp1, im2, kp2, matches, None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS
)
imshow(res)
