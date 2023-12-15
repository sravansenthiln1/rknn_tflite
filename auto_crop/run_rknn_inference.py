#
# RKNN implementation of https://github.com/sravansenthiln1/armnn_tflite/tree/main/auto_crop
#

import numpy as np
import cv2
import time

from rknnlite.api import RKNNLite

# Set path to the RKNN model
#
# Model path:
MODEL_PATH = "./auto_crop.rknn"

# Set path to the input image (for this example)
#
# Image path:
IMAGE_PATH = "./sample.png"

def find_closest_coordinate(point, coordinates):
    distances = np.linalg.norm(coordinates - point, axis=1)
    closest_index = np.argmin(distances)
    return coordinates[closest_index][::-1]

rknn_lite = RKNNLite()

print('--> Load RKNN model')
ret = rknn_lite.load_rknn(MODEL_PATH)
if ret != 0:
    print('Load RKNN model failed')
    exit(ret)
print('done')

img = cv2.imread(IMAGE_PATH)

y_scale = img.shape[0] / 256
x_scale = img.shape[1] / 192

img = cv2.resize(img, (192, 256))
img = np.expand_dims(img, 0).astype(np.float32) / 255

print('--> Init runtime environment')
ret = rknn_lite.init_runtime()
if ret != 0:
    print('Init runtime environment failed!')
    exit(ret)
print('done')

st = time.time()
output = rknn_lite.inference(inputs=[img])
en = time.time()

print("Inference in: ", (en - st) * 1000, "ms" )

seg = output[0][0] > 0.9
out = np.argwhere(seg)[:,:2]

# calculate anchors
tl = find_closest_coordinate([0, 0], out) * (x_scale, y_scale)
tr = find_closest_coordinate([0, 192 - 1], out) * (x_scale, y_scale)
bl = find_closest_coordinate([256 - 1, 0], out) * (x_scale, y_scale)
br = find_closest_coordinate([256 - 1, 192 - 1], out) * (x_scale, y_scale)

anchor = np.array([tl, tr, br, bl], dtype=np.float32)

x1 = np.linalg.norm(br - bl)
x2 = np.linalg.norm(tr - tl)

y1 = np.linalg.norm(tr - br)
y2 = np.linalg.norm(tl - bl)

w = max(int(x1), int(x2))
h = max(int(y1), int(y2))

dst = np.array([
	[0, 0],
	[w - 1, 0],
	[w - 1, h - 1],
	[0, h - 1]], dtype = "float32")

M = cv2.getPerspectiveTransform(anchor, dst)

img = cv2.imread(IMAGE_PATH)
warp = cv2.warpPerspective(img, M, (w, h))

cv2.imwrite('output.png', warp)
