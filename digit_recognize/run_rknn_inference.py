#
# RKNN implementation of https://github.com/sravansenthiln1/armnn_tflite/tree/main/digit_recognize
#

import numpy as np
from PIL import Image
import time

from rknnlite.api import RKNNLite

# Set path to the RKNN model
#
# Model path:
MODEL_PATH = "./digit_recognize_28.rknn"

# Set path to the input image (for this example)
#
# Image path:
IMAGE_PATH = "./digit7.png"

rknn_lite = RKNNLite()

print('--> Load RKNN model')
ret = rknn_lite.load_rknn(MODEL_PATH)
if ret != 0:
    print('Load RKNN model failed')
    exit(ret)
print('done')

img = Image.open(IMAGE_PATH)
img = img.convert("L")
img = np.expand_dims(img, 0)

print('--> Init runtime environment')
ret = rknn_lite.init_runtime()
if ret != 0:
    print('Init runtime environment failed!')
    exit(ret)
print('done')

img = np.array([img])

st = time.time()
output = rknn_lite.inference(inputs=[img])
en = time.time()

print("Inference in: ", (en - st) * 1000, "ms" )
print("predicts: ", np.argmax(output))

