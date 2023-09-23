#
# RKNN implementation of https://github.com/sravansenthiln1/armnn_tflite/tree/main/mobilenet_v1
#

import numpy as np
from PIL import Image
import time

from rknnlite.api import RKNNLite

# Set path to the RKNN model
#
# Model path:
MODEL_PATH = "./mobilenet_v1.rknn"

# Set path to the input image (for this example)
#
# Image path:
IMAGE_PATH = "./sample.png"

rknn_lite = RKNNLite()

print('--> Load RKNN model')
ret = rknn_lite.load_rknn('mobilenet_v1.rknn')
if ret != 0:
    print('Load RKNN model failed')
    exit(ret)
print('done')

img = Image.open(IMAGE_PATH).resize((224, 224))
img = np.expand_dims(img, 0)

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

print(np.argmax(output))

