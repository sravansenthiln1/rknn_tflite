#
# RKNN implementation of https://github.com/sravansenthiln1/armnn_tflite/tree/main/sine_model
#

import numpy as np
import math
import time

from rknnlite.api import RKNNLite

# Set path to the RKNN model
#
# Model path:
MODEL_PATH = "./sine_model.rknn"

# Sample input value
input_value = math.pi/2

rknn_lite = RKNNLite()

print('--> Load RKNN model')
ret = rknn_lite.load_rknn(MODEL_PATH)
if ret != 0:
    print('Load RKNN model failed')
    exit(ret)
print('done')

print('--> Init runtime environment')
ret = rknn_lite.init_runtime()
if ret != 0:
    print('Init runtime environment failed!')
    exit(ret)
print('done')

value = np.array([input_value]).astype(np.float16)

st = time.time()
output = rknn_lite.inference(inputs=[value])
en = time.time()

print("Inference in: ", (en - st) * 1000, "ms" )

actual = math.sin(input_value)
print('Actual {}, Predicted {}'.format(actual, output[0][0][0]))
