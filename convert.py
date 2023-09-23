import sys, os
from rknn.api import RKNN

try:
    MODEL_NAME = sys.argv[1]
except:
    print("to run conversion, provide the input model name:\n" 
    "for example: to convert detect_model.tflite\n"
    "run the command: python3 convert.py detect_model\n")
    exit(1)



if(os.path.exists(MODEL_NAME + '.tflite')):
    pass
else:
    print(MODEL_NAME + '.tflite does not exist in this directory!')
    exit(1)

 
rknn = RKNN(verbose=True)

print('--> Config model')
rknn.config(target_platform='rk3588s', optimization_level=0)
print('done')

print('--> Loading model')
ret = rknn.load_tflite(model= './' + MODEL_NAME + '.tflite')
if ret != 0:
    print('Load model failed!')
    exit(ret)
print('done')

# quantization must be disabled.
print('--> Building model')
ret = rknn.build(do_quantization=False)
if ret != 0:
    print('Build model failed!')
    exit(ret)
print('done')
 
print('--> Export rknn model')
ret = rknn.export_rknn('./' + MODEL_NAME + '.rknn')
if ret != 0:
    print('Export rknn model failed!')
    exit(ret)
print('done')

print(MODEL_NAME + '.rknn created!')
