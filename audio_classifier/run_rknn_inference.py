#
# RKNN implementation of https://github.com/sravansenthiln1/armnn_tflite/tree/main/audio_classifier
#

import numpy as np
import librosa
import time

from rknnlite.api import RKNNLite

# Set path to the RKNN model
#
# Model path:
MODEL_PATH = "./audio_classifier.rknn"

# Set path to the input audio (for this example)
#
# Audio path:
AUDIO_PATH = "./sample.wav"

# Map the tag output to the appropriate string
TAGS = {
        0:'none',
        1:'hello',
        2:'khadas',
        3:'vim',
        4:'edge',
        5:'tone',
        6:'mind'
}

rknn_lite = RKNNLite()

print('--> Load RKNN model')
ret = rknn_lite.load_rknn(MODEL_PATH)
if ret != 0:
    print('Load RKNN model failed')
    exit(ret)
print('done')

scale, sr = librosa.load(AUDIO_PATH)
mel_spectrogram = librosa.feature.melspectrogram(y=scale, sr=sr, n_fft=4096, hop_length=512, n_mels=256, fmax=8000)
log_mel_spectrogram = librosa.power_to_db(mel_spectrogram)
log_mel_spectrogram = np.expand_dims(log_mel_spectrogram, axis=-1)

print('--> Init runtime environment')
ret = rknn_lite.init_runtime()
if ret != 0:
    print('Init runtime environment failed!')
    exit(ret)
print('done')

st = time.time()
output = rknn_lite.inference(inputs=[log_mel_spectrogram])
en = time.time()

print("Inference in: ", (en - st) * 1000, "ms" )

prediction = np.argmax(output[0])

print(prediction, TAGS[prediction])
