#
# RKNN implementation of https://github.com/sravansenthiln1/armnn_tflite/tree/main/yolov8n
#

import numpy as np
import cv2
import time

from rknnlite.api import RKNNLite

# Set path to the RKNN model
#
# Model path:
MODEL_PATH = "./yolov8n_float32.rknn"

# Set path to the input image (for this example)
#
# Image path:
IMAGE_PATH = "./sample.png"

IMGSZ = (640, 640)

# Class Names for the respective IDs.
#
# Classes:
CLASSES = ("person", "bicycle", "car", "motorbike ", "aeroplane ", "bus ", "train", "truck ", "boat", "traffic light",
           "fire hydrant", "stop sign ", "parking meter", "bench", "bird", "cat", "dog ", "horse ", "sheep", "cow", "elephant",
           "bear", "zebra ", "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite",
           "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup", "fork", "knife ",
           "spoon", "bowl", "banana", "apple", "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza ", "donut", "cake", "chair", "sofa",
           "pottedplant", "bed", "diningtable", "toilet ", "tvmonitor", "laptop	", "mouse	", "remote ", "keyboard ", "cell phone", "microwave ",
           "oven ", "toaster", "sink", "refrigerator ", "book", "clock", "vase", "scissors ", "teddy bear ", "hair drier", "toothbrush ")

PALETTE = np.random.uniform(0, 255, size=(len(CLASSES), 3))

def preprocess(img):
    # Resize the image
    img = cv2.resize(img, IMGSZ)

    # Expand the dims and match the expected input shape
    img = np.expand_dims(img, 0).astype(np.float32) / 255

    return img

def postprocess(output, confidence_threshold=0.5, iou_threshold=0.5):
    # Extracting the output tensor and arranging it for further processing
    outputs = np.transpose(np.squeeze(output[0]))

    # Get the number of rows in the output tensor
    rows = outputs.shape[0]

    # Lists to store information about each detected object
    boxes = []
    scores = []
    class_ids = []

    # Extract factors for scaling back to the original image size
    x_factor, y_factor = IMGSZ

    # Iterate through each row in the output tensor
    for i in range(rows):
        # Extract scores for each class for the current row
        classes_scores = outputs[i][4:]

        # Find the maximum score and check if it's above the confidence threshold
        max_score = np.amax(classes_scores)
        if max_score >= confidence_threshold:
            # Get the class with the highest score
            class_id = np.argmax(classes_scores)

            # Extract bounding box coordinates and dimensions
            x, y, w, h = outputs[i][0], outputs[i][1], outputs[i][2], outputs[i][3]

            # Calculate the coordinates and dimensions in terms of the original image size
            left = int((x - w / 2) * x_factor)
            top = int((y - h / 2) * y_factor)
            width = int(w * x_factor)
            height = int(h * y_factor)

            # Append information to the lists
            class_ids.append(class_id)
            scores.append(max_score)
            boxes.append([left, top, width, height])

    # Apply Non-Maximum Suppression to remove overlapping bounding boxes
    indices = cv2.dnn.NMSBoxes(boxes, scores, confidence_threshold, iou_threshold)

    # Create a list to store the final detected objects
    detections = []

    # Iterate through the indices after NMS and append information to the final list
    for i in indices:
        detections.append([
            boxes[i],
            scores[i],
            class_ids[i]
        ])

    return detections

def draw_detections(img, detections):
    # Iterate through each detection
    for ((x1, y1, w, h), score, class_id) in detections:
        # Get color for the bounding box based on class_id using a predefined color palette
        color = PALETTE[class_id]

        # Draw a rectangle around the detected object
        cv2.rectangle(img, (int(x1), int(y1)), (int(x1 + w), int(y1 + h)), color, 2)

        # Create a label with class name and confidence score
        label = f'{CLASSES[class_id]}: {score:.2f}'

        # Calculate the width and height of the label text
        (label_width, label_height), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)

        # Calculate the label position
        label_x = x1
        label_y = y1 - 10 if y1 - 10 > label_height else y1 + 10

        # Draw a filled rectangle as the background for the label
        cv2.rectangle(img, (label_x, label_y - label_height), (label_x + label_width, label_y + label_height), color, cv2.FILLED)

        # Put the label text on the image
        cv2.putText(img, label, (label_x, label_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)

    return img

rknn_lite = RKNNLite()

print('--> Load RKNN model')
ret = rknn_lite.load_rknn(MODEL_PATH)
if ret != 0:
    print('Load RKNN model failed')
    exit(ret)
print('done')

img = cv2.imread(IMAGE_PATH)

print('--> Init runtime environment')
ret = rknn_lite.init_runtime()
if ret != 0:
    print('Init runtime environment failed!')
    exit(ret)
print('done')

img = preprocess(img)

st = time.time()
output = rknn_lite.inference(inputs=[img])
en = time.time()

print("Inference in: ", (en - st) * 1000, "ms" )

detections = postprocess(output, 0.3, 0.3)

img = cv2.imread(IMAGE_PATH)
img = cv2.resize(img, IMGSZ)
draw_detections(img, detections)

cv2.imwrite("output.png", img)
