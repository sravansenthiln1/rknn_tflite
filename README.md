# RKNN TFLite
TFlite implementations from https://github.com/sravansenthiln1/armnn_tflite/
adapted to run on Rockchip's RKNN NPU hardware platform.

Compatible with Edge2

## Examples
Note: make sure you have [setup the runtime requirements](https://github.com/sravansenthiln1/rknn_tflite#rknn-deployment)

#### Simple Neural Networks
| model | description |
|---|---|
|[Sine Model](./sine_model/) | Basic Neural network TFLite model |
|[Digit recognize Model](./digit_recognize/) | Digit recognization model |

#### Audio Processing
| model | description |
|---|---|
| [Audio classifier Model](./audio_classifier/) | Audio classifier model |

#### Image processing
| model | description |
|---|---|
| [Mobilenet v1 Model](./mobilenet_v1/) | Mobilenet v1 image classification model |
| [YOLOv8n Model](./yolov8n/) | YOLOv8n image detection model |
| [Auto crop Model](./auto_crop/) | Automatic document crop model |

## RKNN Conversion
You can convert TFLite models to run the NPU using the `convert.py` conversion script

**Requires:** Ubuntu 22.04/20.04/18.04 x86 Host computer.

After you have cloned this repo:

### get the necessary system packages
```shell
sudo apt-get install git python3 python3-dev python3-pip
sudo apt-get install libxslt1-dev zlib1g-dev libglib2.0 libsm6 libgl1-mesa-glx libprotobuf-dev gcc cmake
```

### Clone the conversion tools
```shell
git clone https://github.com/rockchip-linux/rknn-toolkit2
cd rknn-toolkit2
git checkout b25dadacc24b88eb7dfcaa47c9c525ecca89b319
```

Note: at this point of time, you can also create a virtual environment to store all the packages you need.
This will keep your system packages clean and not disturb their package versions.
for this you need to install [Conda](https://conda.io/projects/conda/en/stable/user-guide/install/linux.html)
```
conda create -n npu-env
conda activate npu-env
```

whenever you need to convert the models, you need to activate this env.

### Find the appropriate python version
```
python3 --version
```
and run the command accordingly
| python version | command |
|---|---|
| 3.11 | `version=cp311` |
| 3.10 | `version=cp310` |
| 3.9 | `version=cp39` |
| 3.8 | `version=cp38` |
| 3.7 | `version=cp37` |
| 3.6 | `version=cp36` |

### Install the requirements
```shell
pip3 install -r rknn-toolkit2/packages/requirements_$version-*.txt
```

### Install the appropriate toolkit wheel
```shell
pip3 install rknn-toolkit2/packages/rknn_toolkit2-*-$version-$version-linux_x86_64.whl
cd ../
```

### Try using the conversion tool
```shell
python3 convert.py
```

eg. to convert a file such as detect_model.tflite, run
```shell
python3 convert.py detect_model
```
in the same directory, a file called detect_model.rknn will have been created.

## RKNN Deployment
To run it on your board, you need to install appropriate RKNN API wheel

**Requires:** Edge2 with Ubuntu 22.04 OS.

After cloning this repo:

### Install pip
```shell
sudo apt-get install python3-pip
```

### Install necessary python packages
```shell
pip3 install numpy pillow opencv-python librosa sounddevice
```

### clone the toolkit
```shell
git clone https://github.com/rockchip-linux/rknn-toolkit2
cd rknn-toolkit2
git checkout b25dadacc24b88eb7dfcaa47c9c525ecca89b319
```

### Find the system python version
```
python3 --version
```
and run the command accordingly
| python version | command |
|---|---|
| 3.11 | `version=cp311` |
| 3.10 | `version=cp310` |
| 3.9 | `version=cp39` |
| 3.8 | `version=cp38` |
| 3.7 | `version=cp37` |
| 3.6 | `version=cp36` |

### Install the appropriate toolkit wheel
```shell
pip3 install rknn_toolkit_lite2/packages/rknn_toolkit_lite2-*-$version-$version-linux_aarch64.whl
```

### Copy the runtime library
```shell
sudo cp rknpu2/runtime/Linux/librknn_api/aarch64/librknnrt.so /usr/lib/
cd ../
```

Now try the [examples](https://github.com/sravansenthiln1/rknn_tflite#run-the-examples)

