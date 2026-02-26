# from-pytorch-pt-to-jetson-tensorrt

a simple example for resnet50 image classifier from torchvision pretrained weights to tensorrt c++ inference,including:

• pytorch python inference and torch2onnx;

• onnxruntime python inference and onnx to tensorrt engine;

• nvidia jetson c++ tensorrt inference.

Platform: Yahboom Jetson orin NX Super 8G

The hardware and software environment are listed here:
```
 Platform                                    Serial Number: [s|XX CLICK TO READ XXX]
  Machine: aarch64                           Hardware
  System: Linux                               Model: NVIDIA Jetson Orin NX Engineering Reference Developer Kit Super
  Distribution: Ubuntu 22.04 Jammy Jellyfish  699-level Part Number: ...
  Release: 5.15.148-tegra                     P-Number: ...
  Python: 3.10.12                             Module: NVIDIA Jetson Orin NX (8GB ram)
                                              SoC: tegra234
 Libraries                                    CUDA Arch BIN: 8.7
  CUDA: 12.6.85                               L4T: 36.4.3
  cuDNN: 9.6.0.74                             Jetpack: 6.2
  TensorRT: 10.7.0.23
  VPI: 3.2.4                                 Hostname: yahboom
  Vulkan: 1.3.204                            Interfaces
  OpenCV: 4.10.0 with CUDA: YES               eno1: ....
                                              docker0: .....
```

# pytorch python inference and torch2onnx

```Python
resnet50_imagenet_pytorch.py
```
Load resnet50.pt from torchvison, infer and transfer it to onnx model .


# onnxruntime python inference and onnx to tensorrt engine
```Python
resnet50_imagenet_onnx.py
```

if onnxruntime use cuda provider, jetson onnxruntime-gpu should be installed.

from onnx format model to trt engine:

```Shell
/usr/src/tensorrt/bin/trtexec --onnx=ResNet50.onnx --saveEngine=ResNet50.engine  --optShapes=input:1x3x224x224  --fp16=true
```

# nvidia jetson c++ tensorrt inference

c++ relevated files are listed below.

```
.
├── class_labels.txt
├── CMakeLists.txt
├── images
│   ├── binoculars.jpeg
│   ├── reflex_camera.jpeg
│   └── tabby_tiger_cat.jpg
├── main.cpp
└── models
```

compile, make and install:
```Shell
cd build/ && rm -rf * && cmake .. && make
```

execute the bin file:
```Shell
cp trt_infer .. && cd .. && ./trt_infer
```

Part of this prj references https://github.com/atinfinity/trt-infer-example-cpp .


