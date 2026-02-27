
# 🚀 ResNet50 Image Classification: PyTorch → ONNX → TensorRT on Jetson Orin NX

End-to-end deployment pipeline for TorchVision-pretrained ResNet50 on NVIDIA Jetson Orin NX (8GB):

- ✅ PyTorch Python inference & export to ONNX  
- ✅ ONNX Runtime inference & conversion to TensorRT engine (`FP16`)  
- ✅ C++ TensorRT inference on Jetson  

> **Platform**: Yahboom Jetson Orin NX Super 8G 
> **Verified Environment**: JetPack 6.2 / L4T 36.4.3 / TensorRT 10.7.0.23

---

## 🖥️ Hardware & Software Environment

| Category       | Detail                                                                 |
|----------------|------------------------------------------------------------------------|
| **Board**      | NVIDIA Jetson Orin NX Engineering Reference Developer Kit (Super, 8GB RAM) |
| **SoC**        | `tegra234` (Ampere architecture)                                      |
| **OS**         | Ubuntu 22.04.4 LTS (kernel `5.15.148-tegra`)                          |
| **CUDA**       | `12.6.85`                                                             |
| **TensorRT**   | `10.7.0.23`                                                           |
| **Python**     | `3.10.12`                                                             |

---

## 🐍 Step 1: PyTorch Inference & Export to ONNX

**Script**: `resnet50_imagenet_pytorch.py`  
Loads TorchVision ResNet50, performs inference, exports to ONNX:

```python
# PyTorch model loading and ONNX export
model = torchvision.models.resnet50(weights="DEFAULT")

torch.onnx.export(
    model.to("cpu"),
    dummy_input,
    "models/resnet50_torchvision.onnx",
    export_params=True,
    opset_version=13,
    input_names=["input"],
    output_names=["output"],
    dynamic_axes={"input": {0: "batch"}, "output": {0: "batch"}}
)

```

---

## 📦 Step 2: ONNX Runtime & TensorRT Conversion

**Script**: `resnet50_imagenet_onnx.py`  
Requires `onnxruntime-gpu` for CUDA acceleration.

**ONNX → TensorRT Engine**:
```bash
/usr/src/tensorrt/bin/trtexec \
  --onnx=ResNet50.onnx \
  --saveEngine=ResNet50.engine \
  --optShapes=input:1x3x224x224 \
  --fp16=true
```

---

## ⚙️ Step 3: C++ TensorRT Inference

**Project Structure**:
```
.
├── class_labels.txt
├── CMakeLists.txt
├── images/
├── main.cpp
└── models/
```

**Build & Run**:
```bash
mkdir -p build && cd build
cmake -DCMAKE_BUILD_TYPE=Release .. && make -j$(nproc)
./trt_infer --image ../images/tabby_tiger_cat.jpg
```

---

## 📚 References
1.  **[C++ TRT Infer Example]**: A minimal C++ example for TensorRT inference.
    - GitHub: [atinfinity/trt-infer-example-cpp](https://github.com/atinfinity/trt-infer-example-cpp)
2.  **[TensorRT ONNX Docker]**: Dockerized inference pipeline.
    - GitHub: [MrLaki5/TensorRT-onnx-dockerized-inference](https://github.com/MrLaki5/TensorRT-onnx-dockerized-inference)
3.  **[Jetson DLA Tutorial]**: NVIDIA's official tutorial for Jetson acceleration (Note: Link currently returns 404).
    - GitHub: [NVIDIA-AI-IOT/jetson_dla_tutorial](https://github.com/NVIDIA-AI-IOT/jetson_dla_tutorial)


