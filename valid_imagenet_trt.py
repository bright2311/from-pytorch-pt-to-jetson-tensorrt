import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit
import cv2
import numpy as np
import os
from tqdm import tqdm
from typing import Tuple, List, Dict

TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)
INPUT_SHAPE = (3, 224, 224)  # C, H, W

class TRTInferencer:
    def __init__(self, engine_path: str):
        self.engine_path = engine_path
        self.engine = None
        self.context = None
        self.input_tensor_name = None
        self.output_tensor_name = None
        self.input_shape = None
        self.output_shape = None
        self.host_input = None
        self.host_output = None
        self.device_input = None
        self.device_output = None

        self._load_engine()
        self._allocate_buffers()

    def _load_engine(self):
        """加载Engine（兼容 8.x/10.x 接口）"""
        with open(self.engine_path, "rb") as f:
            runtime = trt.Runtime(TRT_LOGGER)
            self.engine = runtime.deserialize_cuda_engine(f.read())
        self.context = self.engine.create_execution_context()

        # 适配 10.x 张量接口 / 8.x binding 接口
        if hasattr(self.engine, "num_io_tensors") and self.engine.num_io_tensors > 0:
            # 10.x 张量接口（你的版本必须走这里）
            tensor_names = [self.engine.get_tensor_name(i) for i in range(self.engine.num_io_tensors)]
            for tensor_name in tensor_names:
                if self.engine.get_tensor_mode(tensor_name) == trt.TensorIOMode.INPUT:
                    self.input_tensor_name = tensor_name
                    self.input_shape = self.engine.get_tensor_shape(tensor_name)
                else:
                    self.output_tensor_name = tensor_name
                    self.output_shape = self.engine.get_tensor_shape(tensor_name)
        else:
            # 8.x binding 接口（兜底）
            for binding in self.engine:
                if self.engine.binding_is_input(binding):
                    self.input_tensor_name = binding
                    self.input_shape = self.engine.get_binding_shape(binding)
                else:
                    self.output_tensor_name = binding
                    self.output_shape = self.engine.get_binding_shape(binding)

        if not self.input_tensor_name or not self.output_tensor_name:
            raise ValueError("Engine 未找到有效输入/输出张量")

    def _allocate_buffers(self):
        """分配内存（无变化）"""
        input_volume = trt.volume(self.input_shape)
        output_volume = trt.volume(self.output_shape)

        self.host_input = np.zeros(input_volume, dtype=np.float32)
        self.host_output = np.zeros(output_volume, dtype=np.float32)

        self.device_input = cuda.mem_alloc(self.host_input.nbytes)
        self.device_output = cuda.mem_alloc(self.host_output.nbytes)

    def preprocess(self, img_path: str) -> np.ndarray:
        """预处理（无变化）"""
        img = cv2.imread(img_path)
        if img is None:
            raise ValueError(f"读取图像失败: {img_path}")

        img = cv2.resize(img, (INPUT_SHAPE[2], INPUT_SHAPE[1]))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = img.astype(np.float32) / 255.0
        img = (img - MEAN) / STD
        img = np.transpose(img, (2, 0, 1))
        img = img.ravel()

        return img

    def infer(self, img_path: str) -> Tuple[np.ndarray, np.ndarray]:
        """执行推理（终极兜底：仅用 execute_v2，适配 10.x 张量接口）"""
        # 1. 预处理
        self.host_input = self.preprocess(img_path)

        # 2. 拷贝数据到GPU（同步拷贝，避免Stream兼容问题）
        cuda.memcpy_htod(self.device_input, self.host_input)

        # 3. 核心适配：10.x 张量地址设置（必须保留）
        if hasattr(self.context, "set_tensor_address"):
            self.context.set_tensor_address(self.input_tensor_name, int(self.device_input))
            self.context.set_tensor_address(self.output_tensor_name, int(self.device_output))

        # 4. 终极执行逻辑：仅用 execute_v2，兼容所有版本
        # 关键：10.x 中地址已绑定，传空列表；8.x 传 bindings 列表
        try:
            if hasattr(self.context, "set_tensor_address"):
                # 10.x 版本：传空列表 []
                #self.context.execute_v2([])
                bindings = [int(self.device_input), int(self.device_output)]
                self.context.execute_v2(bindings)
            else:
                # 8.x 版本：传 bindings 列表
                bindings = [int(self.device_input), int(self.device_output)]
                self.context.execute_v2(bindings)
        except Exception as e:
            raise RuntimeError(f"推理执行失败: {e}")

        # 5. 拷贝结果回CPU
        cuda.memcpy_dtoh(self.host_output, self.device_output)

        # 6. 计算softmax
        output = self.host_output
        output = output - np.max(output)
        softmax = np.exp(output) / np.sum(np.exp(output))

        return output, softmax

# --------------------------- 精度验证函数（无变化） ---------------------------
def validate_imagenet(
    engine_path: str,
    val_dir: str,
    label_map_path: str,
    num_samples: int = None
) -> Dict[str, float]:

    img_paths = []
    true_labels = []

    with open(label_map_path, "r") as f:
        for line in f.readlines():
            fName,label_id = line.strip().split()
            img_paths.append(os.path.join(val_dir,fName))
            true_labels.append(int(label_id))

    if num_samples is not None and num_samples < len(img_paths):
        img_paths = img_paths[:num_samples]
        true_labels = true_labels[:num_samples]

    inferencer = TRTInferencer(engine_path)

    top1_correct = 0
    top5_correct = 0
    total = len(img_paths)

    for img_path, true_label in tqdm(zip(img_paths, true_labels), total=total, desc="验证中"):
        try:
            _, softmax = inferencer.infer(img_path)
            top5_pred = np.argsort(softmax)[-5:][::-1]
            if top5_pred[0] == true_label:
                top1_correct += 1
            if true_label in top5_pred:
                top5_correct += 1
        except Exception as e:
            print(f"处理图像失败 {img_path}: {e}")
            continue

    top1_acc = top1_correct / total
    top5_acc = top5_correct / total

    return {
        "top1_acc": round(top1_acc * 100, 2),
        "top5_acc": round(top5_acc * 100, 2),
        "total_samples": total
    }

if __name__ == "__main__":
    import sys
    if len(sys.argv) < 5:
        print(f"用法: {sys.argv[0]} <engine_type> <engine_path> <val_dir> <label_map_path> [num_samples]")
        print(f"示例: {sys.argv[0]} FP32 ./resnet50_fp32.engine ./imagenet_val/ ./synset_to_label.txt 1000")
        sys.exit(1)

    engine_type = sys.argv[1]
    engine_path = sys.argv[2]
    val_dir = sys.argv[3]
    label_map_path = sys.argv[4]
    num_samples = int(sys.argv[5]) if len(sys.argv) == 6 else None

    results = validate_imagenet(engine_path, val_dir, label_map_path, num_samples)

    print(f"\n===== {engine_type} 精度验证结果 =====")
    print(f"验证样本数: {results['total_samples']}")
    print(f"Top-1 准确率: {results['top1_acc']}%")
    print(f"Top-5 准确率: {results['top5_acc']}%")
