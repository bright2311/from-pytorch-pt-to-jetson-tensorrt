import numpy as np
import cv2
import onnxruntime as ort

INPUT_SIZE = 224             
MEAN = [0.485, 0.456, 0.406]  
STD = [0.229, 0.224, 0.225]  

def preprocess(image_path, input_size=INPUT_SIZE):
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Failed to load image: {image_path}")

    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (input_size, input_size)).astype(float)
    img /= 255.0
    img = (img - np.array(MEAN)) / np.array(STD)
    # 转换为NCHW格式
    img = np.transpose(img, (2, 0, 1))
    # 添加batch维度
    img = np.expand_dims(img, axis=0)
    return img.astype(np.float32)

def run_inference(model_path, input_data,epochs=20):

    available_providers = ort.get_available_providers()
    print("Available Providers:", available_providers)

    #Create an InferenceSession
    providers= [ 'CUDAExecutionProvider', 'CPUExecutionProvider' ]

    for _ in range(epochs):
        sess = ort.InferenceSession(model_path, providers=providers)
        input_name = sess.get_inputs()[0].name
        output_name = sess.get_outputs()[0].name
    
        inputs = {input_name: input_data}
        outputs = sess.run([output_name], inputs)

    print(f"Input name: {input_name}, Input shape: {sess.get_inputs()[0].shape}")
    print(f"Output name: {output_name}")
    return outputs

def softmax(x):
    exp_x = np.exp(x - np.max(x))  # Stabilizing to prevent overflow
    return exp_x / np.sum(exp_x)

def get_top5(preds, labels):
    top5_idx = np.argsort(preds[0])[::-1][:5]

    results = []
    for idx in top5_idx:
        results.append((labels[idx], preds[0][idx]))

    return results


if __name__ == "__main__":

    model_file = "models/resnet50_torchvision.onnx"

    imgP = "images/tabby_tiger_cat.jpg"
    dummy_input= preprocess(imgP)
    
    import time 
    start=time.time()
    preds = run_inference(model_file, dummy_input)
    print("\nInference successful.")
    end=time.time()
    print(f"Cost time: {end-start:.2f}s")
    preds_softmax=softmax(preds[0])

    labels=[]
    with open("class_labels.txt","r") as f:
        labels=f.readlines()
    f.close()
    labels=[item.strip() for item in labels]

    print("\nTop 5 Predictions:")
    top5 = get_top5(preds_softmax, labels)
    for label, prob in top5:
        print(f"{label}: {prob:.4f}")    

