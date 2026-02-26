import os 
import cv2
import torch
import torchvision
import numpy as np


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



if __name__ == "__main__":

    model = torchvision.models.resnet50(weights="DEFAULT")
    model.eval() 
    if not os.path.isfile("models/resnet50_torchvision.pt"):
        torch.save(model.state_dict(), 'models/resnet50_torchvision.pt')

    imgP = "images/binoculars.jpeg"
    img_np= preprocess(imgP)
    input_batch = torch.tensor(img_np)

    if torch.cuda.is_available():
        input_batch = input_batch.to('cuda')
        model.to('cuda')
    
    import time 
    start=time.time()
    for _ in range(20):
        with torch.no_grad():
            output = model(input_batch)
    end=time.time()
    print(f"cost time: {end-start:.2f}")
    
    print("output:\n",output[0,:20])
    
    probs = torch.nn.functional.softmax(output[0], dim=0)
    top5_prob, top5_catid = torch.topk(probs, 5)
    
    
    labels=[]
    with open("class_labels.txt","r") as f:
        labels=f.readlines() 
    f.close()
    labels=[item.strip() for item in labels]
    
    for i in range(top5_prob.size(0)):
        print(labels[top5_catid[i].item()],np.round(top5_prob[i].item(),4))
        #print(f"{labels[str(top5_catid[i].item())][1]}: {top5_prob[i].item():.3f}")
    
    
    # Export to ONNX
    if not os.path.isfile("models/resnet50_torchvision.onnx"):
        dummy_input = torch.randn(1, 3, 224, 224)
        
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
        
        print("export to onnx success!")
