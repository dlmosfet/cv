import pyautogui
import torch
from torchvision.models.detection import fasterrcnn_resnet50_fpn
import cv2
import numpy as np
import warnings

def capture_screen():
    # 捕獲屏幕截圖
    screenshot = pyautogui.screenshot()
    return screenshot

def detect_objects(image):
    # 加載預訓練的物件檢測模型
    model = fasterrcnn_resnet50_fpn(pretrained=True, pretrained_backbone=False)

    # 將模型設置為評估模式
    model.eval()

    # 將圖像轉換為Tensor
    image_tensor = torch.tensor(image).permute(2, 0, 1).unsqueeze(0).float()

    # 預測物件
    with torch.no_grad():
        predictions = model(image_tensor)

    # 解析預測結果
    labels = predictions[0]['labels']
    scores = predictions[0]['scores']
    boxes = predictions[0]['boxes']

    # 返回識別結果
    return labels, scores, boxes

def main():
    # 忽略警告
    warnings.filterwarnings("ignore")

    # 捕獲屏幕截圖
    screenshot = capture_screen()

    # 將截圖轉換為OpenCV格式
    image = cv2.cvtColor(np.array(screenshot), cv2.COLOR_RGB2BGR)

    # 物品識別
    labels, scores, boxes = detect_objects(image)

    # 在圖像上繪製識別結果
    for label, score, box in zip(labels, scores, boxes):
        if score > 0.5:  # 設定置信度閾值
            box = box.cpu().numpy().astype(int)
            cv2.rectangle(image, (box[0], box[1]), (box[2], box[3]), (255, 0, 0), 2)
            cv2.putText(image, f"{label}: {score:.2f}", (box[0], box[1]), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

    # 顯示圖像
    cv2.imshow("Screenshot", image)
    cv2.waitKey(0)

if __name__ == "__main__":
    main()
