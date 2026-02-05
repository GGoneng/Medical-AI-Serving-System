# ----------------------------------------------------------
# Modules
# ----------------------------------------------------------

import torch
import torch.nn as nn
import torch.nn.functional as F

from PIL import Image
import albumentations as A
import numpy as np
import cv2 as cv

from io import BytesIO
import base64
import threading
import pickle
import os

import redis

from Modules.TypeVariable import *

# ----------------------------------------------------------
# Internal Variables (do not call externally)
# ----------------------------------------------------------

_BASE_PATH = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_VISION_WEIGHTS_PATH = os.path.join(_BASE_PATH, "Weights", "vision_weights.pth")
_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

_gpu_lock = threading.Lock()

# ----------------------------------------------------------
# External Variables (can be called from outside)
# ----------------------------------------------------------


# ----------------------------------------------------------
# Internal Classes (do not call externally)
# ----------------------------------------------------------

class _Conv(nn.Module):
    conv: nn.Sequential

    def __init__(self, in_ch: int, out_ch: int) -> None:
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.GroupNorm(max(1, out_ch // 8), out_ch),
            nn.LeakyReLU(0.01),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.GroupNorm(max(1, out_ch // 8), out_ch),
            nn.LeakyReLU(0.01)
        )
    
    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return self.conv(input)
    
class _Expand(nn.Module):
    up: nn.Sequential
    conv: _Conv

    def __init__(self, in_ch: int, out_ch: int) -> None:
        super().__init__()
        self.up = nn.Sequential(
            nn.ConvTranspose2d(in_ch, out_ch, 2, stride=2),
            nn.LeakyReLU(0.01)
        )
        self.conv = _Conv(in_ch, out_ch)

    def forward(self, input: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = self.up(input)
        x = torch.cat((x, skip), dim=1)
        x = self.conv(x)

        return x

# ----------------------------------------------------------
# External Classes (can be called from outside)
# ----------------------------------------------------------

class SegmentationUNet(nn.Module):
    encoder1: _Conv
    encoder2: _Conv
    encoder3: _Conv
    encoder4: _Conv

    maxpool: nn.MaxPool2d
    bottleneck: _Conv

    decoder1: _Expand
    decoder2: _Expand
    decoder3: _Expand
    decoder4: _Expand

    output: nn.Conv2d
    dropout: nn.Dropout2d

    def __init__(self, num_classes: int) -> None:
        super().__init__()
        
        self.encoder1 = _Conv(1, 64)
        self.encoder2 = _Conv(64, 128)
        self.encoder3 = _Conv(128, 256)
        self.encoder4 = _Conv(256, 512)

        self.maxpool = nn.MaxPool2d(2)

        self.bottleneck = _Conv(512, 1024)

        self.decoder1 = _Expand(1024, 512)
        self.decoder2 = _Expand(512, 256)
        self.decoder3 = _Expand(256, 128)
        self.decoder4 = _Expand(128, 64)

        self.output = nn.Conv2d(64, num_classes, 1)

        self.dropout = nn.Dropout2d(0.2)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        in1 = self.encoder1(input)
        in2 = self.encoder2(self.maxpool(in1))
        in3 = self.encoder3(self.maxpool(in2))
        in4 = self.encoder4(self.maxpool(in3))

        bn = self.bottleneck(self.dropout(self.maxpool(in4)))

        out1 = self.decoder1(bn, in4)
        out2 = self.decoder2(out1, in3)
        out3 = self.decoder3(out2, in2)
        out4 = self.decoder4(out3, in1)

        final_output = self.output(out4)

        return final_output

# ----------------------------------------------------------
# Internal Functions (do not call externally)
# ----------------------------------------------------------

def _image_preprocess(img: np.ndarray, device: DeviceType="cpu") -> torch.Tensor:
    img = cv.cvtColor(img, cv.COLOR_RGB2GRAY)
    
    transform = A.Compose([A.Resize(224, 224),
                           A.pytorch.ToTensorV2()])
    
    img = np.array(img, dtype=np.float32) / 255.0

    augmented = transform(image=img)
    img_tensor = augmented["image"].unsqueeze(0).to(device)

    return img_tensor

def _model_infer(img: np.ndarray, num_classes: int, 
                 weights: str, device: DeviceType="cpu") -> torch.Tensor:
    img_tensor = _image_preprocess(img, device)

    model = SegmentationUNet(num_classes=num_classes).to(device)
    model.load_state_dict(torch.load(weights, map_location=torch.device(device), weights_only=True))

    model.eval()

    pred = model(img_tensor)
    pred = torch.argmax(F.softmax(pred, dim=1), dim=1).squeeze()

    return pred

# ----------------------------------------------------------
# External Functions (can be called from outside)
# ----------------------------------------------------------

def predict_vision(id: str, vision_memory: redis.Redis, 
                   llm_memory: redis.Redis) -> ResponseType:
    
    vision_data = pickle.loads(vision_memory.get(id))
    img = vision_data["inputs"][-1]
    img = Image.open(BytesIO(img)).convert("RGB")

    img_np = np.array(img, dtype=np.float32)
    num_classes = 5
    
    with _gpu_lock:
        pred = _model_infer(img=img_np, num_classes=num_classes, weights=_VISION_WEIGHTS_PATH, device=_DEVICE)
        pred_np = pred.cpu().numpy()

    symptom_list = ["증상 없음", "유문협착증", "기복증", "공기액체층", "변비"]
    symptom_class = max(np.unique(pred_np))

    symptom = symptom_list[symptom_class]

    llm_data = pickle.loads(llm_memory.get(id))
    llm_data["symptom"].append(symptom)

    palette = np.array([
        [0, 0, 0],        # class 0 → black
        [255, 0, 0],      # class 1 → red
        [0, 255, 0],      # class 2 → green
        [0, 0, 255],      # class 3 → blue
        [255, 255, 0],    # class 4 → yellow
    ], dtype=np.uint8)

    color_mask = palette[pred_np] 
    color_mask = color_mask.astype(np.float32)

    blend_ratio = 0.3  # 투명도 (0.0~1.0)
    pred_img = (img_np * (1 - blend_ratio) + color_mask * blend_ratio).astype(np.uint8)

    buffer = BytesIO()
    Image.fromarray(pred_img).save(buffer, format="PNG")
    base64_img = base64.b64encode(buffer.getvalue()).decode("utf-8")

    vision_data["outputs"].append(base64_img)

    vision_memory.set(id, pickle.dumps(vision_data))
    llm_memory.set(id, pickle.dumps(llm_data))

    result_path = os.path.join(_BASE_PATH, "result.png")
    Image.fromarray(pred_img).save(result_path)

    return {"id": id, "vision_result": "redis 저장 성공"}