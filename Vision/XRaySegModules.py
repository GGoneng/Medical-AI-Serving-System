import torch
import torch.nn as nn
import torch.nn.functional as F

import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler

from torch.utils.data import Dataset, DataLoader

from PIL import Image, ImageDraw

import os
import numpy as np

import albumentations as A

from typing import List, Optional, Tuple, Literal

import random

import yaml

# 실험 조건 고정 함수
def set_seed(seed: int=7) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True

class XRayDataset(Dataset):
    img_path: List[str]
    json_list: List[str]
    transform: Optional[A.Compose]

    def __init__(self, img_path: List[str], json_list: List[str], transform: Optional[A.Compose]=None) -> None:
        self.img_path = img_path
        self.label = json_list
        self.transform = transform

    def __len__(self) -> int:
        return len(self.img_path)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        img_path = self.img_path[idx]
        item = self.label[idx]

        img = Image.open(img_path).convert('L')
        mask = Image.new('L', img.size, 0)
        draw = ImageDraw.Draw(mask)

        for shape in item["shapes"]:
            points = [tuple(point) for point in shape["points"]]
            class_id = shape["class"]
            draw.polygon(points, fill=class_id)
        
        img = np.array(img, dtype=np.float32) / 255.0
        mask = np.array(mask, dtype=np.uint8)


        if self.transform:
            augment = self.transform(image=img, mask=mask)
        else:
            augment = A.pytorch.ToTensorV2()(image=img, mask=mask)
        
        img = augment["image"]
        mask = augment["mask"].long()
        

        return img, mask



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


def _dice_coefficient(pred: torch.Tensor, target: torch.Tensor, smooth: int=1) -> torch.Tensor:
    num_classes = pred.shape[1]
    pred = F.softmax(pred, dim=1) 
    
    target_onehot = F.one_hot(target, num_classes=num_classes)     # [B, H, W, C]
    target_onehot = target_onehot.permute(0, 3, 1, 2).float()
    
    dice_scores = []
    
    for i in range(num_classes):
        intersection = (pred[:, i] * target_onehot[:, i]).sum()
        union = pred[:, i].sum() + target_onehot[:, i].sum()
        dice = (2 * intersection + smooth) / (union + smooth)
        dice_scores.append(dice)

    return torch.stack(dice_scores)


def _multiclass_dice_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    dice_per_class = _dice_coefficient(pred, target)
    dice_coef = dice_per_class.mean()

    return 1 - dice_coef



class CustomWeightedLoss(nn.Module):
    device: Literal["cpu", "cuda"]
    class_weights: torch.Tensor
    CELoss: nn.CrossEntropyLoss

    def __init__(self, device: Literal["cpu", "cuda"]="cpu") -> None:
        super().__init__()
        self.device = device
        self.class_weights = torch.tensor([0.1, 1.0, 1.0, 1.0, 1.0], dtype=torch.float32, device=self.device)
        self.CELoss = nn.CrossEntropyLoss(weight=self.class_weights)

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        ce_loss = self.CELoss(pred, target)

        dice_loss = _multiclass_dice_loss(pred, target)

        return ce_loss + 2 * dice_loss
    

# 모델 Validate 함수
def validating(model: SegmentationUNet, valDL: DataLoader, 
            loss_fn: CustomWeightedLoss, 
            device: Literal["cpu", "cuda"]="cpu") -> Tuple[List, List]:
    # Dropout, BatchNorm 등 가중치 규제 비활성화
    model.eval()

    loss_total = 0
    score_total = None

    with torch.no_grad():
        # Val DataLoader에 저장된 Feature, Target 텐서로 학습 진행
        for featureTS, targetTS in valDL:
            featureTS, targetTS = featureTS.to(device), targetTS.to(device)
    
            pre_val = model(featureTS)
            
            loss = loss_fn(pre_val, targetTS)

            # DICE Score 확인 (클래스 객체와 예측값의 면적 비교)
            score = _dice_coefficient(pre_val, targetTS)

            loss_total += loss.item() 
            
            if score_total is None:   
                score_total = score
            else:
                score_total += score

    avg_loss = loss_total / len(valDL)
    avg_score = score_total / len(valDL)

    return avg_loss, avg_score

# 모델 Train 함수
def training(model: SegmentationUNet, trainDL: DataLoader, valDL: DataLoader, 
             optimizer: List, epoch: int, 
             loss_fn: CustomWeightedLoss, scheduler: lr_scheduler.ReduceLROnPlateau, 
             patience: int, threshold: float,
             device: Literal["cpu", "cuda"]="cpu") -> Tuple[List, List]:
    # 가중치 파일 저장 위치 정의
    SAVE_PATH = './saved_models'
    os.makedirs(SAVE_PATH, exist_ok=True)

    # Early Stopping을 위한 변수
    BREAK_CNT_LOSS = 0

    # Loss가 더 낮은 가중치 파일을 저장하기 위하여 Loss 로그를 담을 리스트
    LOSS_HISTORY, SCORE_HISTORY = [[], []], [[], []]

    best_idx = 0

    for count in range(1, epoch  + 1):
        # GPU 환경에서 training과 testing을 반복하므로 eval 모드 -> train 모드로 전환
        # testing에서는 train 모드 -> eval 모드
        model.train()

        SAVE_WEIGHT = os.path.join(SAVE_PATH, f"best_model_weights.pth")

        loss_total = 0
        score_total = None

        # Train DataLoader에 저장된 Feature, Target 텐서로 학습 진행
        for featureTS, targetTS in trainDL:
            featureTS, targetTS = featureTS.to(device), targetTS.to(device)

            # 결과 추론
            pre_val = model(featureTS)

            # 추론값으로 Loss값 계산
            loss = loss_fn(pre_val, targetTS)

            # DICE Score 확인 (클래스 객체와 예측값의 면적 비교)
            score = _dice_coefficient(pre_val, targetTS).detach()

            loss_total += loss.item()

            if score_total is None:
                score_total = score
            else:
                score_total += score

            # 이전 gradient 초기화
            optimizer.zero_grad()

            # 역전파로 gradient 계산
            loss.backward()

            # 계산된 gradient로 가중치 업데이트
            optimizer.step()
        
        # Val Loss, Score 계산
        val_loss, val_score = validating(model, valDL, loss_fn, device)

        LOSS_HISTORY[0].append(loss_total / len(trainDL))
        SCORE_HISTORY[0].append(score_total / len(trainDL))

        LOSS_HISTORY[1].append(val_loss)
        SCORE_HISTORY[1].append(val_score)

        print(f"\n\n[{count} / {epoch}]\n - TRAIN LOSS : {LOSS_HISTORY[0][-1]:.4f}")

        print(f"- TRAIN DICE SCORE : {SCORE_HISTORY[0][-1].mean():.4f}")
        for idx, score in enumerate(SCORE_HISTORY[0][-1]):
            print(f"- CLASS {idx} : {score.item():.4f}")

        print(f"\n - VAL LOSS : {LOSS_HISTORY[1][-1]}")
        
        print(f"- VAL DICE SCORE : {SCORE_HISTORY[1][-1].mean():.4f}")
        for idx, score in enumerate(SCORE_HISTORY[1][-1]):
            print(f"- CLASS {idx} : {score.item():.4f}")

        # Val Score 결과로 스케줄러 업데이트
        scheduler.step(val_loss)

        # Early Stopping 구현
        if len(LOSS_HISTORY[1]) >= 2:
            if (LOSS_HISTORY[1][-2] - LOSS_HISTORY[1][-1]) < threshold: 
                BREAK_CNT_LOSS += 1
            
            else: BREAK_CNT_LOSS = 0
        
        if len(LOSS_HISTORY[1]) == 1:
            torch.save(model.state_dict(), SAVE_WEIGHT)

        else:
            if LOSS_HISTORY[1][-1] < min(LOSS_HISTORY[1][:-1]):
                torch.save(model.state_dict(), SAVE_WEIGHT)
                best_idx = count - 1

        if BREAK_CNT_LOSS > patience:
            print(f"성능 및 손실 개선이 없어서 {count} EPOCH에 학습 중단")
            break

    print(f"\n\n최종 EPOCH : {best_idx + 1}")
    print(f"최종 VAL LOSS : {LOSS_HISTORY[1][best_idx]}")
    print(f"최종 VAL SCORE : {SCORE_HISTORY[1][best_idx]}")

    return LOSS_HISTORY, SCORE_HISTORY

def testing(model: SegmentationUNet, weights: str, 
            testDL: DataLoader, device: Literal["cpu", "cuda"]="cpu"):

    model.load_state_dict(torch.load(weights, map_location=torch.device(device), weights_only=True))
    model.eval()

    score_total = None
    
    with torch.no_grad():
        for featureTS, targetTS in testDL:
            featureTS, targetTS = featureTS.to(device), targetTS.to(device)

            pre_val = model(featureTS)

            score = _dice_coefficient(pre_val, targetTS)

            if score_total is None:
                score_total = score
            else:
                score_total += score
    
    final_score = (score_total / len(testDL)).mean()

    return final_score


def load_config(config_file):
    with open(config_file, "r", encoding="utf-8") as file:
        config = yaml.safe_load(file)

    return config