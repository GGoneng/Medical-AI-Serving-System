# -----------------------------------------------------------------------------------
# 파일명       : Hyperparameter_Tuning.py
# 설명         : Image Segmentation(U-Net) 모델의 Optimizer 선택    
# 작성자       : 이민하
# 작성일       : 2026-01-25
# -----------------------------------------------------------------------------------
# >> 주요 기능
# - 정상 1, 비정상 4가지의 상황을 Image Segmentation을 통해 구별
# - Computer Vision 분야의 U-Net 구조를 직접 구현 
# - 검증 Cost를 줄이기 위하여 작은 해상도의 데이터셋에서 Optimizer 비교
#
# >> 성능 (Train Score는 데이터 노이즈 추가로 인한 감소로 에상)
# - (Seed 7)
# - Adam : 
#   - Best Epoch : 36
#   - Train Dice Score : 0.9192
#   - Val Dice Score : 0.9659
#
# - RMSprop :
#   - Best Epoch : 14
#   - Train Dice Score : 0.9284
#   - Val Dice Score : 0.9689
#
# - (Seed 8)
# - Adam :     
#   - Best Epoch : 35
#   - Train Dice Score : 0.9161
#   - Val Dice Score : 0.9649
#
# - RMSprop :
#   - Best Epoch : 12    
#   - Train Dice Score : 0.9210
#   - Val Dice Score : 0.9654
#
# - (Seed 1)
# - Adam :     
#   - Best Epoch : 39
#   - Train Dice Score : 0.9207
#   - Val Dice Score : 0.9687
#
# - RMSprop :
#   - Best Epoch : 12    
#   - Train Dice Score : 0.9207
#   - Val Dice Score : 0.9624
#
# -----------------------------------------------------------------------------------


import os
import json

import torch
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler

from torch.utils.data import DataLoader

import albumentations as A

from functools import reduce

from XRaySegModules import *

# 실험 조건 고정
set_seed(1)

# 데이터 경로 설정
TRAIN_DATA_PATH = "F:/Stomach_X-ray/Pediatric_Abdominal_X-ray/Train/Source_Data"
TRAIN_LABEL_PATH = "F:/Stomach_X-ray/Pediatric_Abdominal_X-ray/Train/Labeling_Data"

VAL_DATA_PATH = "F:/Stomach_X-ray/Pediatric_Abdominal_X-ray/Validate/Source_Data"
VAL_LABEL_PATH = "F:/Stomach_X-ray/Pediatric_Abdominal_X-ray/Validate/Labeling_Data"

# Training 데이터 준비
folder_list = []
label_file_list = []
label_list = []

for folder in os.listdir(TRAIN_LABEL_PATH):
    folder_list.append(os.path.join(TRAIN_LABEL_PATH, folder))

for path in folder_list:
    for file_name in os.listdir(path):
        label_file_list.append(os.path.join(path, file_name))

for file in label_file_list:
    with open(file, "r", encoding="utf-8") as f:
        label_list.append(json.load(f))

# Validation 데이터 준비
val_folder_list = []
val_label_file_list = []
val_label_list = []

for folder in os.listdir(VAL_LABEL_PATH):
    val_folder_list.append(os.path.join(VAL_LABEL_PATH, folder))

for path in val_folder_list:
    for file_name in os.listdir(path):
        val_label_file_list.append(os.path.join(path, file_name))

for file in val_label_file_list:
    with open(file, "r", encoding="utf-8") as f:
        val_label_list.append(json.load(f))


replace_dict = {"Labeling_Data": "Source_Data", ".json": ".png"}

train_file_list = [reduce(lambda x, y: x.replace(*y), replace_dict.items(), file) for file in label_file_list]
val_file_list = [reduce(lambda x, y: x.replace(*y), replace_dict.items(), file) for file in val_label_file_list]

transform = A.Compose([
    A.Resize(224, 224), 
    A.ShiftScaleRotate(shift_limit=0.005, scale_limit=0, rotate_limit=1, p=0.5), 
    A.RandomBrightnessContrast(brightness_limit=0.03, contrast_limit=0.03, p=0.5), 
    A.pytorch.ToTensorV2()
])

val_transform = A.Compose([
    A.Resize(224, 224),
    A.pytorch.ToTensorV2()
])

BATCH_SIZE = 8

trainDS = XRayDataset(train_file_list, label_list, transform)
trainDL = DataLoader(trainDS, batch_size=BATCH_SIZE, shuffle=True)

valDS = XRayDataset(val_file_list, val_label_list, val_transform)
valDL = DataLoader(valDS, batch_size=BATCH_SIZE)

EPOCH = 300
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
LR = 1e-4

num_classes = 5

model = SegmentationUNet(num_classes=num_classes).to(DEVICE)

loss_fn = CustomWeightedLoss(device=DEVICE)

optimizer_list = [optim.Adam(model.parameters(), lr=LR),
             optim.RMSprop(model.parameters(), lr=LR)]

patience = 10
threshold = 1e-3

for optimizer in optimizer_list:
    scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", patience=3)

    loss, score = training(model=model, trainDL=trainDL, valDL=valDL, optimizer=optimizer, 
                        epoch=EPOCH, loss_fn=loss_fn, scheduler=scheduler, 
                        patience=patience, threshold=threshold, device=DEVICE)