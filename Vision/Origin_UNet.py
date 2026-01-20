# -----------------------------------------------------------------------------------
# 파일명       : Origin_UNet.py
# 설명         : Image Segmentation(U-Net)을 통한 소아 복부 질환 탐지       
# 작성자       : 이민하
# 작성일       : 2025-08-26
# -----------------------------------------------------------------------------------
# >> 주요 기능
# - 정상 1, 비정상 4가지의 상황을 Image Segmentation을 통해 구별
# - Computer Vision 분야의 U-Net 구조를 직접 구현 
#
# >> 성능
# Train Multi-Class Dice Score : 0.898 -> 0.908
# Test Multi-Class Dice Score : 0.883 -> 0.912
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

# 데이터 경로 설정
TRAIN_DATA_DIR = "F:/Stomach_X-ray/Pediatric_Abdominal_X-ray/Train/Source_Data"
TRAIN_LABEL_DIR = "F:/Stomach_X-ray/Pediatric_Abdominal_X-ray/Train/Labeling_Data"

VAL_DATA_DIR = "F:/Stomach_X-ray/Pediatric_Abdominal_X-ray/Validate/Source_Data"
VAL_LABEL_DIR = "F:/Stomach_X-ray/Pediatric_Abdominal_X-ray/Validate/Labeling_Data"

TEST_DATA_DIR = "F:/Stomach_X-ray/Pediatric_Abdominal_X-ray/Test/Source_Data"
TEST_LABEL_DIR = "F:/Stomach_X-ray/Pediatric_Abdominal_X-ray/Test/Labeling_Data"

# Training 데이터 준비
folder_list = []
label_file_list = []
label_list = []

for folder in os.listdir(TRAIN_LABEL_DIR):
    folder_list.append(os.path.join(TRAIN_LABEL_DIR, folder))

for dir in folder_list:
    for file_name in os.listdir(dir):
        label_file_list.append(os.path.join(dir, file_name))

for file in label_file_list:
    with open(file, "r", encoding="utf-8") as f:
        label_list.append(json.load(f))

# Validation 데이터 준비
val_folder_list = []
val_label_file_list = []
val_label_list = []

for folder in os.listdir(VAL_LABEL_DIR):
    val_folder_list.append(os.path.join(VAL_LABEL_DIR, folder))

for dir in val_folder_list:
    for file_name in os.listdir(dir):
        val_label_file_list.append(os.path.join(dir, file_name))

for file in val_label_file_list:
    with open(file, "r", encoding="utf-8") as f:
        val_label_list.append(json.load(f))

# Test 데이터 준비
test_folder_list = []
test_label_file_list = []
test_label_list = []

for folder in os.listdir(TEST_LABEL_DIR):
    test_folder_list.append(os.path.join(TEST_LABEL_DIR, folder))

for dir in test_folder_list:
    for file_name in os.listdir(dir):
        test_label_file_list.append(os.path.join(dir, file_name))

for file in test_label_file_list:
    with open(file, "r", encoding="utf-8") as f:
        test_label_list.append(json.load(f))


replace_dict = {"Labeling_Data": "Source_Data", ".json": ".png"}

train_file_list = [reduce(lambda x, y: x.replace(*y), replace_dict.items(), file) for file in label_file_list]
val_file_list = [reduce(lambda x, y: x.replace(*y), replace_dict.items(), file) for file in val_label_file_list]
test_file_list = [reduce(lambda x, y: x.replace(*y), replace_dict.items(), file) for file in test_label_file_list]

transform = A.Compose([ 
    A.ShiftScaleRotate(shift_limit=0.01, scale_limit=0, rotate_limit=2, p=0.5), 
    A.RandomBrightnessContrast(brightness_limit=0.05, contrast_limit=0.05, p=0.5), 
    A.pytorch.ToTensorV2()
    ])

BATCH_SIZE = 8

trainDS = XRayDataset(train_file_list, label_list, transform)
trainDL = DataLoader(trainDS, batch_size=BATCH_SIZE, shuffle=True)

valDS = XRayDataset(val_file_list, val_label_list, transform)
valDL = DataLoader(valDS, batch_size=BATCH_SIZE)


EPOCH = 300 
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
LR = 1e-4

num_classes = 5

model = OriginUNet(num_classes=num_classes).to(DEVICE)

loss_fn = CustomWeightedLoss(device=DEVICE)

optimizer = optim.AdamW(model.parameters(), lr=LR)

scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", patience=3)


loss, score = training(model=model, trainDL=trainDL, valDL=valDL, optimizer=optimizer, 
                       epoch=EPOCH, data_size=len(trainDS), val_data_size=len(valDS), 
                       loss_fn=loss_fn, scheduler=scheduler, device=DEVICE)