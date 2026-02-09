# -----------------------------------------------------------------------------------
# 파일명       : Quantization.py
# 설명         : 모델 양자화 파이프라인  
# 작성자       : 이민하
# 작성일       : 2026-02-10
# -----------------------------------------------------------------------------------
# >> 주요 기능
# - LLM 모델 서빙을 위한 양자화 파이프라인
# -----------------------------------------------------------------------------------


from transformers import AutoModelForCausalLM, AutoTokenizer

from llmcompressor import oneshot
from llmcompressor.modifiers.awq import AWQModifier

import os
import yaml


# Config 파일 불러오기
BASE_PATH = os.path.dirname(os.path.abspath(__file__))
CONFIG_PATH = os.path.join(BASE_PATH, "config.yaml")
 
with open(CONFIG_PATH, "r", encoding="utf-8") as file:
    config = yaml.safe_load(file)

# Parameter 설정
MODEL_NAME = config["model"]
SAVE_PATH = config["save_path"]

IGNORE = config["modifier_parameters"]["ignore"]
SCHEME = config["modifier_parameters"]["scheme"]
TARGETS = config["modifier_parameters"]["targets"]

# Quantization Modifier 생성
recipe = [
    AWQModifier(ignore=IGNORE, 
                scheme=SCHEME, 
                targets=TARGETS)
]

# Model 불러오기
model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

# Log 출력
print(f"Quantizing model: {MODEL_NAME}")
print(f"Ignore: {IGNORE}, Scheme: {SCHEME}, Targets: {TARGETS}")

# 양자화
oneshot(
    model=model,
    recipe=recipe
)

# Model, Tokenizer 저장
model.save_pretrained(
    SAVE_PATH,
    save_compressed=True
)

tokenizer.save_pretrained(SAVE_PATH)