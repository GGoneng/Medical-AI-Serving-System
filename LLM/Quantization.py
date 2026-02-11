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
from datasets import load_dataset

from llmcompressor import oneshot
from llmcompressor.modifiers.awq import AWQModifier
from llmcompressor.utils import dispatch_for_generation

import os
import yaml

from QuantizationModules import *

# Config 파일 불러오기
BASE_PATH = os.path.dirname(os.path.abspath(__file__))
CONFIG_PATH = os.path.join(BASE_PATH, "config.yaml")
 
config = load_config(CONFIG_PATH)

# Parameter 설정
MODEL_NAME = config["model"]["model_name"]
SAVE_PATH = config["model"]["save_path"]

DATASET_PATH = config["dataset"]["dataset_path"]
NUM_CALIBRATION_SAMPLES = config["dataset"]["num_calibration_samples"]
MAX_SEQUENCE_LENGTH = config["dataset"]["max_sequence_length"]

IGNORE = config["modifier_parameters"]["ignore"]
SCHEME = config["modifier_parameters"]["scheme"]
TARGETS = config["modifier_parameters"]["targets"]

SEED = config["seed"]

# Quantization Modifier 생성
recipe = [
    AWQModifier(ignore=IGNORE, 
                scheme=SCHEME,
                targets=TARGETS)
]

# Model 불러오기
model = AutoModelForCausalLM.from_pretrained(MODEL_NAME,
                                             dtype="auto",
                                             device_map="auto")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

# Dataset 불러오기
data = load_dataset(DATASET_PATH, name=None, split=f"train[:{NUM_CALIBRATION_SAMPLES}]")
data = data.shuffle(seed=SEED)

# Dataset 전처리
preprocess = build_preprocess(tokenizer)
data = data.map(preprocess)

# Log 출력
print(f"Quantizing model: {MODEL_NAME}\n")
print(f"Ignore: {IGNORE}, \n\nScheme: {SCHEME}, \n\nTARGETS: {TARGETS}")

# 양자화
oneshot(
    model=model,
    dataset=data,
    recipe=recipe,
    max_seq_length=MAX_SEQUENCE_LENGTH,
    num_calibration_samples=NUM_CALIBRATION_SAMPLES
)

# Test용 예시
prompt = '''
### Instruction:
당신은 임상 지식을 갖춘 유능하고 신뢰할 수 있는 한국어 기반 의료 어시스턴트입니다.
사용자의 질문에 대해 정확하고 신중한 임상 추론을 바탕으로 진단 가능성을 제시해 주세요.
반드시 환자의 연령, 증상, 검사 결과, 통증 부위 등 모든 단서를 종합적으로 고려하여 추론 과정과 진단명을 제시해야 합니다.
의학적으로 정확한 용어를 사용하되, 필요하다면 일반인이 이해하기 쉬운 용어도 병행해 설명해 주세요.

### Question:
60세 남성이 복통과 발열을 호소하며 내원하였습니다.
혈액 검사 결과 백혈구 수치가 상승했고, 우측 하복부 압통이 확인되었습니다.
가장 가능성이 높은 진단명은 무엇인가요?
'''.strip()

messages = [
    {"role": "user", "content": prompt}
]

text = tokenizer.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True,
    enable_thinking=True
)

# Quantization Model 테스트
print("\n\n")
print("========== SAMPLE GENERATION ==============")

dispatch_for_generation(model)
model_inputs = tokenizer([text], return_tensors="pt").to(model.device)

generated_ids = model.generate(**model_inputs, max_new_tokens=4096)

generated_ids = [
    output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
]

response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
print(response)

print("==========================================\n\n")

# Model, Tokenizer 저장
model.save_pretrained(
    SAVE_PATH,
    save_compressed=True
)

tokenizer.save_pretrained(SAVE_PATH)