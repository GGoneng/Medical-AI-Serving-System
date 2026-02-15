from transformers import AutoTokenizer, AutoModelForCausalLM
import time

# Load tokenizer and model
normal_model_name = "snuh/hari-q3-8b"

normal_model = AutoModelForCausalLM.from_pretrained(
    normal_model_name,
    load_in_4bit=True,
    dtype="auto",
    device_map="auto"
)
normal_tokenizer = AutoTokenizer.from_pretrained(normal_model_name)

awq_model_name = "./hari-q3-8b-awq"

awq_model = AutoModelForCausalLM.from_pretrained(
    awq_model_name,
    dtype="auto",
    device_map="auto"
)
awq_tokenizer = AutoTokenizer.from_pretrained(awq_model_name)

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
text = normal_tokenizer.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True,
    enable_thinking=True
)
model_inputs = normal_tokenizer([text], return_tensors="pt").to(normal_model.device)

t1 = time.time()

generated_ids = normal_model.generate(
    **model_inputs,
    max_new_tokens=4096
)
generated_ids = [
    output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
]

response = normal_tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]

t2 = time.time()

print(response)
print("Normal 4-bits hari-q3:")
print(f"Response Time : {t2 - t1:.4f}")

text = awq_tokenizer.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True,
    enable_thinking=True
)

model_inputs = awq_tokenizer([text], return_tensors="pt").to(awq_model.device)

t1 = time.time()

generated_ids = awq_model.generate(
    **model_inputs,
    max_new_tokens=4096
)
generated_ids = [
    output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
]

response = awq_tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]

t2 = time.time()

print(response)
print("AWQ 4-bits hari-q3:")
print(f"Response Time : {t2 - t1:.4f}")