# ----------------------------------------------------------
# Modules
# ----------------------------------------------------------

import yaml

from typing import Dict, Any

# ----------------------------------------------------------
# Internal Variables (do not call externally)
# ----------------------------------------------------------
_prompt = '''
### Instruction:
당신은 임상 지식을 갖춘 유능하고 신뢰할 수 있는 한국어 기반 의료 어시스턴트입니다.
사용자의 질문에 대해 정확하고 신중한 임상 추론을 바탕으로 진단 가능성을 제시해 주세요.
반드시 환자의 연령, 증상, 검사 결과, 통증 부위 등 모든 단서를 종합적으로 고려하여 추론 과정과 진단명을 제시해야 합니다.
의학적으로 정확한 용어를 사용하되, 필요하다면 일반인이 이해하기 쉬운 용어도 병행해 설명해 주세요.

### Question:
{question}
'''.strip()


# ----------------------------------------------------------
# External Variables (can be called from outside)
# ----------------------------------------------------------

# ----------------------------------------------------------
# Internal Classes (do not call externally)
# ----------------------------------------------------------

# ----------------------------------------------------------
# External Classes (can be called from outside)
# ----------------------------------------------------------

# ----------------------------------------------------------
# Internal Functions (do not call externally)
# ----------------------------------------------------------

# ----------------------------------------------------------
# External Functions (can be called from outside)
# ----------------------------------------------------------

def load_config(config_file: str) -> Dict[str, Any]:
    with open(config_file, "r", encoding="utf-8") as file:
        config = yaml.safe_load(file)
        
        return config
    
def build_preprocess(tokenizer):
    def preprocess(example):
        messages = [
            {
                "role": "user",
                "content": _prompt.format(question=example["question"])
            }
        ]

        return {
            "text": tokenizer.apply_chat_template(
                messages,
                tokenize=False
            )
        }
    return preprocess