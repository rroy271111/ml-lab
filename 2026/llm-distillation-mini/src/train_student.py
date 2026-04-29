from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments

from peft import LoraConfig, get_peft_model

from trl import SFTrainer
from utils import load_jsonl

STUDENT_MODEL = "TinyLlama/TinyLlama-1.1B-chat-v1.0"

DATASET_PATH = "../data/distillation_dataset.jsonl"

OUTPUT_DIR = "../outputs/student-distilled"


def formatting_func(example):

    text = f"""
### Instruction:
{example['prompt']}

### Response:
{example['response']}
"""

    return text
