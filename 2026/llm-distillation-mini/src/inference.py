from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
import torch

BASE_MODEL = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"

ADAPTER_PATH = "./outputs/student-distilled"


def load_model():
    tokenizer = AutoTokenizer.from_pretrained(ADAPTER_PATH)

    base_model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL, torch_dtype=torch.float16, device_map="auto"
    )

    model = PeftModel.from_pretrained(base_model, ADAPTER_PATH)

    model.eval()

    return model, tokenizer
