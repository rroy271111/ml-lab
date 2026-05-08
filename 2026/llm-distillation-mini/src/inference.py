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


def generate_response(model, tokenizer, prompt):
    formatted_prompt = f"Instruction:\n{prompt}\n\n Response:\n"

    inputs = tokenizer(formatted_prompt, return_tensors="pt").to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs, max_new_tokens=100, temperature=0.7, do_sample=True, top_p=0.9
        )

        response = tokenizer.decode(outputs[0], skip_special_tokens=True)

        return response


def main():
    model, tokenizer = load_model()

    print("\nStudent model loaded successfully\n")

    while True:
        prompt = input("Enter prompt (or type 'exit'): ")

        if prompt.lower() == "exit":
            break

        response = generate_response(model, tokenizer, prompt)

        print("\nModel Response:\n")
        print(response)
        print("\n" + "=" * 60 + "\n")


if __name__ == "__main__":
    main()
