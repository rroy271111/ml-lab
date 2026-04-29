from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

from utils import load_prompts, save_jsonl

TEACHER_MODEL = "Qwen/Qwen2.5-3B-Instruct"

PROMPT_FILE = "../data/prompts.json"

OUTPUT_FILE = "../data/distillation_dataset.jsonl"


def load_teacher_model():
    tokenizer = AutoTokenizer.from_pretrained(TEACHER_MODEL)

    model = AutoModelForCausalLM.from_pretrained(
        TEACHER_MODEL, torch_dtype=torch.float16, device_map="auto"
    )

    return tokenizer, model


def generate_response(tokenizer, model, prompt):

    messages = [{"role": "user", "content": prompt}]

    text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )

    inputs = tokenizer(text, return_tensors="pt").to(model.device)

    outputs = model.generate(
        **inputs, max_new_tokens=256, temperature=0.7, do_sample=True
    )

    decoded = tokenizer.decode(outputs[0], skip_special_tokens=True)

    return decoded


def main():
    prompts = load_prompts(PROMPT_FILE)
    tokenizer, model = load_teacher_model()

    dataset = []

    for idx, prompt in enumerate(prompts):
        print(f"Generating response {idx + 1} / {len(prompts)}")

        response = generate_response(tokenizer, model, prompt)

        sample = {"prompt": prompt, "response": response}

        dataset.append(sample)
    save_jsonl(dataset, OUTPUT_FILE)

    print("Dataset generation complete")


if __name__ == "__main__":
    main()
