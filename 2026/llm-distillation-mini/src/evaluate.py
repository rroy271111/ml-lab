from transformers import AutoTokenizer, AutoModelForCausalLM

import torch

MODEL_PATH = "../outputs/student-distilled"


def generate_answer(prompt):

    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH, torch_dtype=torch.float16, device_map="auto"
    )

    messages = [{"role": "user", "content": prompt}]

    text = tokenizer.apply_chat_template(
        messages, tokenizer=False, add_generation_prompt=True
    )

    inputs = tokenizer(text, return_tensors="pt").to(model.device)

    outputs = model.generate(**inputs, max_new_tokens=200, temperature=0.7)

    decoded = tokenizer.decode(outputs[0], skip_special_tokens=True)

    return decoded


def main():
    test_prompt = "Explain Docker simply"

    response = generate_answer(test_prompt)

    print("\nMODEL RESPONSE:\n")
    print(response)


if __name__ == "__main__":
    main()
