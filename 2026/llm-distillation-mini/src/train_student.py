from pathlib import Path
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments

from peft import LoraConfig, get_peft_model

from trl import SFTTrainer
from utils import load_jsonl

BASE_DIR = Path(__file__).resolve().parent.parent

STUDENT_MODEL = "TinyLlama/TinyLlama-1.1B-chat-v1.0"

DATASET_PATH = BASE_DIR / "data" / "distillation_dataset.jsonl"

OUTPUT_DIR = "../outputs/student-distilled"


def formatting_func(example):

    text = f"""
### Instruction:
{example['prompt']}

### Response:
{example['response']}
"""

    return text


def main():

    data = load_jsonl(DATASET_PATH)

    formatted_data = []

    for item in data:
        formatted_data.append({"text": formatting_func(item)})

    dataset = Dataset.from_list(formatted_data)

    tokenizer = AutoTokenizer.from_pretrained(STUDENT_MODEL)

    tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(STUDENT_MODEL, device_map="auto")

    lora_config = LoraConfig(
        r=8,
        lora_alpha=16,
        target_modules=["q_proj", "v_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
    )

    model = get_peft_model(model, lora_config)

    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=4,
        learning_rate=2e-4,
        num_train_epochs=1,
        logging_steps=5,
        save_steps=25,
        fp16=True,
    )

    trainer = SFTTrainer(model=model, train_dataset=dataset, args=training_args)

    trainer.train()

    trainer.save_model(OUTPUT_DIR)

    tokenizer.save_pretrained(OUTPUT_DIR)

    print("Training complete")


if __name__ == "__main__":
    main()
