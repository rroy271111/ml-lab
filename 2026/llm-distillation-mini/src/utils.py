import json
from pathlib import Path


def load_prompts(prompt_file: str):
    """
    Load prompts from a JSON file
    """

    with open(prompt_file, "r") as f:
        prompts = json.load(f)

    return prompts


def save_jsonl(data, output_file: str):
    """Save list of dictionaries to jsonl format."""

    output_path = Path(output_file)

    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_file, "a") as f:
        for item in data:
            f.write(json.dumps(item) + "\n")


def load_jsonl(file_path: str):
    """
    Load jsonl dataset.
    """

    data = []

    with open(file_path, "r") as f:
        for line in f:
            data.append(json.loads(line))
    return data
