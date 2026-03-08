import random
import json
from pathlib import Path
import pandas as pd
from config import ExperimentConfig
from prompts import FEW_SHOT_PROMPTS_MAPPER


def compute_samples_per_label(cfg: ExperimentConfig):

    num = cfg.generator.num_samples
    dist = cfg.generator.labels_distribution

    return {
        label: int(num * dist[label])
        for label in cfg.generator.labels
    }

def clean_json_output(output: str) -> str:

    output = output.strip()

    if output.startswith("```"):
        parts = output.split("```")
        if len(parts) >= 2:
            output = parts[1]

    return output

def parse_output(output: str):
    output = clean_json_output(output)
    data = json.loads(output)

    return data["text"], data["label"]

def save_dataset(dataset, path: Path):
    df = pd.DataFrame(dataset)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)

    print(f"Saved dataset → {path}")


ALLOWED_LABELS = {"positive", "neutral", "negative"}

def filter_sample(text: str, label: str):

    if not text or not label:
        return False
    text = text.strip()
    label = label.strip().lower()
    if label not in ALLOWED_LABELS:
        return False
    if len(text) < 5:
        return False
    if len(text.split()) < 3:
        return False
    if "label" in text.lower():
        return False
    return True


def build_examples(label):
    examples = FEW_SHOT_PROMPTS_MAPPER[label]
    formatted = []
    for text in examples:
        formatted.append(
            f'{{"text": "{text}", "label": "{label}"}}'
        )
    return "\n".join(formatted)

def choose_topic(topics):
    return random.choice(topics) if topics else None

def load_topics(topics_path):
    topics_df = pd.read_csv(topics_path)
    topics = topics_df["topic"].tolist()

    return topics
