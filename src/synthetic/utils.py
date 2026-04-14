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

def build_decoding_params(gen_cfg):
    params = {}

    if hasattr(gen_cfg, "temperature"):
        params["temperature"] = gen_cfg.temperature

    if hasattr(gen_cfg, "top_p"):
        params["top_p"] = gen_cfg.top_p

    if hasattr(gen_cfg, "frequency_penalty"):
        params["frequency_penalty"] = gen_cfg.frequency_penalty

    if hasattr(gen_cfg, "presence_penalty"):
        params["presence_penalty"] = gen_cfg.presence_penalty

    if gen_cfg.max_completion_tokens is not None:
        params["max_completion_tokens"] = gen_cfg.max_completion_tokens

    return params
