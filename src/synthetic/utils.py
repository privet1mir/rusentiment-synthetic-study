import random
import json
from pathlib import Path
import pandas as pd
import numpy as np
from config import ExperimentConfig
from prompts import FEW_SHOT_PROMPTS_MAPPER


def compute_samples_per_label(cfg: ExperimentConfig):
    num = cfg.generator.num_samples
    if cfg.generator.semantic_pruning:
        num = int(
            num *
            cfg.generator.semantic_pruning_oversampling
        )

    dist = cfg.generator.labels_distribution
    counts = {
        label: int(num * dist[label])
        for label in cfg.generator.labels
    }
    diff = num - sum(counts.values())
    counts[cfg.generator.labels[0]] += diff

    return counts

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
    
    if len(text) < 2:
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

def rank_by_semantic_redundancy(
    embeddings,
    k=5
):

    n = len(embeddings)

    if n <= 1:
        return (
            np.arange(n),
            np.zeros(n)
        )

    k = min(k, n - 1)

    norms = np.linalg.norm(
        embeddings,
        axis=1,
        keepdims=True
    )

    norms = np.clip(
        norms,
        1e-12,
        None
    )

    emb_norm = embeddings / norms

    sim_matrix = np.dot(
        emb_norm,
        emb_norm.T
    )

    np.fill_diagonal(
        sim_matrix,
        -1
    )

    topk = np.partition(
        sim_matrix,
        -k,
        axis=1
    )[:, -k:]

    redundancy_scores = topk.mean(
        axis=1
    )

    ranked_indices = np.argsort(
        redundancy_scores
    )

    return (
        ranked_indices,
        redundancy_scores
    )


async def get_embeddings(client, texts, model_name="text-embedding-3-small", batch_size=128):
    embeddings = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        response = await client.embeddings.create(
            model=model_name,
            input=batch
        )

        embeddings.extend([x.embedding for x in response.data])
    return np.array(embeddings, dtype=np.float32)


def compute_similarity_stats(embeddings):

    if len(embeddings) <= 1:
        return {
            "mean": 0.0,
            "std": 0.0,
            "min": 0.0,
            "max": 0.0,
        }

    norms = np.linalg.norm(
        embeddings,
        axis=1,
        keepdims=True
    )
    norms = np.clip(norms, 1e-12, None)
    emb_norm = embeddings / norms

    sim_matrix = np.dot(
        emb_norm,
        emb_norm.T
    )
    mask = ~np.eye(
        sim_matrix.shape[0],
        dtype=bool
    )
    similarities = sim_matrix[mask]

    return {
        "mean": similarities.mean(),
        "std": similarities.std(),
        "min": similarities.min(),
        "max": similarities.max(),
    }

async def prune_semantically_redundant_samples(
    client,
    results,
    cfg,
    logger,
):

    logger.info(
        "Starting semantic redundancy pruning"
    )

    dedup_results = []

    for label in cfg.generator.labels:

        label_samples = [
            x for x in results
            if x["label"] == label
        ]

        texts = [
            x["text"]
            for x in label_samples
        ]

        logger.info(
            f"Computing embeddings for "
            f"label='{label}' "
            f"({len(texts)} samples)"
        )

        embeddings = await get_embeddings(
            client=client,
            texts=texts
        )

        before_stats = compute_similarity_stats(
            embeddings
        )

        logger.info(
            f"[BEFORE] label='{label}' | "
            f"mean={before_stats['mean']:.4f} | "
            f"std={before_stats['std']:.4f} | "
            f"min={before_stats['min']:.4f} | "
            f"max={before_stats['max']:.4f}"
        )

        ranked_indices, redundancy_scores = (
            rank_by_semantic_redundancy(
                embeddings=embeddings,
                k=5
            )
        )

        logger.info(
            f"Mean redundancy score "
            f"for label='{label}': "
            f"{redundancy_scores.mean():.4f}"
        )

        target_n = int(
            cfg.generator.num_samples *
            cfg.generator.labels_distribution[label]
        )

        keep_indices = ranked_indices[:target_n]

        keep_indices = sorted(keep_indices)

        label_samples = [
            label_samples[i]
            for i in keep_indices
        ]

        dedup_embeddings = embeddings[
            keep_indices
        ]

        after_stats = compute_similarity_stats(
            dedup_embeddings
        )

        logger.info(
            f"[AFTER] label='{label}' | "
            f"mean={after_stats['mean']:.4f} | "
            f"std={after_stats['std']:.4f} | "
            f"min={after_stats['min']:.4f} | "
            f"max={after_stats['max']:.4f}"
        )

        retention_ratio = (
            len(label_samples) /
            len(texts)
        )

        logger.info(
            f"Retention ratio for "
            f"label='{label}': "
            f"{retention_ratio:.3f}"
        )

        dedup_results.extend(label_samples)

    logger.info(
        f"Final dataset size after pruning: "
        f"{len(dedup_results)}"
    )

    return dedup_results
