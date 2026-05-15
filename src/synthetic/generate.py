import random
import os
import asyncio
import logging
import pandas as pd

from dotenv import load_dotenv
from openai import AsyncOpenAI, RateLimitError
from tqdm.asyncio import tqdm

from typing import Any

from utils import compute_samples_per_label, parse_output, save_dataset, filter_sample, build_examples, choose_topic, load_topics, build_decoding_params, prune_semantically_redundant_samples, sample_generation_config
from config import ExperimentConfig
from prompts import RAW_SENTIMENT_PROMPT, FEW_SHOT_SENTIMENT_PROMPT, TAXONOMY_SENTIMENT_PROMPT, DIVERSE_SENTIMENT_PROMPT, LATENT_CONFIG_PROMPT
from metrics import compute_diversity_metrics


load_dotenv()

client = AsyncOpenAI(
    api_key=os.getenv("OPENAI_API_KEY"),
    max_retries=0
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s"
)

logger = logging.getLogger(__name__)

CONCURRENCY = 10

MAX_RETRIES = 10

async def generate_sample(model: str, gen_cfg: Any, label: str, semaphore: asyncio.Semaphore, prompt_type="base", topic=None):

    latent_cfg = None

    if prompt_type == "few_shot":
        few_shot_examples = build_examples(label)
        prompt = FEW_SHOT_SENTIMENT_PROMPT.format(label=label, examples=few_shot_examples)

    elif prompt_type == "taxonomy_based":
        prompt = TAXONOMY_SENTIMENT_PROMPT.format(
            label=label,
            topic=topic
        )

    elif prompt_type == "decoding_params":
        prompt = DIVERSE_SENTIMENT_PROMPT.format(
            label=label
        )

    elif prompt_type == "latent_config":
        latent_cfg = sample_generation_config(label)
        prompt = LATENT_CONFIG_PROMPT.format(
            label=label,

            topic=latent_cfg["topic"],
            intent=latent_cfg["intent"],
            emotion=latent_cfg["emotion"],
            style=latent_cfg["style"],
            structure=latent_cfg["structure"],
            archetype=latent_cfg["archetype"],

            capslock=latent_cfg["surface"]["capslock"],
            emoji_usage=latent_cfg["surface"]["emoji_usage"],
            typos=latent_cfg["surface"]["typos"],
            profanity=latent_cfg["surface"]["profanity"],
            punctuation=latent_cfg["surface"]["punctuation"],
        )
        
    else:
        prompt = RAW_SENTIMENT_PROMPT.format(label=label)

    decoding_params = build_decoding_params(gen_cfg)

    for attempt in range(MAX_RETRIES):
        try:
            async with semaphore:
                response = await client.chat.completions.create(
                    model=model,
                    messages=[
                        {"role": "user", "content": prompt}
                    ],
                    **decoding_params
                )

            return {
                "output": response.choices[0].message.content,
                "latent_cfg": latent_cfg
            }
                    
        except RateLimitError:
            sleep_time = min(
                (2 ** attempt) + random.uniform(0, 1.5),
                30
            )
            logger.warning(
                f"Rate limit hit. "
                f"Retry {attempt + 1}/{MAX_RETRIES}. "
                f"Sleeping {sleep_time}s"
            )
            await asyncio.sleep(sleep_time)

    raise RuntimeError(
        f"Failed generation after {MAX_RETRIES} retries"
    )

async def generate_dataset(cfg: ExperimentConfig):
    counts = compute_samples_per_label(cfg)
    semaphore = asyncio.Semaphore(CONCURRENCY)
    tasks = []

    topics = None
    gen_cfg = cfg.generator

    if cfg.prompt_type == "decoding_params":
        logger.info(
            f"Using decoding params: "
            f"{build_decoding_params(gen_cfg)}"
        )

    if cfg.prompt_type == "taxonomy_based":
        topics = load_topics(cfg.generator.topic_taxonomy_path)

    for label, n in counts.items():
        logger.info(f"Scheduling {n} samples for label='{label}'")
        for _ in range(n):

            tasks.append(
                generate_sample(
                    cfg.generator.model,
                    gen_cfg,
                    label,
                    semaphore,
                    cfg.prompt_type,
                    topic=choose_topic(topics),
                )
            )

    results = []
    filtered_failed = 0
    parse_failed = 0
    for coro in tqdm(asyncio.as_completed(tasks), total=len(tasks)):
        try:
            result = await coro
            output = result["output"]
            latent_cfg = result["latent_cfg"]
            text, label = parse_output(output)
            if filter_sample(text, label):
                sample = {
                    "text": text,
                    "label": label
                }
                if latent_cfg is not None:
                    sample.update({
                        "topic": latent_cfg["topic"],
                        "intent": latent_cfg["intent"],
                        "emotion": latent_cfg["emotion"],
                        "style": latent_cfg["style"],
                        "structure": latent_cfg["structure"],
                        "archetype": latent_cfg["archetype"],

                        "capslock": latent_cfg["surface"]["capslock"],
                        "emoji_usage": latent_cfg["surface"]["emoji_usage"],
                        "typos": latent_cfg["surface"]["typos"],
                        "profanity": latent_cfg["surface"]["profanity"],
                        "punctuation": latent_cfg["surface"]["punctuation"],

                        "prompt_type": cfg.prompt_type
                    })

                results.append(sample)
            else:
                filtered_failed += 1

        except Exception as e:
            parse_failed += 1
            logger.warning(f"Failed generation: {repr(e)}")

    logger.info(f"Generated samples: {len(results)}")
    logger.info(f"Failed parses: {parse_failed}")
    logger.info(f"Failed filtered: {filtered_failed}")

    if cfg.generator.semantic_pruning:
        results = await prune_semantically_redundant_samples(
            client=client,
            results=results,
            cfg=cfg,
            logger=logger
        )

    return results

async def main():
    cfg = ExperimentConfig.from_yaml(
        "src/synthetic/configs/e5_latent_taxonomy.yaml"
    )
    logger.info(f"Experiment: {cfg.experiment_name}")
    dataset = await generate_dataset(cfg)

    logger.info(f"Prompt type: {cfg.prompt_type}")

    texts = [x["text"] for x in dataset]
    metrics = compute_diversity_metrics(texts)

    logger.info(f"Diversity metrics: {metrics}")

    df = pd.DataFrame(dataset)
    if "topic" in df.columns:
        logger.info("\nTOPIC DISTRIBUTION:")
        logger.info(
            df["topic"].value_counts(normalize=True)
        )
        logger.info("\nSTYLE DISTRIBUTION:")
        logger.info(
            df["style"].value_counts(normalize=True)
        )
        logger.info("\nSTRUCTURE DISTRIBUTION:")
        logger.info(
            df["structure"].value_counts(normalize=True)
        )
        logger.info("\nTOPIC vs LABEL:")
        logger.info(
            pd.crosstab(
                df["topic"],
                df["label"],
                normalize="index"
            )
        )
        logger.info("\nEMOTION vs LABEL:")
        logger.info(
            pd.crosstab(
                df["emotion"],
                df["label"],
                normalize="index"
            )
        )
        logger.info("\nAVG TEXT LENGTH BY TOPIC:")
        df["word_len"] = df["text"].str.split().apply(len)
        logger.info(
            df.groupby("topic")["word_len"]
            .mean()
            .sort_values()
        )

    output_path = cfg.generator.dataset_path / f"synthetic_{cfg.prompt_type}_1_5k.csv"
    save_dataset(dataset, output_path)
    logger.info(f"Dataset saved → {output_path}")


if __name__ == "__main__":
    asyncio.run(main())
