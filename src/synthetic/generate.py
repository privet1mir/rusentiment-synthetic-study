import os
import asyncio
import logging

from dotenv import load_dotenv
from openai import AsyncOpenAI
from tqdm.asyncio import tqdm

from typing import Any

from utils import compute_samples_per_label, parse_output, save_dataset, filter_sample, build_examples, choose_topic, load_topics, build_decoding_params
from config import ExperimentConfig
from prompts import RAW_SENTIMENT_PROMPT, FEW_SHOT_SENTIMENT_PROMPT, TAXONOMY_SENTIMENT_PROMPT, DIVERSE_SENTIMENT_PROMPT
from metrics import compute_diversity_metrics


load_dotenv()

client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s"
)

logger = logging.getLogger(__name__)

CONCURRENCY = 20

async def generate_sample(model: str, gen_cfg: Any, label: str, semaphore: asyncio.Semaphore, prompt_type="base", topic=None):

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
        
    else:
        prompt = RAW_SENTIMENT_PROMPT.format(label=label)

    decoding_params = build_decoding_params(gen_cfg)

    logger.info(f"Using decoding params: {decoding_params}")

    async with semaphore:

        response = await client.chat.completions.create(
            model=model,
            messages=[
                {"role": "user", "content": prompt}
            ],
            **decoding_params
        )

    return response.choices[0].message.content


async def generate_dataset(cfg: ExperimentConfig):
    counts = compute_samples_per_label(cfg)
    semaphore = asyncio.Semaphore(CONCURRENCY)
    tasks = []

    topics = None
    gen_cfg = cfg.generator

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
            output = await coro
            text, label = parse_output(output)
            if filter_sample(text, label):
                results.append({
                    "text": text,
                    "label": label
                })
            else:
                filtered_failed += 1

        except Exception as e:
            parse_failed += 1
            logger.warning(f"Failed generation: {repr(e)}")

    logger.info(f"Generated samples: {len(results)}")
    logger.info(f"Failed parses: {parse_failed}")
    logger.info(f"Failed filtered: {filtered_failed}")

    return results

async def main():
    cfg = ExperimentConfig.from_yaml(
        "src/synthetic/configs/e4_raw_params.yaml"
    )
    logger.info(f"Experiment: {cfg.experiment_name}")
    dataset = await generate_dataset(cfg)

    logger.info(f"Prompt type: {cfg.prompt_type}")

    texts = [x["text"] for x in dataset]
    metrics = compute_diversity_metrics(texts)

    logger.info(f"Diversity metrics: {metrics}")

    output_path = cfg.generator.dataset_path / f"synthetic_{cfg.prompt_type}_1_5k.csv"
    save_dataset(dataset, output_path)
    logger.info(f"Dataset saved → {output_path}")


if __name__ == "__main__":
    asyncio.run(main())
