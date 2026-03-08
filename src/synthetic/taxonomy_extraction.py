import os
import json
import asyncio
import pandas as pd
from dotenv import load_dotenv
from const import PROJECT_ROOT
from openai import AsyncOpenAI
from config import ExperimentConfig


TOPIC_EXTRACTION_PROMPT = """
You are analyzing a Russian social media dataset used for sentiment classification.

Below are example posts from the dataset.

Your task:
Identify 15–20 high-level topical categories that describe the content of these posts.

Requirements:
- Categories should be short (1–3 words)
- Avoid sentiment words
- Avoid duplicates
- Use general topics suitable for social media posts
- Categories should represent WHAT the text is about, not how the author feels

Return JSON format:

{{
  "topics": [
    "topic1",
    "topic2",
    "topic3"
  ]
}}

Posts:
{posts}
"""


TOPIC_MERGE_PROMPT = """
You are refining topic categories extracted from a Russian social media dataset.

Below is a list of topic categories produced earlier.

Your task:
Merge similar or overlapping categories into a clean and compact taxonomy.

Requirements:
- Produce 10–15 final categories
- Merge similar categories (e.g. "movies", "tv shows", "music" → "entertainment")
- Keep categories short (1–3 words)
- Avoid duplicates
- Categories should represent general social media topics

Return JSON format:

{{
  "topics": [
    "topic1",
    "topic2",
    "topic3"
  ]
}}

Topics:
{topics}
"""


load_dotenv()
client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))

train_df = pd.read_csv(PROJECT_ROOT / "data/processed/train.csv")

cfg = ExperimentConfig.from_yaml(
    "src/synthetic/configs/e3_taxonomy_based.yaml"
)

model = cfg.generator.model


async def extract_topics():
    samples = train_df.sample(700)["text"].tolist()
    posts_str = "\n".join(
        f"{i+1}. {t}" for i, t in enumerate(samples)
    )
    prompt = TOPIC_EXTRACTION_PROMPT.format(posts=posts_str)

    response = await client.chat.completions.create(
        model=model,
        messages=[
            {"role": "user", "content": prompt}
        ]
    )
    parsed = json.loads(response.choices[0].message.content)
    topics = parsed["topics"]

    print("\nRaw topics:\n")

    for t in topics:
        print("-", t)

    return topics


async def merge_topics(raw_topics):
    topics_str = "\n".join(raw_topics)
    prompt = TOPIC_MERGE_PROMPT.format(topics=topics_str)
    response = await client.chat.completions.create(
        model=model,
        messages=[
            {"role": "user", "content": prompt}
        ]
    )

    parsed = json.loads(response.choices[0].message.content)
    final_topics = parsed["topics"]
    print("\nFinal taxonomy:\n")
    for t in final_topics:
        print("-", t)

    return final_topics


async def main():
    raw_topics = await extract_topics()
    final_topics = await merge_topics(raw_topics)

    df_topics = pd.DataFrame({
        "topic": final_topics
    })

    output_path = PROJECT_ROOT / "data/processed/topic_taxonomy.csv"
    df_topics.to_csv(output_path, index=False)
    print(f"\nSaved taxonomy → {output_path}")


if __name__ == "__main__":
    asyncio.run(main())
