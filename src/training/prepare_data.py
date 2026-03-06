import os
import pandas as pd
from const import PROJECT_ROOT
from sklearn.model_selection import train_test_split


LABELS = ["negative", "neutral", "positive"]

LABEL_MAP = {
    "negative": 0,
    "neutral": 1,
    "positive": 2
}


def balanced_sample(df, label_col="label", seed=42):

    counts = df[label_col].value_counts()
    n = counts.min()

    parts = []

    for label in LABELS:
        part = df[df[label_col] == label].sample(n=n, random_state=seed)
        parts.append(part)

    df_balanced = pd.concat(parts).sample(frac=1, random_state=seed).reset_index(drop=True)

    return df_balanced


def prepare_rusentiment(sample_size=None, seed=42):

    preselected_path = PROJECT_ROOT / "data/raw/rusentiment_preselected_posts.csv"
    test_path = PROJECT_ROOT / "data/raw/rusentiment_test.csv"

    print("Loading train/val source:", preselected_path)

    df = pd.read_csv(preselected_path)

    print("Raw distribution:\n", df["label"].value_counts(), "\n")

    df = df[df["label"].isin(LABELS)].copy()

    print("Filtered distribution:\n", df["label"].value_counts(), "\n")

    df = balanced_sample(df, seed=seed)

    print("Balanced train/val distribution:\n", df["label"].value_counts(), "\n")

    if sample_size:
        df = df.sample(n=sample_size, random_state=seed).reset_index(drop=True)

        print("Sampled distribution:\n", df["label"].value_counts(), "\n")

    df["label_numeric"] = df["label"].map(LABEL_MAP)

    train_df, val_df = train_test_split(
        df,
        test_size=0.2,
        random_state=seed,
        stratify=df["label"]
    )

    print("Train size:", len(train_df))
    print("Train distribution:\n", train_df["label"].value_counts(), "\n")

    print("Val size:", len(val_df))
    print("Val distribution:\n", val_df["label"].value_counts(), "\n")

    print("Loading test set:", test_path)

    test_df = pd.read_csv(test_path)

    print("Raw test distribution:\n", test_df["label"].value_counts(), "\n")

    test_df = test_df[test_df["label"].isin(LABELS)].copy()

    test_df = balanced_sample(test_df, seed=seed)

    test_df["label_numeric"] = test_df["label"].map(LABEL_MAP)

    print("Balanced test distribution:\n", test_df["label"].value_counts(), "\n")

    os.makedirs(PROJECT_ROOT / "data/processed", exist_ok=True)

    train_df.to_csv(PROJECT_ROOT / "data/processed/train.csv", index=False)
    val_df.to_csv(PROJECT_ROOT / "data/processed/val.csv", index=False)
    test_df.to_csv(PROJECT_ROOT / "data/processed/test.csv", index=False)

    print("Saved processed datasets to:", PROJECT_ROOT / "data/processed")


if __name__ == "__main__":
    prepare_rusentiment()
