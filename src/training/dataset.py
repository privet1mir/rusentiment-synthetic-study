import torch
from torch.utils.data import Dataset


class SyntheticSentimentDataset(Dataset):
    def __init__(self, df, tokenizer, max_len=128):

        self.texts = df["text"].tolist()
        self.labels = df["label_numeric"].tolist()

        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):

        encoded = self.tokenizer(
            self.texts[idx],
            truncation=True,
            padding="max_length",
            max_length=self.max_len,
            return_tensors="pt"
        )

        item = {k: v.squeeze() for k, v in encoded.items()}
        item["labels"] = torch.tensor(self.labels[idx], dtype=torch.long)

        return item