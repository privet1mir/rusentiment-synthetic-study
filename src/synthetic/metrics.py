import random
import numpy as np
import torch
import torch.nn.functional as F
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from transformers import AutoTokenizer, AutoModel

MODEL_NAME = "cointegrated/rubert-tiny2"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModel.from_pretrained(MODEL_NAME).to(device).eval()


def distinct_n(texts, n=1):
    ngrams = []
    for text in texts:
        tokens = text.split()
        for i in range(len(tokens) - n + 1):
            ngrams.append(tuple(tokens[i:i+n]))

    total = len(ngrams)
    unique = len(set(ngrams))
    return unique / total if total > 0 else 0


def compute_distinct(texts):
    return {
        "distinct_1": distinct_n(texts, 1),
        "distinct_2": distinct_n(texts, 2)
    }


def compute_self_bleu(texts, sample_size=200):
    smoothie = SmoothingFunction().method1

    if len(texts) > sample_size:
        texts = random.sample(texts, sample_size)

    scores = []

    for i, hypothesis in enumerate(texts):
        refs = texts[:i] + texts[i+1:]
        refs_tokens = [r.split() for r in refs]

        score = sentence_bleu(
            refs_tokens,
            hypothesis.split(),
            smoothing_function=smoothie
        )

        scores.append(score)

    return float(np.mean(scores))


def mean_pool(hidden_state, mask):
    mask = mask.unsqueeze(-1).float()
    summed = torch.sum(hidden_state * mask, dim=1)
    counts = torch.clamp(mask.sum(dim=1), min=1e-9)
    return summed / counts


def compute_embeddings(texts, batch_size=32):
    all_embeddings = []

    for i in range(0, len(texts), batch_size):
        batch = texts[i:i+batch_size]

        tokens = tokenizer(
            batch,
            padding=True,
            truncation=True,
            max_length=256,
            return_tensors="pt"
        ).to(device)

        with torch.no_grad():
            outputs = model(**tokens)

        cls = outputs.last_hidden_state[:, 0, :]
        cls = F.normalize(cls, p=2, dim=1)

        all_embeddings.append(cls.cpu())

    return torch.cat(all_embeddings, dim=0)


def compute_embedding_similarity(texts, sample_size=300):
    if len(texts) > sample_size:
        texts = random.sample(texts, sample_size)

    embeddings = compute_embeddings(texts)
    sim_matrix = torch.matmul(embeddings, embeddings.T).cpu().numpy()

    mask = ~np.eye(sim_matrix.shape[0], dtype=bool)
    avg_similarity = sim_matrix[mask].mean()

    return float(avg_similarity)


def compute_diversity_metrics(texts):
    distinct = compute_distinct(texts)
    self_bleu = compute_self_bleu(texts)
    emb_sim = compute_embedding_similarity(texts)

    return {
        **distinct,
        "self_bleu": self_bleu,
        "embedding_similarity": emb_sim
    }
