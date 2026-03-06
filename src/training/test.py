import mlflow
import logging
import os
import torch
import pandas as pd
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from torch.utils.data import DataLoader

from dataset import SyntheticSentimentDataset
from metrics import evaluate_metrics
from visualize import plot_confusion_matrix
from config import ExperimentConfig
from const import PROJECT_ROOT, LABELS
from utils import save_test_metrics_csv


logger = logging.getLogger(__name__)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def test_evaluation(cfg: ExperimentConfig, model_path: str, log_to_mlflow=True):

    logger.info("Starting test evaluation")

    tokenizer = AutoTokenizer.from_pretrained(cfg.module.model_name)

    model = AutoModelForSequenceClassification.from_pretrained(
        cfg.module.model_name,
        num_labels=3
    )

    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()

    logger.info(f"Best model loaded from {model_path}")

    test_df = pd.read_csv(PROJECT_ROOT / "data/processed/test.csv")

    logger.info(f"Test size: {len(test_df)}")

    test_dataset = SyntheticSentimentDataset(test_df, tokenizer)

    test_loader = DataLoader(
        test_dataset,
        batch_size=cfg.trainer.per_device_eval_batch_size,
        shuffle=False,
        num_workers=4
    )

    metrics = evaluate_metrics(
        model,
        test_loader,
        device=device,
        num_classes=3
    )

    log_dir = os.path.dirname(model_path)

    save_test_metrics_csv(
        metrics,
        log_dir,
        log_to_mlflow=log_to_mlflow
    )

    logger.info(
        f"TEST RESULTS | acc={metrics['acc']:.4f} "
        f"macro_f1={metrics['macro_f1']:.4f} "
        f"loss={metrics['val_loss']:.4f}"
    )

    cm_path = os.path.join(log_dir, "test_confusion_matrix.png")
    plot_confusion_matrix(
        metrics["cm"].numpy(),
        LABELS,
        cm_path
    )

    logger.info("Test confusion matrix saved")

    if log_to_mlflow:
        mlflow.log_metric("test_acc", metrics["acc"])
        mlflow.log_metric("test_macro_f1", metrics["macro_f1"])
        mlflow.log_metric("test_loss", metrics["val_loss"])

        mlflow.log_artifact(cm_path)
