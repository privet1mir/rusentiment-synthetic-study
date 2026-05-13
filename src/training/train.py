import logging
import os
import mlflow
import torch
from tqdm.auto import tqdm
from transformers import AutoModelForSequenceClassification
from transformers import get_linear_schedule_with_warmup
from torch.utils.data import DataLoader
from dataset import SyntheticSentimentDataset
from transformers import AutoTokenizer
import pandas as pd
from config import ExperimentConfig
from const import PROJECT_ROOT, LABELS
from metrics import evaluate_metrics
from visualize import plot_training_curves, plot_confusion_matrix, plot_per_class_metric
from test import test_evaluation
from utils import save_metrics_csv


def setup_logging(experiment_name):

    log_dir = os.path.join("logs", experiment_name)
    os.makedirs(log_dir, exist_ok=True)

    log_path = os.path.join(log_dir, "train.log")

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(log_path)
        ]
    )

    return log_dir



device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_mlflow(experiment_name = "Deep Learning Experiment TEST 2"):
    mlflow.set_tracking_uri("http://127.0.0.1:5050")
    mlflow.set_experiment(experiment_name)
    mlflow.config.enable_system_metrics_logging()
    mlflow.config.set_system_metrics_sampling_interval(1)


def train(cfg: ExperimentConfig) -> None:

    log_dir = setup_logging(cfg.experiment_name)
    logger = logging.getLogger(__name__)

    load_mlflow(cfg.experiment_name)

    with mlflow.start_run():
        mlflow.log_params({
            "model_name": cfg.module.model_name,
            "batch_size": cfg.trainer.per_device_train_batch_size,
            "lr": cfg.trainer.learning_rate,
            "epochs": cfg.trainer.num_train_epochs
        })
        logger.info("Training started")
        logger.info(f"Device: {device}")
        logger.info(f"Model: {cfg.module.model_name}")
        logger.info(f"Epochs: {cfg.trainer.num_train_epochs}")

        train_df = pd.read_csv(PROJECT_ROOT / "data/processed/train.csv")
        val_df = pd.read_csv(PROJECT_ROOT / "data/processed/val.csv")

        logger.info(f"Train size: {len(train_df)}")
        logger.info(f"Validation size: {len(val_df)}")

        tokenizer = AutoTokenizer.from_pretrained(cfg.module.model_name)

        train_dataset = SyntheticSentimentDataset(train_df, tokenizer)
        val_dataset = SyntheticSentimentDataset(val_df, tokenizer)

        train_loader = DataLoader(
            train_dataset,
            batch_size=cfg.trainer.per_device_train_batch_size,
            shuffle=True
        )

        val_loader = DataLoader(
            val_dataset,
            batch_size=cfg.trainer.per_device_eval_batch_size,
            shuffle=False
        )

        model = AutoModelForSequenceClassification.from_pretrained(
            cfg.module.model_name,
            num_labels=3
        )

        model.to(device)

        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=cfg.trainer.learning_rate,
            weight_decay=cfg.trainer.weight_decay
        )

        num_training_steps = len(train_loader) * cfg.trainer.num_train_epochs

        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=int(0.1 * num_training_steps),
            num_training_steps=num_training_steps
        )

        train_losses, val_losses, val_accs, val_f1s  = [], [], [], []

        precision_history, recall_history, f1_history = [], [], []

        best_f1 = -1
        patience = 3
        patience_counter = 0
        best_model_path = os.path.join(log_dir, "best_model.pt")

        for epoch in range(cfg.trainer.num_train_epochs):
            model.train()

            total_loss = 0

            progress_bar = tqdm(
                train_loader,
                desc=f"Epoch {epoch+1}/{cfg.trainer.num_train_epochs}",
                leave=True
            )

            for step, batch in enumerate(progress_bar):

                batch = {k: v.to(device) for k, v in batch.items()}

                optimizer.zero_grad()
                outputs = model(**batch)
                loss = outputs.loss
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()
                total_loss += loss.item()
                progress_bar.set_postfix({"loss": f"{loss.item():.4f}"})

                if step % 20 == 0:
                    mlflow.log_metric("step_loss", loss.item(), step=epoch * len(train_loader) + step)

            avg_loss = total_loss / len(train_loader)
            mlflow.log_metric("train_loss", avg_loss, step=epoch)

            metrics = evaluate_metrics(model, val_loader, device=device, num_classes=3)

            mlflow.log_metric("val_acc", metrics["acc"], step=epoch)
            mlflow.log_metric("val_macro_f1", metrics["macro_f1"], step=epoch)
            mlflow.log_metric("val_loss", metrics["val_loss"], step=epoch)

            logger.info(
                f"Epoch {epoch} | train_loss={avg_loss:.4f} val_acc={metrics['acc']:.4f} val_f1={metrics['macro_f1']:.4f}"
            )

            train_losses.append(avg_loss)
            val_losses.append(metrics["val_loss"])
            val_accs.append(metrics["acc"])
            val_f1s.append(metrics["macro_f1"])

            precision_history.append(metrics["precision_per_class"])
            recall_history.append(metrics["recall_per_class"])
            f1_history.append(metrics["f1_per_class"])

            current_f1 = metrics["macro_f1"]

            if current_f1 > best_f1:
                best_f1 = current_f1
                patience_counter = 0
                torch.save(model.state_dict(), best_model_path)
                logger.info(f"New best model saved | macro_f1={best_f1:.4f}")
            else:
                patience_counter += 1

            if patience_counter >= patience:
                logger.info(f"Early stopping triggered at epoch {epoch}")
                break

            
        save_metrics_csv(
            train_losses,
            val_losses,
            val_accs,
            val_f1s,
            precision_history,
            recall_history,
            f1_history,
            LABELS,
            log_dir,
            log_to_mlflow=True
        )


        fig_metrics_path = os.path.join(log_dir, "training_curves.png")
        plot_training_curves(
            train_losses,
            val_losses,
            val_accs,
            val_f1s,
            fig_metrics_path
        )
        fig_cm_path = os.path.join(log_dir, "confusion_matrix.png")
        plot_confusion_matrix(
            metrics["cm"].numpy(),
            LABELS,
            fig_cm_path
        )

        mlflow.log_artifact(fig_metrics_path)
        mlflow.log_artifact(fig_cm_path)

        f1_path = os.path.join(log_dir, "f1_per_class.png")
        precision_path = os.path.join(log_dir, "precision_per_class.png")
        recall_path = os.path.join(log_dir, "recall_per_class.png")

        plot_per_class_metric(
            f1_history,
            LABELS,
            "f1",
            f1_path
        )

        plot_per_class_metric(
            precision_history,
            LABELS,
            "precision",
            precision_path
        )

        plot_per_class_metric(
            recall_history,
            LABELS,
            "recall",
            recall_path
        )
        mlflow.log_artifact(f1_path)
        mlflow.log_artifact(precision_path)
        mlflow.log_artifact(recall_path)

        # mlflow.log_artifact(best_model_path)
        mlflow.log_artifact(os.path.join(log_dir, "train.log"))

        logger.info("Starting final test evaluation")
        test_evaluation(cfg, model_path=best_model_path, log_to_mlflow=True)


if __name__ == "__main__":
    cfg = ExperimentConfig.from_yaml(
    PROJECT_ROOT / "src/training/configs/e4_diverse_aware_1_5k.yaml"
)
    train(cfg)
