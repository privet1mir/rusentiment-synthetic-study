import pandas as pd
import os
import mlflow


def save_metrics_csv(
    train_losses,
    val_losses,
    val_accs,
    val_f1s,
    precision_history,
    recall_history,
    f1_history,
    labels,
    log_dir,
    log_to_mlflow=True
):

    metrics_rows = []

    for i in range(len(train_losses)):

        row = {
            "epoch": i + 1,
            "train_loss": train_losses[i],
            "val_loss": val_losses[i],
            "val_acc": val_accs[i],
            "val_macro_f1": val_f1s[i],
        }

        for j, label in enumerate(labels):
            row[f"precision_{label}"] = precision_history[i][j]
            row[f"recall_{label}"] = recall_history[i][j]
            row[f"f1_{label}"] = f1_history[i][j]

        metrics_rows.append(row)

    metrics_df = pd.DataFrame(metrics_rows)

    metrics_path = os.path.join(log_dir, "metrics.csv")

    metrics_df.to_csv(metrics_path, index=False)

    if log_to_mlflow:
        mlflow.log_artifact(metrics_path)

    return metrics_path


def save_test_metrics_csv(
    metrics,
    log_dir,
    log_to_mlflow=True
):

    row = {
        "acc": metrics["acc"],
        "macro_f1": metrics["macro_f1"],
        "loss": metrics["val_loss"],
    }

    metrics_df = pd.DataFrame([row])

    metrics_path = os.path.join(log_dir, "test_metrics.csv")

    metrics_df.to_csv(metrics_path, index=False)

    if log_to_mlflow:
        mlflow.log_artifact(metrics_path)

    return metrics_path