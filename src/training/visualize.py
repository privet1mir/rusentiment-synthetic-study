import matplotlib
matplotlib.use("Agg")

import numpy as np
import matplotlib.pyplot as plt


def plot_training_curves(train_loss, val_loss, val_acc, val_f1, path):

    epochs = list(range(1, len(train_loss) + 1))

    plt.figure(figsize=(10,6))

    plt.plot(epochs, train_loss, label="train_loss")
    plt.plot(epochs, val_loss, label="val_loss")
    plt.plot(epochs, val_acc, label="val_acc")
    plt.plot(epochs, val_f1, label="val_macro_f1")

    plt.xlabel("epoch")
    plt.ylabel("value")
    plt.title("Training metrics")
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.savefig(path, dpi=120)
    plt.close()


def plot_confusion_matrix(cm, labels, path):

    cm = np.array(cm)
    cm_percent = cm / np.maximum(cm.sum(axis=1, keepdims=True), 1)

    fig, ax = plt.subplots(figsize=(6,6))

    im = ax.imshow(cm_percent, cmap="Blues")

    ax.set_xticks(range(len(labels)))
    ax.set_yticks(range(len(labels)))

    ax.set_xticklabels(labels)
    ax.set_yticklabels(labels)

    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title("Confusion Matrix")

    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(
                j,
                i,
                f"{cm[i,j]}\n{cm_percent[i,j]*100:.1f}%",
                ha="center",
                va="center",
                color="black"
            )

    fig.colorbar(im)

    plt.tight_layout()
    plt.savefig(path, dpi=120)
    plt.close()


def plot_per_class_metric(history, labels, metric_name, path):

    epochs = list(range(1, len(history) + 1))

    history = list(zip(*history))

    plt.figure(figsize=(10,6))

    for i, label in enumerate(labels):
        plt.plot(
            epochs,
            history[i],
            marker="o",
            label=f"{metric_name}_{label}"
        )

    plt.xlabel("epoch")
    plt.ylabel(metric_name)
    plt.title(f"{metric_name} per class")
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.savefig(path, dpi=120)
    plt.close()
