import torch


def confusion_matrix(y_true: torch.Tensor, y_pred: torch.Tensor, num_classes: int) -> torch.Tensor:
    cm = torch.zeros((num_classes, num_classes), dtype=torch.long)
    for t, p in zip(y_true, y_pred):
        cm[t, p] += 1
    return cm


def macro_f1_from_cm(cm: torch.Tensor) -> float:
    tp = torch.diag(cm).float()
    fp = cm.sum(dim=0).float() - tp
    fn = cm.sum(dim=1).float() - tp

    denom = (2 * tp + fp + fn).clamp(min=1e-12)
    f1 = (2 * tp) / denom
    return f1.mean().item()


def accuracy_from_cm(cm: torch.Tensor) -> float:
    return (torch.diag(cm).sum().float() / cm.sum().clamp(min=1)).item()


def precision_recall_f1_micro(cm: torch.Tensor):
    tp = torch.diag(cm).sum().float()
    fp = cm.sum(dim=0).sum().float() - tp
    fn = cm.sum(dim=1).sum().float() - tp

    precision = tp / (tp + fp).clamp(min=1e-12)
    recall = tp / (tp + fn).clamp(min=1e-12)
    f1 = 2 * precision * recall / (precision + recall).clamp(min=1e-12)

    return precision.item(), recall.item(), f1.item()


def per_class_metrics(cm: torch.Tensor):

    tp = torch.diag(cm).float()
    fp = cm.sum(dim=0).float() - tp
    fn = cm.sum(dim=1).float() - tp

    precision = tp / (tp + fp).clamp(min=1e-12)
    recall = tp / (tp + fn).clamp(min=1e-12)
    f1 = 2 * precision * recall / (precision + recall).clamp(min=1e-12)

    return precision, recall, f1


@torch.no_grad()
def evaluate_metrics(model, loader, device, num_classes: int):

    model.eval()

    all_true = []
    all_pred = []

    total_loss = 0.0
    n_batches = 0

    for batch in loader:

        batch = {k: v.to(device) for k, v in batch.items()}
        labels = batch["labels"]

        outputs = model(**batch)

        total_loss += outputs.loss.item()
        n_batches += 1

        preds = outputs.logits.argmax(dim=-1)

        all_true.append(labels.cpu())
        all_pred.append(preds.cpu())

    y_true = torch.cat(all_true)
    y_pred = torch.cat(all_pred)

    cm = confusion_matrix(y_true, y_pred, num_classes=num_classes)

    precision_micro, recall_micro, f1_micro = precision_recall_f1_micro(cm)

    precision_cls, recall_cls, f1_cls = per_class_metrics(cm)

    return {
        "val_loss": total_loss / max(n_batches, 1),
        "acc": accuracy_from_cm(cm),
        "macro_f1": macro_f1_from_cm(cm),
        "precision_micro": precision_micro,
        "recall_micro": recall_micro,
        "f1_micro": f1_micro,
        "precision_per_class": precision_cls.tolist(),
        "recall_per_class": recall_cls.tolist(),
        "f1_per_class": f1_cls.tolist(),
        "cm": cm,
    }
