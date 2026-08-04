from __future__ import annotations
import numpy as np


def confusion_matrix(y_true, y_pred, n):
    out = np.zeros((n, n), dtype=np.int64)
    for a, b in zip(y_true, y_pred):
        if 0 <= int(a) < n and 0 <= int(b) < n:
            out[int(a), int(b)] += 1
    return out


def class_report(y_true, y_pred, names):
    cm = confusion_matrix(y_true, y_pred, len(names))
    rows = {}
    f1s, recalls = [], []
    for i, name in enumerate(names):
        tp = cm[i, i]
        fp = cm[:, i].sum() - tp
        fn = cm[i, :].sum() - tp
        precision = tp / max(tp + fp, 1)
        recall = tp / max(tp + fn, 1)
        f1 = 2 * precision * recall / max(precision + recall, 1e-12)
        rows[name] = {
            "precision": float(precision),
            "recall": float(recall),
            "f1": float(f1),
            "support": int(cm[i].sum()),
        }
        f1s.append(f1)
        recalls.append(recall)
    return {
        "accuracy": float(np.trace(cm) / max(cm.sum(), 1)),
        "macro_f1": float(np.mean(f1s)),
        "balanced_accuracy": float(np.mean(recalls)),
        "classes": rows,
        "confusion_matrix": cm.tolist(),
    }
