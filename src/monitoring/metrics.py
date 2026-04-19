from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix


def compute_technical_metrics(y_true, y_pred) -> dict:
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "confusion_matrix": confusion_matrix(y_true, y_pred).tolist()
    }


def compute_business_metrics(y_true, y_pred, positive_class: int = 1) -> dict:
    tp = sum((yt == positive_class and yp == positive_class) for yt, yp in zip(y_true, y_pred))
    fn = sum((yt == positive_class and yp != positive_class) for yt, yp in zip(y_true, y_pred))
    positives_flagged = sum(yp == positive_class for yp in y_pred)

    recall_positive = 0.0 if (tp + fn) == 0 else tp / (tp + fn)
    false_negative_rate = 0.0 if (tp + fn) == 0 else fn / (tp + fn)

    return {
        "priority_recall_positive_class": float(recall_positive),
        "false_negative_rate": float(false_negative_rate),
        "municipios_priorizados": int(positives_flagged),
        "true_positives": int(tp),
        "false_negatives": int(fn)
    }