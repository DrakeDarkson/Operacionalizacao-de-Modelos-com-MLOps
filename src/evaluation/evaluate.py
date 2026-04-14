from typing import Dict

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix
)


def evaluate(model, X_test, y_test) -> Dict:

    y_pred = model.predict(X_test)

    results = {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred, zero_division=0),
        "recall": recall_score(y_test, y_pred, zero_division=0),
        "f1": f1_score(y_test, y_pred, zero_division=0),
        "classification_report": classification_report(
            y_test, y_pred, zero_division=0
        ),
        "confusion_matrix": confusion_matrix(y_test, y_pred)
    }

    return results