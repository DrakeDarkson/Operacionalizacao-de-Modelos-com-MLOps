from sklearn.model_selection import train_test_split

from src.evaluation.evaluate import evaluate


def split_data(X, y, ids, test_size=0.2, random_state=42):

    return train_test_split(
        X,
        y,
        ids,
        test_size=test_size,
        random_state=random_state,
        stratify=y
    )


def train_and_evaluate(model, X_train, X_test, y_train, y_test):

    model.fit(X_train, y_train)
    metrics = evaluate(model, X_test, y_test)
    return model, metrics