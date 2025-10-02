from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, matthews_corrcoef
from sklearn.model_selection import train_test_split

from preprocessing import build_preprocessor

def train_baseline(X, y, cat_cols, num_cols, test_size=0.2, random_state=42):
    """Train baseline logistic regression with preprocessing."""

    preprocessor = build_preprocessor(cat_cols, num_cols)

    model = Pipeline([
        ("pre", preprocessor),
        ("clf", LogisticRegression(max_iter=500))
    ])

    # train/val split
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=random_state
    )

    model.fit(X_train, y_train)
    y_pred = model.predict(X_val)

    metrics = {
        "accuracy": accuracy_score(y_val, y_pred),
        "f1": f1_score(y_val, y_pred),
        "mcc": matthews_corrcoef(y_val, y_pred)
    }

    return model, metrics
