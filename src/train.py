import mlflow
import mlflow.sklearn
import pandas as pd
import yaml
import json
import joblib
import os
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, f1_score,
    confusion_matrix, precision_score, recall_score,
)

EVAL_THRESHOLD = 0.70

# Bonus 1: support remote MLflow tracking (e.g. DagsHub) via MLFLOW_TRACKING_URI env var.
# Set locally: export MLFLOW_TRACKING_URI=sqlite:///mlflow.db
# Set in CI:   add MLFLOW_TRACKING_URI secret pointing to DagsHub server.
_tracking_uri = os.environ.get("MLFLOW_TRACKING_URI", "sqlite:///mlflow.db")
mlflow.set_tracking_uri(_tracking_uri)


def _check_label_distribution(y: pd.Series) -> dict:
    """Bonus 5: warn when any class occupies < 10% of the training set."""
    total = len(y)
    distribution = {}
    for cls in [0, 1, 2]:
        ratio = int((y == cls).sum()) / total
        distribution[f"class_{cls}_ratio"] = round(ratio, 4)
        if ratio < 0.10:
            print(
                f"[CANH BAO PHAN PHOI] Lop {cls} chi chiem {ratio:.1%} "
                f"tong mau (nguong: 10%)"
            )
    return distribution


def _build_model(model_type: str, tree_params: dict):
    """Bonus 2: select classifier based on model_type."""
    if model_type == "gradient_boosting":
        return GradientBoostingClassifier(**tree_params, random_state=42)
    elif model_type == "logistic_regression":
        # LogisticRegression does not accept tree hyperparameters
        return LogisticRegression(random_state=42, max_iter=1000)
    else:
        return RandomForestClassifier(**tree_params, random_state=42)


def _write_report(
    model_type: str,
    acc: float,
    f1: float,
    cm,
    precision,
    recall,
    distribution: dict,
) -> str:
    """Bonus 3: build a text performance report and write it to outputs/report.txt."""
    lines = [
        "=== Performance Report ===",
        f"Model Type : {model_type}",
        f"Accuracy   : {acc:.4f}",
        f"F1 Score   : {f1:.4f}",
        "",
        "Confusion Matrix (rows=actual, cols=predicted):",
        "           Pred_0  Pred_1  Pred_2",
    ]
    for i, row in enumerate(cm):
        lines.append(f"  Actual_{i} : {row[0]:6d}  {row[1]:6d}  {row[2]:6d}")

    lines += [
        "",
        "Per-class Precision and Recall:",
        f"  Class 0 (thap)       : precision={precision[0]:.4f}  recall={recall[0]:.4f}",
        f"  Class 1 (trung_binh) : precision={precision[1]:.4f}  recall={recall[1]:.4f}",
        f"  Class 2 (cao)        : precision={precision[2]:.4f}  recall={recall[2]:.4f}",
        "",
        "Label Distribution in Training Set:",
    ]
    for k, v in distribution.items():
        lines.append(f"  {k} : {v:.4f}")

    report = "\n".join(lines)
    print(report)
    with open("outputs/report.txt", "w", encoding="utf-8") as f:
        f.write(report)
    return report


def train(
    params: dict,
    data_path: str = "data/train_phase1.csv",
    eval_path: str = "data/eval.csv",
) -> float:
    df_train = pd.read_csv(data_path)
    df_eval  = pd.read_csv(eval_path)

    X_train = df_train.drop(columns=["target"])
    y_train = df_train["target"]
    X_eval  = df_eval.drop(columns=["target"])
    y_eval  = df_eval["target"]

    # Bonus 5: check label distribution before training
    distribution = _check_label_distribution(y_train)

    with mlflow.start_run():
        mlflow.log_params(params)

        # Bonus 2: pick algorithm based on model_type
        params_copy = dict(params)
        model_type  = params_copy.pop("model_type", "random_forest")
        model = _build_model(model_type, params_copy)
        model.fit(X_train, y_train)

        preds = model.predict(X_eval)
        acc   = accuracy_score(y_eval, preds)
        f1    = f1_score(y_eval, preds, average="weighted")

        mlflow.log_metric("accuracy", acc)
        mlflow.log_metric("f1_score", f1)
        mlflow.sklearn.log_model(model, "model")

        print(f"Accuracy: {acc:.4f} | F1: {f1:.4f}")

        os.makedirs("outputs", exist_ok=True)

        # Bonus 5: include label distribution in metrics.json
        metrics = {"accuracy": acc, "f1_score": f1, **distribution}
        with open("outputs/metrics.json", "w") as f:
            json.dump(metrics, f, indent=2)

        # Bonus 3: compute and save detailed performance report
        cm        = confusion_matrix(y_eval, preds, labels=[0, 1, 2])
        precision = precision_score(
            y_eval, preds, average=None, labels=[0, 1, 2], zero_division=0
        )
        recall    = recall_score(
            y_eval, preds, average=None, labels=[0, 1, 2], zero_division=0
        )
        _write_report(model_type, acc, f1, cm, precision, recall, distribution)

        os.makedirs("models", exist_ok=True)
        joblib.dump(model, "models/model.pkl")

    return acc


if __name__ == "__main__":
    with open("params.yaml") as f:
        params = yaml.safe_load(f)
    train(params)
