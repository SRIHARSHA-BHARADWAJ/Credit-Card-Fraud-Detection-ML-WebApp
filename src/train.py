import argparse
from pathlib import Path
import joblib
import numpy as np

from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score

from sklearn.ensemble import RandomForestClassifier

from src.preprocess import load_data, basic_preprocess, resample_smote


# ------------------------
# Logistic Regression
# ------------------------
def train_logreg(X_train, y_train):
    pipe = make_pipeline(
        PCA(n_components=0.95, random_state=42),
        LogisticRegression(
            max_iter=500,
            class_weight="balanced",
            solver="liblinear",
            random_state=42
        )
    )

    params = {"logisticregression__C": [0.1, 1, 10]}

    grid = GridSearchCV(pipe, params, cv=3, scoring="f1", n_jobs=-1)
    grid.fit(X_train, y_train)

    print("Best params (LogReg):", grid.best_params_)
    return grid.best_estimator_


# ------------------------
# Random Forest
# ------------------------
def train_rf(X_train, y_train):
    model = RandomForestClassifier(
        n_estimators=150,
        random_state=42,
        class_weight="balanced",
        n_jobs=-1
    )
    model.fit(X_train, y_train)
    return model


# ------------------------
# Evaluation
# ------------------------
def evaluate(model, X_test, y_test, name):
    print(f"\n--- Evaluating {name} ---")
    y_pred = model.predict(X_test)

    try:
        y_prob = model.predict_proba(X_test)[:, 1]
        print("ROC AUC:", roc_auc_score(y_test, y_prob))
    except:
        print("No probability output.")

    print("Precision:", precision_score(y_test, y_pred))
    print("Recall:", recall_score(y_test, y_pred))
    print("F1:", f1_score(y_test, y_pred))


# ------------------------
# MAIN
# ------------------------
def main(args):
    print(">>> Loading data...")
    df = load_data(args.data)
    X, y = basic_preprocess(df)
    X_res, y_res = resample_smote(X, y)

    X_train, X_test, y_train, y_test = train_test_split(
        X_res, y_res, test_size=0.2, random_state=42, stratify=y_res
    )

    out_dir = Path(args.out_dir)
    out_dir.mkdir(exist_ok=True)

    # Logistic Regression
    print("\n>>> Training Logistic Regression...")
    logreg = train_logreg(X_train, y_train)
    evaluate(logreg, X_test, y_test, "Logistic Regression")
    joblib.dump(logreg, out_dir / "logreg.pkl")
    print("Saved: logreg.pkl")

    # Random Forest
    print("\n>>> Training Random Forest...")
    rf = train_rf(X_train, y_train)
    evaluate(rf, X_test, y_test, "Random Forest")
    joblib.dump(rf, out_dir / "rf.pkl")
    print("Saved: rf.pkl")

    print("\n>>> Training Complete!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True)
    parser.add_argument("--out_dir", default="models")
    args = parser.parse_args()
    main(args)
