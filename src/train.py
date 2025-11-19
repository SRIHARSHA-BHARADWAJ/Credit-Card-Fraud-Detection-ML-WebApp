import argparse
from pathlib import Path
import joblib
import numpy as np

from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import classification_report, precision_score, recall_score, f1_score, roc_auc_score

from src.preprocess import load_data, basic_preprocess, resample_smote

# MODEL IMPORTS
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier


# ------------------------
# Logistic Regression
# ------------------------
def train_logreg(X_train, y_train):
    pipe = make_pipeline(
        PCA(n_components=0.95, random_state=42),
        LogisticRegression(max_iter=1000, class_weight='balanced', solver='liblinear', random_state=42)
    )
    params = {'logisticregression__C': [0.01, 0.1, 1, 10]}
    grid = GridSearchCV(pipe, params, cv=3, scoring='f1', n_jobs=-1)
    grid.fit(X_train, y_train)
    print("Best params (LogReg):", grid.best_params_)
    return grid.best_estimator_


# ------------------------
# Random Forest
# ------------------------
def train_rf(X_train, y_train):
    model = RandomForestClassifier(
        n_estimators=100,      
        max_depth=10,          
        random_state=42,
        class_weight='balanced',
        n_jobs=1               
    )
    model.fit(X_train, y_train)
    return model






# ------------------------
# KNN
# ------------------------
def train_knn(X_train, y_train):
    model = KNeighborsClassifier(n_neighbors=3,algorithm='ball_tree')
    model.fit(X_train, y_train)
    return model


# ------------------------
# Decision Tree
# ------------------------
def train_dt(X_train, y_train):
    model = DecisionTreeClassifier(
        random_state=42,
        class_weight='balanced'
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

    # ----------------------------------------------
    # 1. Logistic Regression
    # ----------------------------------------------
    print("\n>>> Training Logistic Regression...")
    logreg = train_logreg(X_train, y_train)
    evaluate(logreg, X_test, y_test, "Logistic Regression")
    joblib.dump(logreg, out_dir / "logreg.pkl")
    print("Saved: logreg.pkl")

    # ----------------------------------------------
    # 2. Random Forest
    # ----------------------------------------------
    print("\n>>> Training Random Forest...")
    rf = train_rf(X_train, y_train)
    evaluate(rf, X_test, y_test, "Random Forest")
    joblib.dump(rf, out_dir / "rf.pkl")
    print("Saved: rf.pkl")

    

    # ----------------------------------------------
    # 4. KNN
    # ----------------------------------------------
    print("\n>>> Training KNN...")
    knn = train_knn(X_train, y_train)
    evaluate(knn, X_test, y_test, "KNN")
    joblib.dump(knn, out_dir / "knn.pkl")
    print("Saved: knn.pkl")

    # ----------------------------------------------
    # 5. Decision Tree
    # ----------------------------------------------
    print("\n>>> Training Decision Tree...")
    dt = train_dt(X_train, y_train)
    evaluate(dt, X_test, y_test, "Decision Tree")
    joblib.dump(dt, out_dir / "dt.pkl")
    print("Saved: dt.pkl")

    print("\n>>> All models saved successfully!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True)
    parser.add_argument("--out_dir", default="models")
    args = parser.parse_args()
    main(args)
  