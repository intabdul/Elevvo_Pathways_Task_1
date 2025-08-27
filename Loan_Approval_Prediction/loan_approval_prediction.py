import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
    roc_curve,
    precision_recall_fscore_support,
)
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier

# Try to use SMOTE if available
try:
    from imblearn.over_sampling import SMOTE
    from imblearn.pipeline import Pipeline as ImbPipeline
    SMOTE_AVAILABLE = True
except ImportError:
    SMOTE_AVAILABLE = False


#  Utility functions 

def coerce_target(y: pd.Series) -> pd.Series:
    """Convert text target into binary 1/0 for classification"""
    y = y.astype(str).str.strip().str.lower()

    # Common mappings
    if set(y.unique()).issubset({"y", "n"}):
        return y.map({"y": 1, "n": 0})
    if set(y.unique()).issubset({"yes", "no"}):
        return y.map({"yes": 1, "no": 0})
    if set(y.unique()).issubset({"approve", "reject"}):
        return y.map({"approve": 1, "reject": 0})
    if set(y.unique()).issubset({"approved", "not approved"}):
        return y.map({"approved": 1, "not approved": 0})

    # Fallback: use factorize (first label → 0, second → 1)
    codes, uniques = pd.factorize(y)
    print("⚠️ Target labels factorized automatically:", dict(enumerate(uniques)))
    return pd.Series(codes, index=y.index)

def infer_feature_types(X: pd.DataFrame):
    num_cols = X.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = [c for c in X.columns if c not in num_cols]
    return num_cols, cat_cols


def build_preprocessor(num_cols, cat_cols):
    num_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler(with_mean=False)),
    ])
    cat_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore")),
    ])
    return ColumnTransformer([
        ("num", num_pipe, num_cols),
        ("cat", cat_pipe, cat_cols),
    ])


def evaluate_model(name, pipe, X_train, y_train, X_test, y_test):
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    try:
        cv_scores = cross_val_score(pipe, X_train, y_train, cv=cv, scoring="f1_macro")
        cv_mean = np.mean(cv_scores)
    except:
        cv_mean = np.nan
    
    pipe.fit(X_train, y_train)
    y_pred = pipe.predict(X_test)

    metrics = {}
    p, r, f1, _ = precision_recall_fscore_support(y_test, y_pred, average="binary", zero_division=0)
    metrics.update({"precision":p, "recall":r, "f1":f1})
    metrics["confusion_matrix"] = confusion_matrix(y_test, y_pred).tolist()

    if hasattr(pipe, "predict_proba"):
        try:
            y_proba = pipe.predict_proba(X_test)[:,1]
            metrics["roc_auc"] = roc_auc_score(y_test, y_proba)
        except:
            metrics["roc_auc"] = np.nan
    
    metrics["cv_f1_macro_mean"] = cv_mean
    metrics["classification_report"] = classification_report(y_test, y_pred, zero_division=0)

    return metrics


# Main 

if __name__ == "__main__":
    DATASET_PATH = "loan_approval_dataset.csv"  # make sure the file is here
    df = pd.read_csv(DATASET_PATH)

    target_col = " loan_status"    # keep the space to match your CSV
    y = coerce_target(df[target_col])
    X = df.drop(columns=[target_col])

    # Drop obvious ID column if exists
    for c in ["Loan_ID","loan_id","id"]:
        if c in X.columns:
            X = X.drop(columns=[c])

    # Train/Test Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    # Feature processing
    num_cols, cat_cols = infer_feature_types(X)
    pre = build_preprocessor(num_cols, cat_cols)

    # Define models
    pipelines = []
    lr = LogisticRegression(solver="liblinear", max_iter=1000, random_state=42)
    dt = DecisionTreeClassifier(random_state=42)

    if SMOTE_AVAILABLE:
        pipelines.append(("LogReg+SMOTE", ImbPipeline([("pre", pre), ("smote", SMOTE()), ("clf", lr)])))
        pipelines.append(("Tree+SMOTE", ImbPipeline([("pre", pre), ("smote", SMOTE()), ("clf", dt)])))
    
    pipelines.append(("LogReg+ClassWeight", Pipeline([("pre", pre), ("clf", LogisticRegression(solver="liblinear", class_weight="balanced", random_state=42))])))
    pipelines.append(("Tree+ClassWeight", Pipeline([("pre", pre), ("clf", DecisionTreeClassifier(class_weight="balanced", random_state=42))])))

    # Train & Evaluate
    results = {}
    for name, pipe in pipelines:
        metrics = evaluate_model(name, pipe, X_train, y_train, X_test, y_test)
        results[name] = metrics
        print("\n", "="*20, name, "="*20)
        print("Precision:", metrics["precision"])
        print("Recall:", metrics["recall"])
        print("F1:", metrics["f1"])
        print("ROC AUC:", metrics["roc_auc"])
        print("Confusion Matrix:\n", metrics["confusion_matrix"])
        print("Classification Report:\n", metrics["classification_report"])

    # Plot ROC curves
    plt.figure()
    for name, pipe in pipelines:
        if hasattr(pipe, "predict_proba"):
            y_proba = pipe.fit(X_train, y_train).predict_proba(X_test)[:,1]
            fpr, tpr, _ = roc_curve(y_test, y_proba)
            plt.plot(fpr, tpr, label=name)
    plt.plot([0,1],[0,1], "--")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curves")
    plt.legend()
    plt.show()
