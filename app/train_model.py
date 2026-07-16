import os
import json
import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import NearestNeighbors
from xgboost import XGBClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, average_precision_score,
    confusion_matrix, roc_curve, precision_recall_curve
)

RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)

# Setup paths relative to the script location
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)
csv_path = os.path.join(project_root, "Default_Fin.csv")
models_dir = os.path.join(script_dir, "models")
os.makedirs(models_dir, exist_ok=True)

# 1. Manual SMOTE and Apply SMOTE functions as in the notebook
def manual_smote(X, y, minority_label=1, k=5, random_state=RANDOM_STATE):
    """Implementasi sederhana SMOTE: interpolasi linear antar titik minoritas dan k-NN-nya."""
    rng = np.random.RandomState(random_state)
    X = np.asarray(X, dtype=float)
    y = np.asarray(y)

    X_min = X[y == minority_label]
    n_min = X_min.shape[0]
    n_maj = (y != minority_label).sum()
    n_to_generate = n_maj - n_min
    if n_to_generate <= 0:
        return X, y

    nn = NearestNeighbors(n_neighbors=k + 1).fit(X_min)
    neighbors = nn.kneighbors(X_min, return_distance=False)[:, 1:]

    synthetic = np.zeros((n_to_generate, X.shape[1]))
    for i in range(n_to_generate):
        idx = rng.randint(0, n_min)
        nn_idx = neighbors[idx][rng.randint(0, k)]
        gap = rng.rand()
        synthetic[i] = X_min[idx] + gap * (X_min[nn_idx] - X_min[idx])

    X_res = np.vstack([X, synthetic])
    y_res = np.concatenate([y, np.full(n_to_generate, minority_label)])
    return X_res, y_res

def apply_smote(X, y, random_state=RANDOM_STATE):
    """Coba pakai imblearn.SMOTE, fallback ke manual_smote bila tidak tersedia."""
    cols = X.columns if hasattr(X, "columns") else None
    try:
        from imblearn.over_sampling import SMOTE
        sm = SMOTE(random_state=random_state)
        X_res, y_res = sm.fit_resample(X, y)
        print("Menggunakan imbalanced-learn SMOTE.")
    except ImportError:
        X_res, y_res = manual_smote(np.asarray(X), np.asarray(y), random_state=random_state)
        print("Menggunakan fallback manual SMOTE.")

    if cols is not None:
        X_res = pd.DataFrame(X_res, columns=cols)
    y_res = pd.Series(np.asarray(y_res), name=getattr(y, "name", "target"))
    return X_res, y_res

def main():
    print(f"Membaca dataset dari: {csv_path}")
    df = pd.read_csv(csv_path)

    # 2. Data cleaning
    df_clean = df.drop(columns=["Index"], errors="ignore").copy()
    df_clean = df_clean.drop_duplicates().reset_index(drop=True)
    print(f"Shape setelah data cleaning (drop Index & duplicates): {df_clean.shape}")

    # 3. Split: stratified train:val:test = 70:15:15
    X = df_clean[["Employed", "Bank Balance", "Annual Salary"]].copy()
    y = df_clean["Defaulted?"].copy()

    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.30, stratify=y, random_state=RANDOM_STATE
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5, stratify=y_temp, random_state=RANDOM_STATE
    )
    print(f"Train shape: {X_train.shape} | Val shape: {X_val.shape} | Test shape: {X_test.shape}")

    # 4. Feature Engineering
    def add_engineered_features(d: pd.DataFrame) -> pd.DataFrame:
        d = d.copy()
        d["Balance_to_Salary_Ratio"] = d["Bank Balance"] / d["Annual Salary"]
        d["Balance_per_Employment"] = d["Bank Balance"] * d["Employed"]
        return d

    X_train_fe = add_engineered_features(X_train)
    X_val_fe = add_engineered_features(X_val)
    X_test_fe = add_engineered_features(X_test)

    # Bins dihitung dari training set saja
    bins = pd.qcut(X_train_fe["Annual Salary"], q=3, retbins=True, labels=["rendah", "menengah", "tinggi"])[1]
    bins_list = bins.tolist()

    X_train_fe["Salary_Bin"] = pd.cut(X_train_fe["Annual Salary"], bins=bins, labels=["rendah", "menengah", "tinggi"], include_lowest=True)
    X_val_fe["Salary_Bin"] = pd.cut(X_val_fe["Annual Salary"], bins=bins, labels=["rendah", "menengah", "tinggi"], include_lowest=True)
    X_test_fe["Salary_Bin"] = pd.cut(X_test_fe["Annual Salary"], bins=bins, labels=["rendah", "menengah", "tinggi"], include_lowest=True)

    # One-hot encode Salary_Bin
    X_train_fe = pd.get_dummies(X_train_fe, columns=["Salary_Bin"], drop_first=True)
    X_val_fe = pd.get_dummies(X_val_fe, columns=["Salary_Bin"], drop_first=True)
    X_test_fe = pd.get_dummies(X_test_fe, columns=["Salary_Bin"], drop_first=True)

    # Pastikan tipe dummy columns menjadi integer (0 atau 1) untuk konsistensi tipe
    for col in ["Salary_Bin_menengah", "Salary_Bin_tinggi"]:
        if col in X_train_fe.columns:
            X_train_fe[col] = X_train_fe[col].astype(int)
        if col in X_val_fe.columns:
            X_val_fe[col] = X_val_fe[col].astype(int)
        if col in X_test_fe.columns:
            X_test_fe[col] = X_test_fe[col].astype(int)

    X_val_fe = X_val_fe.reindex(columns=X_train_fe.columns, fill_value=0)
    X_test_fe = X_test_fe.reindex(columns=X_train_fe.columns, fill_value=0)

    feature_names = X_train_fe.columns.tolist()
    print(f"Urutan fitur final: {feature_names}")

    # 5. Scaling
    numeric_cols = ["Bank Balance", "Annual Salary", "Balance_to_Salary_Ratio", "Balance_per_Employment"]
    scaler = StandardScaler()
    X_train_scaled = X_train_fe.copy()
    X_val_scaled = X_val_fe.copy()
    X_test_scaled = X_test_fe.copy()

    X_train_scaled[numeric_cols] = scaler.fit_transform(X_train_fe[numeric_cols])
    X_val_scaled[numeric_cols] = scaler.transform(X_val_fe[numeric_cols])
    X_test_scaled[numeric_cols] = scaler.transform(X_test_fe[numeric_cols])

    # 6. Penanganan imbalance menggunakan SMOTE pada training set
    X_train_res, y_train_res = apply_smote(X_train_scaled, y_train)

    # 7. Model training
    # Random Forest Classifier (GridSearchCV best parameters: depth=None, leaf=3, estimators=400)
    print("Melatih model Random Forest...")
    rf_model = RandomForestClassifier(n_estimators=400, max_depth=None, min_samples_leaf=3, random_state=RANDOM_STATE, n_jobs=-1)
    rf_model.fit(X_train_res, y_train_res)

    # Logistic Regression Classifier
    print("Melatih model Logistic Regression...")
    lr_model = LogisticRegression(max_iter=1000, random_state=RANDOM_STATE)
    lr_model.fit(X_train_res, y_train_res)

    # XGBoost Classifier (hasil tuning GridSearchCV di notebook)
    print("Melatih model XGBoost...")
    xgb_model = XGBClassifier(
        n_estimators=200,
        max_depth=7,
        learning_rate=0.1,
        eval_metric="logloss",
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )
    xgb_model.fit(X_train_res, y_train_res)

    # 8. Evaluasi pada Test Set
    # Random Forest Evaluation
    y_pred_rf = rf_model.predict(X_test_scaled)
    y_proba_rf = rf_model.predict_proba(X_test_scaled)[:, 1]

    accuracy_rf = accuracy_score(y_test, y_pred_rf)
    precision_rf = precision_score(y_test, y_pred_rf)
    recall_rf = recall_score(y_test, y_pred_rf)
    f1_rf = f1_score(y_test, y_pred_rf)
    roc_auc_rf = roc_auc_score(y_test, y_proba_rf)
    pr_auc_rf = average_precision_score(y_test, y_proba_rf)
    cm_rf = confusion_matrix(y_test, y_pred_rf).tolist()

    fpr_rf, tpr_rf, thr_roc_rf = roc_curve(y_test, y_proba_rf)
    p_rf, r_rf, thr_pr_rf = precision_recall_curve(y_test, y_proba_rf)

    # Logistic Regression Evaluation
    y_pred_lr = lr_model.predict(X_test_scaled)
    y_proba_lr = lr_model.predict_proba(X_test_scaled)[:, 1]

    accuracy_lr = accuracy_score(y_test, y_pred_lr)
    precision_lr = precision_score(y_test, y_pred_lr)
    recall_lr = recall_score(y_test, y_pred_lr)
    f1_lr = f1_score(y_test, y_pred_lr)
    roc_auc_lr = roc_auc_score(y_test, y_proba_lr)
    pr_auc_lr = average_precision_score(y_test, y_proba_lr)
    cm_lr = confusion_matrix(y_test, y_pred_lr).tolist()

    fpr_lr, tpr_lr, thr_roc_lr = roc_curve(y_test, y_proba_lr)
    p_lr, r_lr, thr_pr_lr = precision_recall_curve(y_test, y_proba_lr)

    # XGBoost Evaluation
    y_pred_xgb = xgb_model.predict(X_test_scaled)
    y_proba_xgb = xgb_model.predict_proba(X_test_scaled)[:, 1]

    accuracy_xgb = accuracy_score(y_test, y_pred_xgb)
    precision_xgb = precision_score(y_test, y_pred_xgb)
    recall_xgb = recall_score(y_test, y_pred_xgb)
    f1_xgb = f1_score(y_test, y_pred_xgb)
    roc_auc_xgb = roc_auc_score(y_test, y_proba_xgb)
    pr_auc_xgb = average_precision_score(y_test, y_proba_xgb)
    cm_xgb = confusion_matrix(y_test, y_pred_xgb).tolist()

    fpr_xgb, tpr_xgb, thr_roc_xgb = roc_curve(y_test, y_proba_xgb)
    p_xgb, r_xgb, thr_pr_xgb = precision_recall_curve(y_test, y_proba_xgb)

    # 9. Feature importance dictionary (Random Forest & XGBoost) & Coefficients dictionary (Logistic Regression)
    rf_feat_importance = dict(zip(feature_names, rf_model.feature_importances_.tolist()))
    xgb_feat_importance = dict(zip(feature_names, xgb_model.feature_importances_.tolist()))
    lr_coefficients = dict(zip(feature_names, lr_model.coef_[0].tolist()))

    print("\n=== HASIL EVALUASI TEST SET ===")
    print(f"Random Forest - Accuracy: {accuracy_rf:.4f}, Precision (kelas 1): {precision_rf:.4f}, Recall (kelas 1): {recall_rf:.4f}, PR-AUC: {pr_auc_rf:.4f}")
    print(f"Logistic Regression - Accuracy: {accuracy_lr:.4f}, Precision (kelas 1): {precision_lr:.4f}, Recall (kelas 1): {recall_lr:.4f}, PR-AUC: {pr_auc_lr:.4f}")
    print(f"XGBoost - Accuracy: {accuracy_xgb:.4f}, Precision (kelas 1): {precision_xgb:.4f}, Recall (kelas 1): {recall_xgb:.4f}, PR-AUC: {pr_auc_xgb:.4f}")

    # 10. Menyimpan file model & artifacts ke folder app/models/
    rf_path = os.path.join(models_dir, "random_forest_model.pkl")
    xgb_path = os.path.join(models_dir, "xgboost_model.pkl")
    scaler_path = os.path.join(models_dir, "scaler.pkl")
    feature_names_path = os.path.join(models_dir, "feature_names.json")
    salary_bins_path = os.path.join(models_dir, "salary_bins.json")
    eval_artifacts_path = os.path.join(models_dir, "eval_artifacts.json")

    print(f"\nMenyimpan model Random Forest ke: {rf_path}")
    joblib.dump(rf_model, rf_path)

    print(f"Menyimpan model XGBoost ke: {xgb_path}")
    joblib.dump(xgb_model, xgb_path)

    print(f"Menyimpan scaler ke: {scaler_path}")
    joblib.dump(scaler, scaler_path)

    print(f"Menyimpan feature_names ke: {feature_names_path}")
    with open(feature_names_path, "w") as f:
        json.dump(feature_names, f, indent=4)

    print(f"Menyimpan salary_bins ke: {salary_bins_path}")
    with open(salary_bins_path, "w") as f:
        json.dump(bins_list, f, indent=4)

    # Menyimpan evaluation artifacts
    eval_artifacts = {
        "RandomForest": {
            "accuracy": accuracy_rf,
            "precision": precision_rf,
            "recall": recall_rf,
            "f1_score": f1_rf,
            "roc_auc": roc_auc_rf,
            "pr_auc": pr_auc_rf,
            "confusion_matrix": cm_rf,
            "roc_curve": {
                "fpr": fpr_rf.tolist(),
                "tpr": tpr_rf.tolist(),
                "thresholds": thr_roc_rf.tolist(),
            },
            "pr_curve": {
                "precision": p_rf.tolist(),
                "recall": r_rf.tolist(),
                "thresholds": thr_pr_rf.tolist(),
            },
            "feature_importance": rf_feat_importance
        },
        "LogisticRegression": {
            "accuracy": accuracy_lr,
            "precision": precision_lr,
            "recall": recall_lr,
            "f1_score": f1_lr,
            "roc_auc": roc_auc_lr,
            "pr_auc": pr_auc_lr,
            "confusion_matrix": cm_lr,
            "roc_curve": {
                "fpr": fpr_lr.tolist(),
                "tpr": tpr_lr.tolist(),
                "thresholds": thr_roc_lr.tolist(),
            },
            "pr_curve": {
                "precision": p_lr.tolist(),
                "recall": r_lr.tolist(),
                "thresholds": thr_pr_lr.tolist(),
            },
            "coefficients": lr_coefficients
        },
        "XGBoost": {
            "accuracy": accuracy_xgb,
            "precision": precision_xgb,
            "recall": recall_xgb,
            "f1_score": f1_xgb,
            "roc_auc": roc_auc_xgb,
            "pr_auc": pr_auc_xgb,
            "confusion_matrix": cm_xgb,
            "roc_curve": {
                "fpr": fpr_xgb.tolist(),
                "tpr": tpr_xgb.tolist(),
                "thresholds": thr_roc_xgb.tolist(),
            },
            "pr_curve": {
                "precision": p_xgb.tolist(),
                "recall": r_xgb.tolist(),
                "thresholds": thr_pr_xgb.tolist(),
            },
            "feature_importance": xgb_feat_importance
        },
    }

    print(f"Menyimpan eval_artifacts ke: {eval_artifacts_path}")
    with open(eval_artifacts_path, "w") as f:
        json.dump(eval_artifacts, f, indent=4)

    print("\nProses training model selesai dan semua artefak berhasil diekspor!")

if __name__ == "__main__":
    main()
