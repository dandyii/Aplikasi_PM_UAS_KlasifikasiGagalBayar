"""Streamlit web app — Credit Default Risk Classification."""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent
MODELS_DIR = BASE_DIR / "models"
DATA_PATH = BASE_DIR.parent / "Default_Fin.csv"

NUMERIC_COLS = [
    "Bank Balance",
    "Annual Salary",
    "Balance_to_Salary_Ratio",
    "Balance_per_Employment",
]

PAGES = [
    "Dashboard EDA",
    "Model Demo",
    "Evaluasi Model",
    "Interpretasi Hasil",
    "Dokumentasi",
]

# ---------------------------------------------------------------------------
# Custom CSS
# ---------------------------------------------------------------------------
CUSTOM_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}

.main .block-container {
    padding-top: 2rem;
    padding-bottom: 3rem;
    max-width: 1200px;
}

[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0f172a 0%, #1e293b 100%);
}

[data-testid="stSidebar"] * {
    color: #e2e8f0 !important;
}

.glass-card {
    background: rgba(255, 255, 255, 0.06);
    backdrop-filter: blur(14px);
    -webkit-backdrop-filter: blur(14px);
    border: 1px solid rgba(255, 255, 255, 0.12);
    border-radius: 16px;
    padding: 1.25rem 1.5rem;
    margin-bottom: 1rem;
    transition: transform 0.2s ease, box-shadow 0.2s ease;
}

.glass-card:hover {
    transform: translateY(-2px);
    box-shadow: 0 8px 32px rgba(0, 0, 0, 0.12);
}

.hero-title {
    font-size: 2rem;
    font-weight: 700;
    background: linear-gradient(135deg, #3b82f6, #8b5cf6);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin-bottom: 0.25rem;
}

.hero-subtitle {
    color: #64748b;
    font-size: 1rem;
    margin-bottom: 1.5rem;
}

.risk-badge {
    display: inline-block;
    padding: 0.55rem 1.4rem;
    border-radius: 999px;
    font-weight: 700;
    font-size: 1.1rem;
    color: white;
    letter-spacing: 0.05em;
}

.metric-label {
    font-size: 0.75rem;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    color: #94a3b8;
    margin-bottom: 0.25rem;
}

.metric-value {
    font-size: 1.75rem;
    font-weight: 700;
    color: #1e293b;
}

.stButton > button {
    background: linear-gradient(135deg, #3b82f6, #6366f1);
    color: white;
    border: none;
    border-radius: 10px;
    padding: 0.6rem 2rem;
    font-weight: 600;
    transition: opacity 0.2s;
}

.stButton > button:hover {
    opacity: 0.9;
    color: white;
    border: none;
}
</style>
"""


def inject_css() -> None:
    st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Data & artifact loaders
# ---------------------------------------------------------------------------
@st.cache_resource
def load_artifacts() -> dict:
    return {
        "model": joblib.load(MODELS_DIR / "random_forest_model.pkl"),
        "model_xgb": joblib.load(MODELS_DIR / "xgboost_model.pkl"),
        "scaler": joblib.load(MODELS_DIR / "scaler.pkl"),
        "feature_names": json.loads((MODELS_DIR / "feature_names.json").read_text(encoding="utf-8")),
        "salary_bins": json.loads((MODELS_DIR / "salary_bins.json").read_text(encoding="utf-8")),
        "eval": json.loads((MODELS_DIR / "eval_artifacts.json").read_text(encoding="utf-8")),
    }


@st.cache_resource
def load_shap_explainer(_model):
    import shap
    return shap.TreeExplainer(_model)


@st.cache_data
def load_dataset() -> pd.DataFrame:
    df = pd.read_csv(DATA_PATH)
    df = df.drop(columns=["Index"], errors="ignore")
    df = df.drop_duplicates().reset_index(drop=True)
    return df


# ---------------------------------------------------------------------------
# Preprocessing & prediction helpers
# ---------------------------------------------------------------------------
def preprocess_input(
    employed: int,
    bank_balance: float,
    annual_salary: float,
    salary_bins: list,
    scaler,
    feature_names: list,
) -> pd.DataFrame:
    df = pd.DataFrame(
        {
            "Employed": [employed],
            "Bank Balance": [bank_balance],
            "Annual Salary": [annual_salary],
        }
    )
    df["Balance_to_Salary_Ratio"] = df["Bank Balance"] / df["Annual Salary"]
    df["Balance_per_Employment"] = df["Bank Balance"] * df["Employed"]

    df["Salary_Bin"] = pd.cut(
        df["Annual Salary"],
        bins=salary_bins,
        labels=["rendah", "menengah", "tinggi"],
        include_lowest=True,
    )
    df = pd.get_dummies(df, columns=["Salary_Bin"], drop_first=True)

    for col in ["Salary_Bin_menengah", "Salary_Bin_tinggi"]:
        if col in df.columns:
            df[col] = df[col].astype(int)

    df = df.reindex(columns=feature_names, fill_value=0)
    df[NUMERIC_COLS] = scaler.transform(df[NUMERIC_COLS])
    return df


def get_risk_tier(prob: float) -> tuple[str, str, str]:
    if prob < 0.30:
        return "LOW", "#22c55e", "Risiko gagal bayar rendah"
    if prob <= 0.60:
        return "MEDIUM", "#f97316", "Risiko gagal bayar sedang"
    return "HIGH", "#ef4444", "Risiko gagal bayar tinggi"


def extract_shap_values(shap_output, index: int = 0) -> np.ndarray:
    if isinstance(shap_output, list):
        return np.asarray(shap_output[1][index])
    arr = np.asarray(shap_output)
    if arr.ndim == 3:
        return arr[index, :, 1]
    return arr[index]


def plot_confusion_matrix(cm: list, title: str) -> go.Figure:
    labels = ["Tidak Default", "Default"]
    fig = go.Figure(
        data=go.Heatmap(
            z=cm,
            x=labels,
            y=labels,
            colorscale="Blues",
            showscale=True,
            text=cm,
            texttemplate="%{text}",
            textfont={"size": 16},
        )
    )
    fig.update_layout(
        title=title,
        xaxis_title="Prediksi",
        yaxis_title="Aktual",
        height=380,
        margin=dict(t=50, b=40, l=40, r=20),
    )
    return fig


# ---------------------------------------------------------------------------
# Pages
# ---------------------------------------------------------------------------
def page_dashboard_eda() -> None:
    st.markdown('<p class="hero-title">Dashboard EDA</p>', unsafe_allow_html=True)
    st.markdown(
        '<p class="hero-subtitle">Exploratory Data Analysis — distribusi kelas, distribusi fitur, dan korelasi.</p>',
        unsafe_allow_html=True,
    )

    df = load_dataset()
    df_plot = df.copy()
    df_plot["Default_Label"] = df_plot["Defaulted?"].map({0: "Tidak Default", 1: "Default"})

    col1, col2, col3 = st.columns(3)
    col1.metric("Total Records", f"{len(df):,}")
    col2.metric("Default Rate", f"{df['Defaulted?'].mean():.2%}")
    col3.metric("Fitur", f"{df.shape[1] - 1}")

    st.markdown("---")

    # Class distribution
    st.subheader("Distribusi Kelas Default")
    class_counts = df["Defaulted?"].value_counts().reset_index()
    class_counts.columns = ["Defaulted?", "Count"]
    class_counts["Label"] = class_counts["Defaulted?"].map({0: "Tidak Default", 1: "Default"})

    fig_bar = px.bar(
        class_counts,
        x="Label",
        y="Count",
        color="Label",
        color_discrete_map={"Tidak Default": "#3b82f6", "Default": "#ef4444"},
        text="Count",
    )
    fig_bar.update_traces(textposition="outside")
    fig_bar.update_layout(showlegend=False, height=400)
    st.plotly_chart(fig_bar, use_container_width=True)

    # Violin plots
    st.subheader("Perbandingan Fitur vs Kelas Default")
    c1, c2 = st.columns(2)

    for container, feature, feature_label in [
        (c1, "Bank Balance", "Saldo Bank"),
        (c2, "Annual Salary", "Gaji Tahunan"),
    ]:
        with container:
            fig_v = px.violin(
                df_plot,
                x="Default_Label",
                y=feature,
                color="Default_Label",
                box=True,
                points="outliers",
                color_discrete_map={"Tidak Default": "#3b82f6", "Default": "#ef4444"},
                labels={"Default_Label": "Default", feature: feature_label},
            )
            fig_v.update_layout(
                title=f"{feature_label} vs Default",
                showlegend=False,
                height=420,
            )
            st.plotly_chart(fig_v, use_container_width=True)

    # Histogram overlay
    st.subheader("Distribusi Fitur (Histogram)")
    feature_choice = st.selectbox("Pilih fitur", ["Bank Balance", "Annual Salary"])
    fig_hist = px.histogram(
        df_plot,
        x=feature_choice,
        color="Default_Label",
        barmode="overlay",
        opacity=0.65,
        nbins=50,
        color_discrete_map={"Tidak Default": "#3b82f6", "Default": "#ef4444"},
        labels={"Default_Label": "Default"},
    )
    fig_hist.update_layout(height=420)
    st.plotly_chart(fig_hist, use_container_width=True)

    # Correlation heatmap
    st.subheader("Heatmap Korelasi")
    numeric_df = df[["Employed", "Bank Balance", "Annual Salary", "Defaulted?"]].copy()
    numeric_df["Balance_to_Salary_Ratio"] = numeric_df["Bank Balance"] / numeric_df["Annual Salary"]
    numeric_df["Balance_per_Employment"] = numeric_df["Bank Balance"] * numeric_df["Employed"]
    numeric_df = numeric_df[
        ["Employed", "Bank Balance", "Annual Salary", "Balance_to_Salary_Ratio", "Balance_per_Employment", "Defaulted?"]
    ]
    corr = numeric_df.corr()
    fig_corr = go.Figure(
        data=go.Heatmap(
            z=corr.values,
            x=corr.columns,
            y=corr.columns,
            colorscale="RdBu_r",
            zmid=0,
            text=np.round(corr.values, 2),
            texttemplate="%{text}",
            textfont={"size": 12},
        )
    )
    fig_corr.update_layout(height=450, title="Matriks Korelasi Fitur Numerik")
    st.plotly_chart(fig_corr, use_container_width=True)


def page_model_demo(artifacts: dict) -> None:
    st.markdown('<p class="hero-title">Model Demo</p>', unsafe_allow_html=True)
    st.markdown(
        '<p class="hero-subtitle">Masukkan profil nasabah untuk mendapatkan prediksi risiko gagal bayar.</p>',
        unsafe_allow_html=True,
    )

    model = artifacts["model"]
    scaler = artifacts["scaler"]
    feature_names = artifacts["feature_names"]
    salary_bins = artifacts["salary_bins"]

    with st.form("prediction_form"):
        col1, col2, col3 = st.columns(3)
        employed = col1.selectbox("Status Bekerja (Employed)", options=[1, 0], format_func=lambda x: "Bekerja" if x == 1 else "Tidak Bekerja")
        bank_balance = col2.number_input("Saldo Bank (Bank Balance)", min_value=0.0, value=50000.0, step=1000.0)
        annual_salary = col3.number_input("Gaji Tahunan (Annual Salary)", min_value=1.0, value=300000.0, step=5000.0)
        submitted = st.form_submit_button("Prediksi Risiko", use_container_width=True)

    if submitted:
        X_input = preprocess_input(employed, bank_balance, annual_salary, salary_bins, scaler, feature_names)
        prob = float(model.predict_proba(X_input)[0, 1])
        tier, color, desc = get_risk_tier(prob)

        st.markdown("---")
        c1, c2, c3 = st.columns([1, 2, 1])
        with c2:
            st.markdown(
                f'<div style="text-align:center">'
                f'<span class="risk-badge" style="background:{color}">{tier}</span>'
                f'<p style="margin-top:0.75rem;font-size:1.1rem;color:#475569">{desc}</p>'
                f'</div>',
                unsafe_allow_html=True,
            )

        m1, m2, m3 = st.columns(3)
        m1.metric("Probabilitas Default", f"{prob:.2%}")
        m2.metric("Prediksi Kelas", "Default" if prob >= 0.5 else "Tidak Default")
        m3.metric("Threshold", "0.50")

        st.subheader("Kontribusi Fitur (SHAP — Local Explanation)")
        try:
            with st.spinner("Menghitung SHAP values..."):
                explainer = load_shap_explainer(model)
                shap_vals = explainer.shap_values(X_input, check_additivity=False)
                sv = extract_shap_values(shap_vals)

            shap_df = pd.DataFrame({"Fitur": feature_names, "SHAP": sv}).sort_values("SHAP", key=abs, ascending=True)

            fig_shap = go.Figure(
                go.Bar(
                    x=shap_df["SHAP"],
                    y=shap_df["Fitur"],
                    orientation="h",
                    marker_color=["#ef4444" if v > 0 else "#22c55e" for v in shap_df["SHAP"]],
                    text=[f"{v:+.4f}" for v in shap_df["SHAP"]],
                    textposition="outside",
                )
            )
            fig_shap.update_layout(
                title="Kontribusi Fitur terhadap Probabilitas Default",
                xaxis_title="SHAP Value (positif → meningkatkan risiko)",
                height=420,
                margin=dict(l=20, r=20, t=50, b=20),
            )
            st.plotly_chart(fig_shap, use_container_width=True)
            st.caption("SHAP positif mendorong prediksi ke arah default; SHAP negatif mengurangi risiko.")
        except Exception as exc:
            st.warning(f"SHAP tidak dapat dihitung: {exc}")


def page_model_evaluation(artifacts: dict) -> None:
    st.markdown('<p class="hero-title">Evaluasi Model</p>', unsafe_allow_html=True)
    st.markdown(
        '<p class="hero-subtitle">Perbandingan Random Forest vs Logistic Regression vs XGBoost pada test set.</p>',
        unsafe_allow_html=True,
    )

    eval_data = artifacts["eval"]
    rf = eval_data["RandomForest"]
    lr = eval_data["LogisticRegression"]
    xgb = eval_data["XGBoost"]

    # Metrics table
    st.subheader("Tabel Metrik Klasifikasi")
    metrics_df = pd.DataFrame(
        {
            "Metrik": ["Accuracy", "Precision", "Recall", "F1-Score", "ROC-AUC", "PR-AUC"],
            "Random Forest": [
                rf["accuracy"], rf["precision"], rf["recall"],
                rf["f1_score"], rf["roc_auc"], rf["pr_auc"],
            ],
            "Logistic Regression": [
                lr["accuracy"], lr["precision"], lr["recall"],
                lr["f1_score"], lr["roc_auc"], lr["pr_auc"],
            ],
            "XGBoost": [
                xgb["accuracy"], xgb["precision"], xgb["recall"],
                xgb["f1_score"], xgb["roc_auc"], xgb["pr_auc"],
            ],
        }
    )
    metrics_display = metrics_df.copy()
    for col in ["Random Forest", "Logistic Regression", "XGBoost"]:
        metrics_display[col] = metrics_display[col].apply(lambda x: f"{x:.4f}")
    st.dataframe(metrics_display, use_container_width=True, hide_index=True)

    # Confusion matrices
    st.subheader("Confusion Matrix")
    c1, c2, c3 = st.columns(3)
    with c1:
        st.plotly_chart(plot_confusion_matrix(rf["confusion_matrix"], "Random Forest"), use_container_width=True)
    with c2:
        st.plotly_chart(plot_confusion_matrix(lr["confusion_matrix"], "Logistic Regression"), use_container_width=True)
    with c3:
        st.plotly_chart(plot_confusion_matrix(xgb["confusion_matrix"], "XGBoost"), use_container_width=True)

    # ROC curves
    st.subheader("ROC Curve")
    fig_roc = go.Figure()
    fig_roc.add_trace(go.Scatter(
        x=rf["roc_curve"]["fpr"], y=rf["roc_curve"]["tpr"],
        mode="lines", name=f"Random Forest (AUC={rf['roc_auc']:.3f})",
        line=dict(color="#3b82f6", width=2),
    ))
    fig_roc.add_trace(go.Scatter(
        x=lr["roc_curve"]["fpr"], y=lr["roc_curve"]["tpr"],
        mode="lines", name=f"Logistic Regression (AUC={lr['roc_auc']:.3f})",
        line=dict(color="#f97316", width=2),
    ))
    fig_roc.add_trace(go.Scatter(
        x=xgb["roc_curve"]["fpr"], y=xgb["roc_curve"]["tpr"],
        mode="lines", name=f"XGBoost (AUC={xgb['roc_auc']:.3f})",
        line=dict(color="#22c55e", width=2),
    ))
    fig_roc.add_trace(go.Scatter(
        x=[0, 1], y=[0, 1], mode="lines",
        name="Random Classifier", line=dict(dash="dash", color="#94a3b8"),
    ))
    fig_roc.update_layout(
        xaxis_title="False Positive Rate",
        yaxis_title="True Positive Rate",
        height=450,
        legend=dict(x=0.02, y=0.02),
    )
    st.plotly_chart(fig_roc, use_container_width=True)

    # PR curves
    st.subheader("Precision-Recall Curve")
    fig_pr = go.Figure()
    fig_pr.add_trace(go.Scatter(
        x=rf["pr_curve"]["recall"], y=rf["pr_curve"]["precision"],
        mode="lines", name=f"Random Forest (PR-AUC={rf['pr_auc']:.3f})",
        line=dict(color="#3b82f6", width=2),
    ))
    fig_pr.add_trace(go.Scatter(
        x=lr["pr_curve"]["recall"], y=lr["pr_curve"]["precision"],
        mode="lines", name=f"Logistic Regression (PR-AUC={lr['pr_auc']:.3f})",
        line=dict(color="#f97316", width=2),
    ))
    fig_pr.add_trace(go.Scatter(
        x=xgb["pr_curve"]["recall"], y=xgb["pr_curve"]["precision"],
        mode="lines", name=f"XGBoost (PR-AUC={xgb['pr_auc']:.3f})",
        line=dict(color="#22c55e", width=2),
    ))
    fig_pr.update_layout(
        xaxis_title="Recall",
        yaxis_title="Precision",
        height=450,
        legend=dict(x=0.55, y=0.95),
    )
    st.plotly_chart(fig_pr, use_container_width=True)


def page_model_interpretation(artifacts: dict) -> None:
    st.markdown('<p class="hero-title">Interpretasi Hasil</p>', unsafe_allow_html=True)
    st.markdown(
        '<p class="hero-subtitle">Feature importance global Random Forest dan implikasi bisnis.</p>',
        unsafe_allow_html=True,
    )

    importance = artifacts["eval"]["RandomForest"]["feature_importance"]
    imp_df = (
        pd.DataFrame(list(importance.items()), columns=["Fitur", "Importance"])
        .sort_values("Importance", ascending=True)
    )

    fig_imp = go.Figure(
        go.Bar(
            x=imp_df["Importance"],
            y=imp_df["Fitur"],
            orientation="h",
            marker_color=imp_df["Importance"],
            marker_colorscale="Viridis",
            text=[f"{v:.1%}" for v in imp_df["Importance"]],
            textposition="outside",
        )
    )
    fig_imp.update_layout(
        title="Global Feature Importance — Random Forest",
        xaxis_title="Importance",
        height=420,
    )
    st.plotly_chart(fig_imp, use_container_width=True)

    st.subheader("Insight Bisnis")
    st.markdown(
        """
<div class="glass-card">

**1. Bank Balance (~47%) — indikator risiko dominan**

Saldo bank yang tinggi berkorelasi kuat dengan risiko gagal bayar. Nasabah dengan saldo besar cenderung
memiliki exposure kredit lebih tinggi, sehingga potensi kerugian lebih besar jika terjadi default.

**Rekomendasi kebijakan:**
- Terapkan **limit kredit dinamis** berdasarkan saldo bank nasabah.
- Lakukan **review berkala** untuk nasabah dengan saldo di atas persentil 75.

---

**2. Balance-to-Salary Ratio (~25%) — rasio utang/pendapatan**

Rasio saldo terhadap gaji mengukur seberapa besar beban finansial relatif terhadap kemampuan bayar.
Rasio tinggi menandakan nasabah memiliki kewajiban besar dibanding pendapatan.

**Rekomendasi kebijakan:**
- Tetapkan **threshold maksimum rasio** (mis. 0.5) sebagai syarat persetujuan kredit.
- Prioritaskan verifikasi pendapatan untuk aplikasi dengan rasio di atas threshold.

---

**3. Balance per Employment (~16%) — interaksi pekerjaan & saldo**

Fitur ini menangkap interaksi antara status pekerjaan dan saldo bank. Nasabah bekerja dengan saldo
tinggi memiliki profil risiko berbeda dari yang tidak bekerja.

**Rekomendasi kebijakan:**
- Segmentasi scoring berdasarkan status pekerjaan **dan** saldo secara bersamaan.
- Berikan bobot lebih tinggi pada verifikasi employment untuk saldo di atas median.

---

**4. Implikasi Model Selection**

Random Forest dipilih sebagai model final karena:
- Recall kelas default lebih seimbang untuk deteksi dini nasabah berisiko.
- PR-AUC lebih representatif pada data imbalanced.
- Kompatibel dengan SHAP TreeExplainer untuk interpretabilitas lokal dan global.

---

**5. Kenapa bukan XGBoost?**

Walaupun XGBoost sangat kompetitif, PR-AUC pada cross validation (skenario SMOTE) sedikit di bawah Random Forest
(RF: 0.9702 vs XGBoost: 0.9669). Selain itu, Random Forest sudah terintegrasi mulus dengan SHAP TreeExplainer yang
dipakai di halaman **Model Demo**, sehingga jalur produksi lebih stabil tanpa perubahan besar pada interpretabilitas.

</div>
        """,
        unsafe_allow_html=True,
    )


def page_documentation() -> None:
    st.markdown('<p class="hero-title">Dokumentasi</p>', unsafe_allow_html=True)
    st.markdown(
        '<p class="hero-subtitle">Sumber data, metodologi, dan panduan penggunaan aplikasi.</p>',
        unsafe_allow_html=True,
    )

    tab1, tab2, tab3 = st.tabs(["Sumber Dataset", "Metodologi", "Panduan Pengguna"])

    with tab1:
        st.markdown(
            """
### Sumber Dataset

Dataset yang digunakan pada aplikasi ini adalah file **`Default_Fin.csv`** yang tersimpan
di folder proyek dan dipakai secara langsung pada proses training maupun inferensi aplikasi.
Dengan demikian, dokumentasi aplikasi ini mengikuti struktur dan atribut pada file dataset
yang benar-benar digunakan, bukan mengacu ke dataset lain.

| Atribut | Deskripsi |
|---------|-----------|
| `Employed` | Status pekerjaan (1 = bekerja, 0 = tidak) |
| `Bank Balance` | Saldo bank nasabah |
| `Annual Salary` | Gaji tahunan |
| `Defaulted?` | Label target (1 = gagal bayar, 0 = tidak) |

**Referensi:**
- Sumber unduh dataset: [Kaggle - Loan Default Prediction](https://www.kaggle.com/datasets/kmldas/loan-default-prediction)
- Dataset lokal proyek yang digunakan aplikasi: `Default_Fin.csv`
- Diproses melalui pipeline training pada `app/train_model.py`
            """
        )

    with tab2:
        st.markdown(
            """
### Metodologi Pipeline ML

**1. Data Cleaning**
- Drop kolom `Index` (identifier).
- Hapus baris duplikat.

**2. Train-Validation-Test Split**
- Stratified split 70% / 15% / 15% (`random_state=42`).

**3. Feature Engineering**
- `Balance_to_Salary_Ratio` = Bank Balance / Annual Salary
- `Balance_per_Employment` = Bank Balance × Employed
- `Salary_Bin` — quantile binning (3 bins) dari training set, one-hot encoded

**4. Feature Scaling**
- `StandardScaler` pada fitur numerik; fitur biner tidak di-scale.

**5. Class Imbalance — SMOTE**
- Synthetic Minority Over-sampling Technique pada training set.
- Fallback manual SMOTE jika `imbalanced-learn` tidak tersedia.

**6. Model Training**
- **Random Forest**: `n_estimators=400`, `max_depth=None`, `min_samples_leaf=3`
- **Logistic Regression**: baseline dengan `max_iter=1000`

**7. Model Selection**
- Random Forest dipilih berdasarkan PR-AUC, recall kelas default, dan interpretabilitas SHAP.

**8. Risk Tiering**
| Tier | Probabilitas | Warna |
|------|-------------|-------|
| LOW | < 0.30 | Hijau |
| MEDIUM | 0.30 – 0.60 | Oranye |
| HIGH | > 0.60 | Merah |
            """
        )

    with tab3:
        st.markdown(
            """
### Panduan Pengguna (Non-Teknis)

**Langkah 1 — Buka halaman "Model Demo"**
Pilih menu **Model Demo** di sidebar kiri.

**Langkah 2 — Isi profil nasabah**
- **Status Bekerja**: pilih "Bekerja" atau "Tidak Bekerja".
- **Saldo Bank**: masukkan saldo rekening nasabah (angka).
- **Gaji Tahunan**: masukkan pendapatan tahunan nasabah.

**Langkah 3 — Klik "Prediksi Risiko"**
Sistem akan menampilkan:
- **Badge risiko** (LOW / MEDIUM / HIGH) dengan warna.
- **Probabilitas default** dalam persen.
- **Grafik SHAP** — fitur mana yang paling mempengaruhi prediksi.

**Langkah 4 — Interpretasi hasil**
- **LOW (Hijau)**: profil aman, proses kredit normal.
- **MEDIUM (Oranye)**: perlu verifikasi tambahan sebelum persetujuan.
- **HIGH (Merah)**: disarankan tolak atau minta agunan/jaminan tambahan.

**Halaman lain:**
- **Dashboard EDA** — lihat distribusi data dan korelasi fitur.
- **Evaluasi Model** — bandingkan performa Random Forest vs Logistic Regression.
- **Interpretasi Hasil** — pahami fitur risiko dominan dan rekomendasi kebijakan.
            """
        )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    st.set_page_config(
        page_title="Credit Default Risk",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    inject_css()

    # Verify artifacts exist
    required = [
        MODELS_DIR / "random_forest_model.pkl",
        MODELS_DIR / "xgboost_model.pkl",
        MODELS_DIR / "scaler.pkl",
        MODELS_DIR / "feature_names.json",
        MODELS_DIR / "salary_bins.json",
        MODELS_DIR / "eval_artifacts.json",
    ]
    missing = [p.name for p in required if not p.exists()]
    if missing:
        st.error(f"Artefak model belum tersedia: {', '.join(missing)}")
        st.info("Jalankan `python app/train_model.py` terlebih dahulu untuk melatih dan mengekspor model.")
        st.stop()

    artifacts = load_artifacts()

    with st.sidebar:
        st.markdown("## 🏦 Credit Risk")
        st.markdown("Klasifikasi Risiko Gagal Bayar")
        st.markdown("---")
        page = st.radio("Navigasi", PAGES, label_visibility="collapsed")
        st.markdown("---")
        st.caption("UAS Pembelajaran Mesin — Random Forest + SHAP")
        st.markdown("### Profil Pengembang")
        st.markdown(
            """
**Nama**: Dandy Prasetyo Nugroho  
**NIM**: A11.2024.15645  
**Kelas**: A11.4404  
**Program Studi**: Teknik Informatika  
**Universitas**: Universitas Dian Nuswantoro
            """
        )

    page_map = {
        "Dashboard EDA": page_dashboard_eda,
        "Model Demo": lambda: page_model_demo(artifacts),
        "Evaluasi Model": lambda: page_model_evaluation(artifacts),
        "Interpretasi Hasil": lambda: page_model_interpretation(artifacts),
        "Dokumentasi": page_documentation,
    }
    page_map[page]()


if __name__ == "__main__":
    main()
