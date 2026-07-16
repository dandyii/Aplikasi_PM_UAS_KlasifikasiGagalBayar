# Aplikasi Streamlit — Klasifikasi Risiko Gagal Bayar

Aplikasi ini memprediksi risiko gagal bayar menggunakan model **Random Forest** sebagai model produksi, serta menampilkan perbandingan performa dengan **Logistic Regression** dan **XGBoost** pada halaman *Evaluasi Model*.

## Struktur proyek

- `Default_Fin.csv`: dataset
- `app/app.py`: aplikasi Streamlit (5 halaman)
- `app/train_model.py`: training pipeline + export artefak ke `app/models/`
- `app/models/`: artefak model dan evaluasi
- `app/requirements.txt` dan `requirements.txt`: dependensi Python

## Menjalankan secara lokal

1. Install dependensi:
   - `python -m pip install -r requirements.txt`

2. Generate artefak model:
   - `python app/train_model.py`

3. Jalankan aplikasi:
   - `python -m streamlit run app/app.py`

Lalu buka URL yang muncul (biasanya `http://localhost:8501`).

## Deploy ke Streamlit Community Cloud

1. Push project ini ke GitHub (pastikan `Default_Fin.csv` dan folder `app/models/` ikut ter-commit).
2. Buka https://share.streamlit.io/ dan login dengan GitHub.
3. Klik **New app** lalu pilih repository dan branch.
4. Set **Main file path** ke `app/app.py`.
5. Klik **Deploy**.

Catatan:
- `requirements.txt` di root sudah disediakan supaya Streamlit Cloud mudah mendeteksi dependensi.
- Model produksi tetap Random Forest (halaman *Model Demo* memakai Random Forest + SHAP TreeExplainer).

