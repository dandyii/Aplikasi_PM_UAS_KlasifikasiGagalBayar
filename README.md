# Aplikasi Web Klasifikasi Risiko Gagal Bayar

Repositori ini berisi aplikasi Streamlit untuk memprediksi risiko gagal bayar nasabah menggunakan pendekatan machine learning. Model produksi yang dipakai pada aplikasi adalah **Random Forest**, sementara **Logistic Regression** dan **XGBoost** tetap ditampilkan sebagai pembanding pada halaman evaluasi model.

## Demo aplikasi

Aplikasi yang sudah dideploy dapat diakses di:

[https://aplikasipmuasklasifikasigagalbayar-sjlhpmb9bbpx2zzbkfpjoz.streamlit.app/](https://aplikasipmuasklasifikasigagalbayar-sjlhpmb9bbpx2zzbkfpjoz.streamlit.app/)

## Ringkasan proyek

- **Judul proyek**: Klasifikasi Risiko Gagal Bayar
- **Platform**: Streamlit
- **Bahasa pemrograman**: Python
- **Dataset utama yang digunakan aplikasi**: `Default_Fin.csv`
- **Dataset hasil preprocessing penuh**: `Default_Fin_processed_full.csv`
- **Sumber unduh dataset**: Kaggle, [Loan Default Prediction](https://www.kaggle.com/datasets/kmldas/loan-default-prediction)
- **Jumlah data setelah cleaning**: 10.000 baris
- **Fitur utama**: `Employed`, `Bank Balance`, `Annual Salary`
- **Label target**: `Defaulted?`
- **Model produksi**: Random Forest
- **Model pembanding**: Logistic Regression dan XGBoost

## Dataset proyek

Proyek ini menyimpan dua versi dataset agar alur pengerjaan dan dokumentasi akademik lebih mudah dipahami.

- `Default_Fin.csv` adalah dataset utama yang dipakai oleh aplikasi Streamlit, proses training, evaluasi model, dan inferensi.
- `Default_Fin_processed_full.csv` adalah dataset hasil preprocessing penuh yang ditambahkan untuk menunjukkan bentuk data setelah transformasi fitur, encoding, scaling, dan penandaan `split`.
- File processed tersebut disediakan untuk kebutuhan dokumentasi dan perbandingan dengan data mentah, bukan sebagai input utama pipeline aplikasi saat ini.

## Fitur aplikasi

Aplikasi memiliki 5 halaman utama:

- **Dashboard EDA**: menampilkan distribusi kelas, distribusi fitur, violin plot, histogram, dan heatmap korelasi.
- **Model Demo**: form prediksi interaktif untuk memasukkan profil nasabah dan melihat hasil klasifikasi risiko.
- **Evaluasi Model**: perbandingan Random Forest, Logistic Regression, dan XGBoost menggunakan metrik klasifikasi, confusion matrix, ROC curve, dan precision-recall curve.
- **Interpretasi Hasil**: visualisasi feature importance global serta penjelasan implikasi bisnis model.
- **Dokumentasi**: ringkasan sumber dataset, metodologi pipeline machine learning, dan panduan penggunaan aplikasi.

## Metodologi machine learning

Pipeline training pada `app/train_model.py` meliputi:

1. **Data cleaning**
   - Menghapus kolom `Index` bila tersedia.
   - Menghapus data duplikat.

2. **Train-validation-test split**
   - Stratified split dengan proporsi `70% / 15% / 15%`.
   - `random_state=42`.

3. **Feature engineering**
   - `Balance_to_Salary_Ratio` = `Bank Balance / Annual Salary`
   - `Balance_per_Employment` = `Bank Balance * Employed`
   - `Salary_Bin` menggunakan quantile binning dari data training

4. **Feature scaling**
   - `StandardScaler` digunakan pada fitur numerik.

5. **Penanganan class imbalance**
   - Menggunakan **SMOTE** pada data training.
   - Tersedia fallback ke implementasi manual jika `imbalanced-learn` tidak tersedia.

6. **Training model**
   - **Random Forest** sebagai model utama
   - **Logistic Regression** sebagai baseline
   - **XGBoost** sebagai model pembanding tambahan

7. **Interpretabilitas**
   - Halaman **Model Demo** menggunakan **SHAP TreeExplainer** untuk menjelaskan kontribusi fitur terhadap prediksi individual.

## Struktur proyek

```text
Aplikasi_PM_UAS_KlasifikasiGagalBayar/
├── app/
│   ├── app.py
│   ├── train_model.py
│   ├── requirements.txt
│   └── models/
│       ├── eval_artifacts.json
│       ├── feature_names.json
│       ├── random_forest_model.pkl
│       ├── salary_bins.json
│       ├── scaler.pkl
│       └── xgboost_model.pkl
├── Default_Fin.csv
├── Default_Fin_processed_full.csv
├── Eksperimen_Klasifikasi_Risiko_Gagal_Bayar.ipynb
├── Final_Report_UAS_PM_A112415645.pdf
├── README.md
└── requirements.txt
```

## Dependensi utama

Dependensi proyek didefinisikan pada `requirements.txt` dan `app/requirements.txt`, meliputi:

- `streamlit`
- `scikit-learn`
- `pandas`
- `numpy`
- `plotly`
- `joblib`
- `shap`
- `matplotlib`
- `seaborn`
- `imbalanced-learn`
- `xgboost`

## Menjalankan secara lokal

1. Install dependensi:
   ```bash
   python -m pip install -r requirements.txt
   ```

2. Generate artefak model:
   ```bash
   python app/train_model.py
   ```

3. Jalankan aplikasi Streamlit:
   ```bash
   python -m streamlit run app/app.py
   ```

4. Buka alamat lokal yang muncul di terminal, biasanya:
   ```text
   http://localhost:8501
   ```

## Artefak model

Setelah menjalankan `python app/train_model.py`, sistem akan menghasilkan artefak berikut di folder `app/models/`:

- `random_forest_model.pkl`
- `xgboost_model.pkl`
- `scaler.pkl`
- `feature_names.json`
- `salary_bins.json`
- `eval_artifacts.json`

Artefak tersebut dibutuhkan agar aplikasi dapat menampilkan prediksi, evaluasi model, dan interpretasi hasil.

## Deploy ke Streamlit Community Cloud

1. Push project ke GitHub.
2. Pastikan `Default_Fin.csv`, `Default_Fin_processed_full.csv`, dan folder `app/models/` ikut ter-commit.
3. Buka [Streamlit Community Cloud](https://share.streamlit.io/).
4. Login menggunakan akun GitHub.
5. Klik **New app**.
6. Pilih repository dan branch yang sesuai.
7. Set **Main file path** ke `app/app.py`.
8. Klik **Deploy**.

## Catatan penting

- `requirements.txt` di root membantu Streamlit Community Cloud mendeteksi dependensi proyek.
- Jika artefak model belum tersedia, aplikasi akan meminta pengguna menjalankan `python app/train_model.py` terlebih dahulu.
- Halaman **Model Demo** menggunakan model **Random Forest** sebagai jalur prediksi utama.
- Sidebar aplikasi menampilkan **Profil Pengembang** untuk keperluan dokumentasi tugas/UAS.

## Profil pengembang

- **Nama**: Dandy Prasetyo Nugroho
- **NIM**: A11.2024.15645
- **Kelas**: A11.4404
- **Program Studi**: Teknik Informatika
- **Universitas**: Universitas Dian Nuswantoro
