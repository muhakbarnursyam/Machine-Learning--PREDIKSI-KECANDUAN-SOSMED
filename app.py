import os
import pandas as pd
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

try:
    from xgboost import XGBClassifier
    xgb = True
except:
    xgb = False

# Konfigurasi Halaman (Hanya satu di paling atas)
st.set_page_config(
    page_title="Prediksi Kecanduan Media Sosial",
    page_icon="📱",
    layout="wide"
)

st.title("📱 Prediksi Tingkat Kecanduan Media Sosial")
st.write("---")

# Sidebar Menu
menu = st.sidebar.selectbox(
    "Pilih Menu",
    ["Home", "Dataset", "EDA", "Preprocessing", "Training", "Prediksi"]
)

# Membaca Dataset secara Global (Hanya ada 1 file_uploader di sini)
uploaded = st.sidebar.file_uploader("Upload Dataset", type="csv")
if uploaded is not None:
    df = pd.read_csv(uploaded)
elif os.path.exists("Dataset_Revisi_Indikator_Kecanduan.csv"):
    df = pd.read_csv("Dataset_Revisi_Indikator_Kecanduan.csv")
else:
    st.warning("Silakan upload dataset terlebih dahulu di sidebar.")
    st.stop()

# ==========================================================
# HOME
# ==========================================================
if menu == "Home":
    st.header("Machine Learning Project")
    st.image("https://images.unsplash.com/photo-1611162618071-b39a2ec055fb")
    st.write("""
    ## Prediksi Tingkat Kecanduan Media Sosial
    
    **Model yang Didukung:**
    ✔ Logistic Regression
    ✔ Decision Tree
    ✔ Random Forest
    ✔ KNN
    ✔ Naive Bayes
    ✔ SVM
    ✔ XGBoost
    """)

# ==========================================================
# DATASET
# ==========================================================
elif menu == "Dataset":
    st.header("Dataset")
    st.dataframe(df)
    st.subheader("Informasi Dimensi (Baris, Kolom)")
    st.write(df.shape)
    st.subheader("Statistik Deskriptif")
    st.write(df.describe())
    st.subheader("Tipe Data")
    st.write(df.dtypes)

# ==========================================================
# EDA
# ==========================================================
elif menu == "EDA":
    st.header("Exploratory Data Analysis")
    st.subheader("5 Data Pertama")
    st.dataframe(df.head())

    st.subheader("Informasi Dataset")
    info = pd.DataFrame({
        "Kolom": df.columns,
        "Tipe Data": df.dtypes,
        "Missing Value": df.isnull().sum(),
        "Unique": df.nunique()
    })
    st.dataframe(info)

    st.subheader("Distribusi Target")
    target = "Addiction_Level"
    if target in df.columns:
        fig, ax = plt.subplots(figsize=(6, 4))
        df[target].value_counts().plot(kind="bar", ax=ax)
        ax.set_xlabel("Level")
        ax.set_ylabel("Jumlah")
        ax.set_title("Distribusi Addiction Level")
        st.pyplot(fig)
    else:
        st.error(f"Kolom target '{target}' tidak ditemukan.")

    st.subheader("Distribusi Semua Variabel Numerik")
    numeric = df.select_dtypes(include=np.number)
    for col in numeric.columns:
        fig, ax = plt.subplots(figsize=(6, 3))
        ax.hist(numeric[col], bins=20)
        ax.set_title(col)
        st.pyplot(fig)

    st.subheader("Boxplot")
    for col in numeric.columns:
        fig, ax = plt.subplots(figsize=(6, 2.5))
        ax.boxplot(numeric[col], vert=False)
        ax.set_title(col)
        st.pyplot(fig)

    st.subheader("Correlation Matrix")
    if not numeric.empty:
        corr = numeric.corr()
        fig, ax = plt.subplots(figsize=(10, 8))
        im = ax.imshow(corr)
        ax.set_xticks(range(len(corr.columns)))
        ax.set_xticklabels(corr.columns, rotation=90)
        ax.set_yticks(range(len(corr.columns)))
        ax.set_yticklabels(corr.columns)
        plt.colorbar(im)
        st.pyplot(fig)

# ==========================================================
# PREPROCESSING
# ==========================================================
elif menu == "Preprocessing":
    st.header("Preprocessing Data")
    data = df.copy()
    
    st.subheader("Ukuran Awal Dataset")
    st.write(data.shape)

    data = data.drop_duplicates()
    st.success("Duplicate berhasil dihapus")

    drop_cols = ["Student_ID", "Addicted_Score", "Sleep_Addiction_Indicator", "Physical_Activity_Indicator"]
    drop_cols = [c for c in drop_cols if c in data.columns]
    data = data.drop(columns=drop_cols)

    target = "Addiction_Level"
    X = data.drop(columns=[target])
    y = data[target]

    encoder = LabelEncoder()
    y = encoder.fit_transform(y)

    kategori = X.select_dtypes(include="object").columns
    for col in kategori:
        le = LabelEncoder()
        X[col] = le.fit_transform(X[col].astype(str))

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42, stratify=y
    )

    st.success("Preprocessing selesai")
    st.write(f"Jumlah Data Train: {X_train.shape}")
    st.write(f"Jumlah Data Test: {X_test.shape}")
    st.write("Kelas Target yang terdeteksi:", encoder.classes_)

# ==========================================================
# TRAINING MODEL
# ==========================================================
elif menu == "Training":
    st.header("Training Machine Learning")
    data = df.copy()

    drop_cols = ["Student_ID", "Addicted_Score", "Sleep_Addiction_Indicator", "Physical_Activity_Indicator"]
    drop_cols = [c for c in drop_cols if c in data.columns]
    data = data.drop(columns=drop_cols)

    target = "Addiction_Level"
    X = data.drop(columns=[target])
    y = data[target]

    encoder = LabelEncoder()
    y = encoder.fit_transform(y)

    kategori = X.select_dtypes(include="object").columns
    for col in kategori:
        le = LabelEncoder()
        X[col] = le.fit_transform(X[col].astype(str))

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.20, random_state=42, stratify=y
    )

    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000),
        "Decision Tree": DecisionTreeClassifier(random_state=42),
        "Random Forest": RandomForestClassifier(n_estimators=200, random_state=42),
        "KNN": KNeighborsClassifier(),
        "Naive Bayes": GaussianNB(),
        "SVM": SVC(probability=True)
    }

    if xgb:
        models["XGBoost"] = XGBClassifier(eval_metric="mlogloss", random_state=42)

    hasil = []
    trained_models = {}
    progress = st.progress(0)
    total = len(models)

    for i, (nama, model) in enumerate(models.items()):
        model.fit(X_train, y_train)
        pred = model.predict(X_test)
        
        acc = accuracy_score(y_test, pred)
        pre = precision_score(y_test, pred, average="weighted")
        rec = recall_score(y_test, pred, average="weighted")
        f1 = f1_score(y_test, pred, average="weighted")

        hasil.append([nama, acc, pre, rec, f1])
        trained_models[nama] = model
        progress.progress((i + 1) / total)

    hasil_df = pd.DataFrame(hasil, columns=["Model", "Accuracy", "Precision", "Recall", "F1 Score"])
    hasil_df = hasil_df.sort_values(by="Accuracy", ascending=False)

    st.success("Training selesai!")
    st.dataframe(hasil_df)

    st.subheader("Model Terbaik")
    st.write(hasil_df.iloc[0])

    st.subheader("Grafik Accuracy")
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(hasil_df["Model"], hasil_df["Accuracy"])
    plt.xticks(rotation=20)
    st.pyplot(fig)

    # Simpan Model dan Scaler
    joblib.dump(trained_models, "Semua_Model.pkl")
    joblib.dump(scaler, "Scaler.pkl")
    st.success("Model dan Scaler berhasil disimpan ke sistem!")

# ==========================================================
# MENU PREDIKSI (VERSI BAHASA INDONESIA)
# ==========================================================
elif menu == "Prediksi":
    st.header("🔍 UJI TINGKAT KECANDUAN")

    # Load Model dan Scaler
    if os.path.exists("Semua_Model.pkl") and os.path.exists("Scaler.pkl"):
        models = joblib.load("Semua_Model.pkl")
        scaler = joblib.load("Scaler.pkl")
    else:
        st.error("Model belum tersedia. Silakan jalankan menu **Training** terlebih dahulu agar model tercipta.")
        st.stop()

    model_name = st.selectbox("Pilih Model untuk Prediksi", list(models.keys()))
    model = models[model_name]

    st.subheader("Masukkan Data Pengguna")
    
    # Input Form Elemen Berbahasa Indonesia
    age = st.number_input("Usia (Tahun)", min_value=10, max_value=60, value=20)
    gender = st.selectbox("Jenis Kelamin", ["Laki-laki", "Perempuan"])
    academic = st.selectbox("Tingkat Pendidikan", ["SMA / Sederajat", "Diploma (D3/D4)", "Sarjana (S1/S2/S3)"])
    country = st.selectbox("Negara Asal", ["Indonesia", "India", "Amerika Serikat", "Inggris", "Kanada", "Australia"])
    usage = st.slider("Rata-rata Penggunaan Harian (Jam)", 0.0, 15.0, 5.0, 0.5)
    platform = st.selectbox("Aplikasi / Platform Paling Sering Digunakan", ["Instagram", "TikTok", "Facebook", "YouTube", "Twitter"])
    academic_effect = st.selectbox("Apakah Mempengaruhi Performa Akademik/Kerja?", ["Ya", "Tidak"])
    sleep = st.slider("Durasi Tidur Per Malam (Jam)", 3.0, 12.0, 7.0, 0.5)
    mental = st.slider("Skor Kesehatan Mental (1: Sangat Buruk - 10: Sangat Baik)", 1, 10, 5)
    physical = st.slider("Aktivitas Fisik / Olahraga (Jam per Minggu)", 0.0, 5.0, 1.0, 0.5)

    if st.button("🔍 Mulai Prediksi Kecanduan", key="prediksi_button"):
        # Mapping dari pilihan Bahasa Indonesia ke nilai numerik sesuai model training
        gender_val = 1 if gender == "Laki-laki" else 0
        academic_map = {"SMA / Sederajat": 0, "Diploma (D3/D4)": 1, "Sarjana (S1/S2/S3)": 2}
        country_map = {"Indonesia": 0, "India": 1, "Amerika Serikat": 2, "Inggris": 3, "Kanada": 4, "Australia": 5}
        platform_map = {"Instagram": 0, "TikTok": 1, "Facebook": 2, "YouTube": 3, "Twitter": 4}
        academic_effect_val = 1 if academic_effect == "Ya" else 0

        # DataFrame Input Data untuk Scaler
        input_data = pd.DataFrame([{
            "Age": age,
            "Gender": gender_val,
            "Academic_Level": academic_map[academic],
            "Country": country_map[country],
            "Avg_Daily_Usage_Hours": usage,
            "Most_Used_Platform": platform_map[platform],
            "Affects_Academic_Performance": academic_effect_val,
            "Sleep_Hours_Per_Night": sleep,
            "Mental_Health_Score": mental,
            "Physical_Activity": physical
        }])

        # Transform menggunakan scaler lama
        input_scaled = scaler.transform(input_data)

        # Proses Prediksi
        hasil = model.predict(input_scaled)[0]

        # Konversi ke Label String hasil prediksi
        if hasil == 0:
            tingkat = "🟢 RENDAH"
        elif hasil == 1:
            tingkat = "🟡 SEDANG"
        else:
            tingkat = "🔴 TINGGI"

        st.success("Prediksi Berhasil!")
        
        # Output st.metric()
        st.metric(label="Hasil Analisis: Tingkat Kecanduan", value=tingkat)

        # Menghitung dan Menampilkan Probabilitas jika model mendukung
        if hasattr(model, "predict_proba"):
            prob = model.predict_proba(input_scaled)[0]
            
            hasil_prob = pd.DataFrame({
                "Kategori": ["Rendah", "Sedang", "Tinggi"],
                "Probabilitas (%)": [
                    round(prob[0] * 100, 2),
                    round(prob[1] * 100, 2),
                    round(prob[2] * 100, 2)
                ]
            })

            st.subheader("📊 Probabilitas Prediksi")
            st.dataframe(hasil_prob)

            # Grafik Probabilitas
            fig, ax = plt.subplots(figsize=(6, 4))
            ax.bar(hasil_prob["Kategori"], hasil_prob["Probabilitas (%)"], color=['green', 'orange', 'red'])
            ax.set_ylabel("Persentase (%)")
            ax.set_title("Grafik Persentase Tingkat Kecanduan")
            st.pyplot(fig)