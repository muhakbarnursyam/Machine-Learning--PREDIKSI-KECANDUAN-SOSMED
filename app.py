import io
import os
import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC

# Cek ketersediaan XGBoost
try:
    from xgboost import XGBClassifier
    xgb = True
except ImportError:
    xgb = False

# ==========================================================
# KONFIGURASI HALAMAN & TEMA MODERN BIRU PUTIH
# ==========================================================
st.set_page_config(
    page_title="Prediksi Kecanduan Media Sosial",
    page_icon="📊",
    layout="wide"
)

# Custom CSS Inject untuk Tema Modern Blue & White
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700&display=swap');

    /* Global Theme Background & Typography */
    .stApp {
        background-color: #f4f7fb;
        color: #1e293b;
        font-family: 'Inter', sans-serif;
    }

    /* Headings */
    h1, h2, h3, h4 {
        font-family: 'Inter', sans-serif !important;
        color: #0f172a !important;
        font-weight: 700 !important;
        letter-spacing: -0.5px;
    }

    /* Sidebar Styling */
    [data-testid="stSidebar"] {
        background-color: #ffffff !important;
        border-right: 1px solid #e2e8f0;
    }

    /* Cards & Container */
    div.stDataFrame, div.stForm, .stAlert {
        background: #ffffff !important;
        border: 1px solid #cbd5e1 !important;
        border-radius: 10px !important;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.05);
    }

    /* Custom Buttons */
    .stButton>button, div[data-testid="stFormSubmitButton"]>button, .stDownloadButton>button {
        font-family: 'Inter', sans-serif !important;
        font-size: 14px !important;
        font-weight: 600 !important;
        color: #ffffff !important;
        background-color: #2563eb !important;
        border: None !important;
        border-radius: 8px !important;
        padding: 10px 24px !important;
        transition: all 0.2s ease;
    }

    .stButton>button:hover, div[data-testid="stFormSubmitButton"]>button:hover, .stDownloadButton>button:hover {
        background-color: #1d4ed8 !important;
        box-shadow: 0 4px 12px rgba(37, 99, 235, 0.3);
    }

    /* Metric Boxes */
    [data-testid="stMetricValue"] {
        font-family: 'Inter', sans-serif !important;
        color: #2563eb !important;
        font-weight: 700 !important;
    }
    
    [data-testid="stMetricLabel"] {
        font-family: 'Inter', sans-serif !important;
        color: #64748b !important;
        font-size: 14px !important;
    }

    /* Form Inputs & Selectbox */
    div[data-baseweb="select"] > div, input {
        background-color: #ffffff !important;
        border-radius: 6px !important;
        border: 1px solid #cbd5e1 !important;
        color: #0f172a !important;
    }

    /* Divider */
    hr {
        border-top: 1px solid #e2e8f0 !important;
    }
    </style>
""", unsafe_allow_html=True)

# Header Utama
st.title("📊 Analisis & Prediksi Kecanduan Media Sosial")
st.caption("Platform Analytics Machine Learning Berbasis Data Sosiologis")
st.write("---")

# ==========================================================
# SIDEBAR MENU
# ==========================================================
menu = st.sidebar.selectbox(
    "📌 MENU UTAMA",
    [
        "Home",
        "Dataset",
        "EDA (Eksplorasi Data)",
        "Preprocessing",
        "Training Model",
        "Prediksi Manual",
        "Prediksi Dataset Upload"
    ]
)

# ==========================================================
# DATASET TRAINING
# ==========================================================
TRAINING_DATASET = "Dataset_Revisi_Indikator_Kecanduan.csv"

# ==========================================================
# FUNGSI PREPROCESSING & TRAINING
# ==========================================================
def prepare_training_data(data):
    """Membersihkan data, melakukan encoding, dan scaling."""
    data = data.copy().drop_duplicates()

    drop_cols = ["Student_ID", "Addicted_Score", "Sleep_Addiction_Indicator", "Physical_Activity_Indicator"]
    drop_cols = [col for col in drop_cols if col in data.columns]
    data = data.drop(columns=drop_cols)

    target = "Addiction_Level"
    if target not in data.columns:
        st.error("Kolom target 'Addiction_Level' tidak ditemukan pada dataset.")
        st.stop()

    X = data.drop(columns=[target])
    y = data[target]

    target_encoder = LabelEncoder()
    y_encoded = target_encoder.fit_transform(y.astype(str))

    feature_encoders = {}
    categorical_columns = X.select_dtypes(include="object").columns
    for col in categorical_columns:
        encoder = LabelEncoder()
        X[col] = encoder.fit_transform(X[col].astype(str))
        feature_encoders[col] = encoder

    feature_columns = list(X.columns)
    scaler = StandardScaler()
    
    return X, y_encoded, target_encoder, feature_encoders, scaler, feature_columns

# ==========================================================
# TAMPILAN MENU
# ==========================================================

# --- HOME ---
if menu == "Home":
    st.header("🏠 Dashboard Utama")
    
    col_info, col_feat = st.columns([2, 1])
    with col_info:
        st.write(
            """
            Selamat datang di **Aplikasi Klasifikasi & Prediksi Kecanduan Media Sosial**. 
            Sistem ini memanfaatkan berbagai algoritma Machine Learning untuk memprediksi tingkat kecanduan 
            pengguna media sosial berdasarkan indikator-indikator perilaku, fisik, dan pola pemakaian harian.
            """
        )
        
        st.subheader("🤖 Algoritma yang Didukung")
        st.markdown("""
        - **Logistic Regression**: Pemodelan linier dasar untuk klasifikasi probabilistik.
        - **Decision Tree & Random Forest**: Model berbasis pohon keputusan bernilai presisi tinggi.
        - **K-Nearest Neighbors (KNN)**: Pengelompokan jarak terdekat antar sampel data.
        - **Naive Bayes**: Klasifikasi berbasis Teorema Bayes.
        - **Support Vector Machine (SVM)**: Pemisah marjin terbaik untuk ruang berdimensi tinggi.
        - **XGBoost**: Extreme Gradient Boosting untuk performa klasifikasi optimal.
        """)

    with col_feat:
        st.info("💡 **Alur Penggunaan:**\n1. Pelajari data di **Dataset** & **EDA**.\n2. Jalankan **Preprocessing**.\n3. Latih model di **Training Model**.\n4. Lakukan pengujian di **Prediksi Manual**.")

# --- DATASET ---
elif menu == "Dataset":
    st.header("📊 Overview Dataset")
    
    if not os.path.exists(TRAINING_DATASET):
        st.error(f"File dataset '{TRAINING_DATASET}' tidak ditemukan di direktori kerja.")
        st.stop()

    df = pd.read_csv(TRAINING_DATASET)
    st.success(f"Dataset Aktif: `{TRAINING_DATASET}`")
    st.dataframe(df, use_container_width=True)

    c1, c2 = st.columns(2)
    c1.metric("Total Baris (Data)", df.shape[0])
    c2.metric("Total Kolom (Fitur)", df.shape[1])
    
    st.subheader("Ringkasan Statistik")
    st.write(df.describe(include="all"))

# --- EDA ---
elif menu == "EDA (Eksplorasi Data)":
    st.header("🔍 Exploratory Data Analysis (EDA)")
    
    if not os.path.exists(TRAINING_DATASET):
        st.error(f"File dataset '{TRAINING_DATASET}' tidak ditemukan.")
        st.stop()

    df = pd.read_csv(TRAINING_DATASET)
    st.subheader("Pratinjau Data (Top 5 Baris)")
    st.dataframe(df.head())

    # Styling matplotlib biru-putih modern
    plt.style.use('default')

    if "Addiction_Level" in df.columns:
        st.subheader("Distribusi Kelas Target (Addiction Level)")
        fig, ax = plt.subplots(figsize=(7, 3.5))
        fig.patch.set_facecolor('#ffffff')
        ax.set_facecolor('#f8fafc')
        
        df["Addiction_Level"].value_counts().plot(
            kind="bar", 
            ax=ax, 
            color='#2563eb', 
            edgecolor='#1d4ed8', 
            linewidth=1.2
        )
        ax.set_xlabel("Tingkat Kecanduan", color='#0f172a', fontweight='bold')
        ax.set_ylabel("Jumlah", color='#0f172a', fontweight='bold')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        st.pyplot(fig)

    numeric = df.select_dtypes(include=np.number)
    if not numeric.empty:
        st.subheader("Matriks Korelasi Variabel Numerik")
        corr = numeric.corr()
        fig, ax = plt.subplots(figsize=(8, 5))
        fig.patch.set_facecolor('#ffffff')
        
        im = ax.imshow(corr, cmap='Blues')
        ax.set_xticks(range(len(corr.columns)))
        ax.set_xticklabels(corr.columns, rotation=45, ha='right', color='#0f172a')
        ax.set_yticks(range(len(corr.columns)))
        ax.set_yticklabels(corr.columns, color='#0f172a')
        plt.colorbar(im)
        st.pyplot(fig)

# --- PREPROCESSING ---
elif menu == "Preprocessing":
    st.header("⚙️ Preprocessing & Transformasi Data")
    if not os.path.exists(TRAINING_DATASET):
        st.error(f"File dataset '{TRAINING_DATASET}' tidak ditemukan.")
        st.stop()

    df = pd.read_csv(TRAINING_DATASET)
    X, y, target_encoder, feature_encoders, scaler, feature_columns = prepare_training_data(df)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    st.success("Preprocessing data berhasil dijalankan!")
    
    col1, col2 = st.columns(2)
    with col1:
        st.info(f"**Data Training (80%):** {X_train.shape[0]} baris")
    with col2:
        st.info(f"**Data Testing (20%):** {X_test.shape[0]} baris")

    st.write("Label Kelas Target Teridentifikasi:")
    st.json(list(target_encoder.classes_))

# --- TRAINING MODEL ---
elif menu == "Training Model":
    st.header("🏋️ Pelatihan Model Machine Learning")
    if not os.path.exists(TRAINING_DATASET):
        st.error(f"File dataset '{TRAINING_DATASET}' tidak ditemukan.")
        st.stop()

    df = pd.read_csv(TRAINING_DATASET)
    X, y, target_encoder, feature_encoders, scaler, feature_columns = prepare_training_data(df)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

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
        model.fit(X_train_scaled, y_train)
        pred = model.predict(X_test_scaled)
        
        acc = accuracy_score(y_test, pred)
        pre = precision_score(y_test, pred, average="weighted", zero_division=0)
        rec = recall_score(y_test, pred, average="weighted", zero_division=0)
        f1 = f1_score(y_test, pred, average="weighted", zero_division=0)
        
        hasil.append([nama, acc, pre, rec, f1])
        trained_models[nama] = model
        progress.progress((i + 1) / total)

    hasil_df = pd.DataFrame(hasil, columns=["Model", "Accuracy", "Precision", "Recall", "F1 Score"]).sort_values(by="Accuracy", ascending=False)

    st.success("Proses Training Selesai!")
    st.subheader("Perbandingan Performa Model")
    st.dataframe(hasil_df.style.format({
        "Accuracy": "{:.2%}", "Precision": "{:.2%}", "Recall": "{:.2%}", "F1 Score": "{:.2%}"
    }), use_container_width=True)

    # Simpan model & pendukung
    joblib.dump(trained_models, "Semua_Model.pkl")
    joblib.dump(scaler, "Scaler.pkl")
    joblib.dump(target_encoder, "Target_Encoder.pkl")
    joblib.dump(feature_encoders, "Feature_Encoders.pkl")
    joblib.dump(feature_columns, "Feature_Columns.pkl")
    st.info("Artefak model (.pkl) berhasil disimpan secara lokal.")

# --- PREDIKSI MANUAL ---
elif menu == "Prediksi Manual":
    st.header("🎯 Simulasi Prediksi Individual")
    
    required_files = ["Semua_Model.pkl", "Scaler.pkl", "Target_Encoder.pkl", "Feature_Encoders.pkl", "Feature_Columns.pkl"]
    if not all(os.path.exists(f) for f in required_files):
        st.warning("Silakan jalankan menu 'Training Model' terlebih dahulu untuk menggenerasi berkas model.")
        st.stop()

    models = joblib.load("Semua_Model.pkl")
    scaler = joblib.load("Scaler.pkl")
    target_encoder = joblib.load("Target_Encoder.pkl")
    feature_encoders = joblib.load("Feature_Encoders.pkl")
    feature_columns = joblib.load("Feature_Columns.pkl")

    model_name = st.selectbox("Pilih Algoritma Model:", list(models.keys()))
    model = models[model_name]

    st.subheader("Form Input Parameter:")
    ui_inputs = {}
    with st.form("form_prediksi_manual"):
        col1, col2 = st.columns(2)
        
        for idx, col_name in enumerate(feature_columns):
            form_col = col1 if idx % 2 == 0 else col2
            if col_name in feature_encoders:
                labels_kategori = list(feature_encoders[col_name].classes_)
                ui_inputs[col_name] = form_col.selectbox(col_name, options=labels_kategori, key=f"ui_{col_name}")
            else:
                ui_inputs[col_name] = form_col.number_input(col_name, value=0, step=1, key=f"ui_{col_name}")
        
        submitted = st.form_submit_button("Hitung Prediksi")

    if submitted:
        input_df = pd.DataFrame([ui_inputs])
        for col, encoder in feature_encoders.items():
            input_df[col] = encoder.transform([input_df.at[0, col]])
        
        input_scaled = scaler.transform(input_df[feature_columns])
        prediksi_angka = model.predict(input_scaled)
        hasil_label = target_encoder.inverse_transform(prediksi_angka)[0]
        
        st.success(f"Hasil Klasifikasi Algoritma **{model_name}**: Tingkat Kecanduan **{hasil_label}**")

# --- PREDIKSI DATASET UPLOAD ---
elif menu == "Prediksi Dataset Upload":
    st.header("📁 Prediksi Massal via Unggah File CSV")
    uploaded_file = st.file_uploader("Unggah berkas CSV data uji baru", type=["csv"])
    
    if uploaded_file is not None:
        user_data = pd.read_csv(uploaded_file)
        st.subheader("Pratinjau Data yang Diunggah:")
        st.dataframe(user_data.head(), use_container_width=True)
