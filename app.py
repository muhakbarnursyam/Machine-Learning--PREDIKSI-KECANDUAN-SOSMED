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
# KONFIGURASI HALAMAN & TEMA RETRO PIXEL SPACE (SPACETOON STYLE)
# ==========================================================
st.set_page_config(
    page_title="Prediksi Kecanduan Media Sosial - Space Pixel Edition",
    page_icon="https://img.icons8.com/pixel-surveillance/64/000000/space-ship.png",
    layout="wide"
)

# Custom CSS Inject untuk Tema Pixel Space Cartoon
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Press+Start+2P&family=VT323&display=swap');

    /* Global Theme - Dark Space Background */
    .stApp {
        background-color: #0b0e14;
        background-image: 
            radial-gradient(white, rgba(255,255,255,.2) 2px, transparent 40px),
            radial-gradient(white, rgba(255,255,255,.15) 1px, transparent 30px),
            radial-gradient(white, rgba(255,255,255,.1) 2px, transparent 20px);
        background-size: 550px 550px, 350px 350px, 250px 250px;
        background-position: 0 0, 40px 60px, 130px 270px;
        color: #e0e6ed;
        font-family: 'VT323', monospace;
        font-size: 20px;
    }

    /* Headings Style */
    h1, h2, h3, h4 {
        font-family: 'Press Start 2P', cursive !important;
        color: #ffcc00 !important;
        text-shadow: 3px 3px 0px #ff0055, 6px 6px 0px #000000;
        letter-spacing: 1px;
    }

    /* Sidebar Pixel Styling */
    [data-testid="stSidebar"] {
        background-color: #121824 !important;
        border-right: 4px solid #ff0055;
    }
    
    [data-testid="stSidebar"] * {
        font-family: 'VT323', monospace !important;
        font-size: 22px !important;
        color: #00ffcc !important;
    }

    /* Pixel Art Container Cards */
    div.stDataFrame, div.stForm, .stAlert {
        background: #182030 !important;
        border: 4px solid #00ffcc !important;
        box-shadow: 6px 6px 0px #ff0055;
        border-radius: 0px !important;
    }

    /* Retro Pixel Buttons */
    .stButton>button, div[data-testid="stFormSubmitButton"]>button {
        font-family: 'Press Start 2P', cursive !important;
        font-size: 12px !important;
        color: #ffffff !important;
        background-color: #ff0055 !important;
        border: 3px solid #ffffff !important;
        box-shadow: 4px 4px 0px #000000;
        border-radius: 0px !important;
        padding: 10px 20px !important;
        transition: all 0.1s ease-in-out;
    }

    .stButton>button:hover, div[data-testid="stFormSubmitButton"]>button:hover {
        background-color: #00ffcc !important;
        color: #000000 !important;
        transform: translate(-2px, -2px);
        box-shadow: 6px 6px 0px #000000;
    }

    /* Selectbox & Input Customization */
    div[data-baseweb="select"] > div, input {
        background-color: #0b0e14 !important;
        color: #00ffcc !important;
        border: 2px solid #00ffcc !important;
        border-radius: 0px !important;
        font-family: 'VT323', monospace !important;
        font-size: 20px !important;
    }

    /* Metric Boxes Pixel Style */
    [data-testid="stMetricValue"] {
        font-family: 'Press Start 2P', cursive !important;
        color: #00ffcc !important;
    }
    
    [data-testid="stMetricLabel"] {
        font-family: 'VT323', monospace !important;
        font-size: 22px !important;
        color: #ffcc00 !important;
    }

    /* Tabs Retro Styling */
    button[data-baseweb="tab"] {
        font-family: 'Press Start 2P', cursive !important;
        font-size: 10px !important;
        color: #ffffff !important;
        background-color: #182030 !important;
        border: 2px solid #00ffcc !important;
        border-radius: 0px !important;
    }

    button[aria-selected="true"] {
        background-color: #ff0055 !important;
        color: #ffffff !important;
    }

    /* Line separator */
    hr {
        border-top: 4px dashed #00ffcc !important;
    }
    </style>
""", unsafe_allow_dict_style=True)

# Header Utama
st.title("PREDIKSI KECANDUAN MEDIA SOSIAL")
st.caption("STASIUN CONTROL PIXEL SPACE - PLANET DATA")
st.write("---")

# ==========================================================
# SIDEBAR MENU
# ==========================================================
menu = st.sidebar.selectbox(
    "MENU NAVIGASI UTAMA",
    [
        "Home",
        "Dataset",
        "EDA",
        "Preprocessing",
        "Training",
        "Prediksi Manual",
        "Prediksi Dataset Upload"
    ]
)

# ==========================================================
# DATASET TRAINING
# ==========================================================
TRAINING_DATASET = "Dataset_Revisi_Indikator_Kecanduan.csv"

if not os.path.exists(TRAINING_DATASET):
    st.error(f"Berkas dataset training '{TRAINING_DATASET}' tidak ditemukan di stasiun lokal.")
    st.stop()

df = pd.read_csv(TRAINING_DATASET)

# ==========================================================
# FUNGSI PREPROCESSING & TRAINING
# ==========================================================
def prepare_training_data(data):
    """Membersihkan data, melakukan encoding, dan scaling."""
    data = data.copy().drop_duplicates()

    # Hapus kolom yang tidak digunakan
    drop_cols = ["Student_ID", "Addicted_Score", "Sleep_Addiction_Indicator", "Physical_Activity_Indicator"]
    drop_cols = [col for col in drop_cols if col in data.columns]
    data = data.drop(columns=drop_cols)

    # Validasi target
    target = "Addiction_Level"
    if target not in data.columns:
        st.error("Kolom target 'Addiction_Level' tidak ditemukan pada dataset.")
        st.stop()

    X = data.drop(columns=[target])
    y = data[target]

    # Encode Target
    target_encoder = LabelEncoder()
    y_encoded = target_encoder.fit_transform(y.astype(str))

    # Encode Fitur Kategorikal
    feature_encoders = {}
    categorical_columns = X.select_dtypes(include="object").columns
    for col in categorical_columns:
        encoder = LabelEncoder()
        X[col] = encoder.fit_transform(X[col].astype(str))
        feature_encoders[col] = encoder

    feature_columns = list(X.columns)
    
    # Inisialisasi Scaler
    scaler = StandardScaler()
    
    return X, y_encoded, target_encoder, feature_encoders, scaler, feature_columns

# ==========================================================
# TAMPILAN MENU
# ==========================================================

# --- HOME ---
if menu == "Home":
    st.header("PROYEK MACHINE LEARNING LANTARAN ANTARIKSA")
    
    # Gambar Cartoon Pixel World Space (Online URL)
    st.image(
        "https://images.unsplash.com/photo-1550684848-fac1c5b4e853?q=80&w=1000&auto=format&fit=crop", 
        caption="Stasiun Pengamatan Sinyal Digital Galaksi",
        use_container_width=True
    )
    
    st.write(
        """
        ### MODUL PREDIKSI TINGKAT KECANDUAN MEDIA SOSIAL
        Sistem ini menganalisis pola konsumsi data digital dan tingkat kecanduan media sosial pada populasi humanoid menggunakan algoritma Machine Learning.

        ### ALGORITMA DUKUNGAN SISTEM:
        - Logistic Regression
        - Decision Tree
        - Random Forest
        - KNN (K-Nearest Neighbors)
        - Naive Bayes
        - SVM (Support Vector Machine)
        - XGBoost

        ### NAVIGASI MODUL:
        - DATASET : Inspeksi data mentah galaksi.
        - EDA : Analisis visual dan sebaran parameter.
        - PREPROCESSING : Transformasi sinyal dan pembagian data.
        - TRAINING : Komputasi simulasi model.
        - PREDIKSI MANUAL : Simulasi kalkulasi individu.
        - PREDIKSI DATASET UPLOAD : Pemrosesan berkas massal.
        """
    )

# --- DATASET ---
elif menu == "Dataset":
    st.header("DATASET TRAINING GALAKSI")
    st.info(f"Sumber Berkas Data: {TRAINING_DATASET}")
    
    st.write("""
    Tabel berikut menampilkan seluruh parameter responden, meliputi demografi, kebiasaan akses perangkat, hingga tingkat kecanduan media sosial.
    """)

    st.dataframe(df, use_container_width=True)

    col1, col2, col3 = st.columns(3)
    col1.metric("JUMLAH SAMPEL", df.shape[0])
    col2.metric("JUMLAH ATRIBUT", df.shape[1])
    
    st.subheader("STATISTIK DESKRIPTIF")
    st.write(df.describe(include="all"))

    st.subheader("TIPE DATA PARAMETER")
    st.write(df.dtypes.astype(str))

# --- EDA ---
elif menu == "EDA":
    st.header("EXPLORATORY DATA ANALYSIS (EDA)")
    
    st.subheader("5 SAMPEL PERTAMA")
    st.dataframe(df.head())

    st.subheader("STRUKTUR DAN KUALITAS DATA")
    info = pd.DataFrame({
        "Kolom": df.columns,
        "Tipe Data": df.dtypes.astype(str),
        "Missing Value": df.isnull().sum(),
        "Nilai Unik": df.nunique()
    })
    st.dataframe(info, use_container_width=True)

    # Style Plot Matplotlib Tema Dark Retro Pixel
    plt.style.use('dark_background')

    if "Addiction_Level" in df.columns:
        st.subheader("DISTRIBUSI ADDICTION LEVEL")
        fig, ax = plt.subplots(figsize=(7, 3.5))
        fig.patch.set_facecolor('#182030')
        ax.set_facecolor('#0b0e14')
        
        df["Addiction_Level"].value_counts().plot(kind="bar", ax=ax, color='#00ffcc', edgecolor='#ff0055', linewidth=2)
        ax.set_xlabel("Tingkat Kecanduan", color='#ffcc00')
        ax.set_ylabel("Jumlah", color='#ffcc00')
        st.pyplot(fig)

    numeric = df.select_dtypes(include=np.number)
    if not numeric.empty:
        st.subheader("SEBARAN VARIABEL NUMERIK")
        for col in numeric.columns:
            fig, ax = plt.subplots(figsize=(6, 2.5))
            fig.patch.set_facecolor('#182030')
            ax.set_facecolor('#0b0e14')
            
            ax.hist(numeric[col], bins=15, color='#ff0055', edgecolor='#00ffcc', linewidth=1.5)
            ax.set_title(f"Distribusi {col}", color='#ffcc00')
            st.pyplot(fig)

        st.subheader("MATRIKS KORELASI PARAMETER")
        corr = numeric.corr()
        fig, ax = plt.subplots(figsize=(8, 5))
        fig.patch.set_facecolor('#182030')
        ax.set_facecolor('#0b0e14')
        
        im = ax.imshow(corr, cmap='magma')
        ax.set_xticks(range(len(corr.columns)))
        ax.set_xticklabels(corr.columns, rotation=90, color='#00ffcc')
        ax.set_yticks(range(len(corr.columns)))
        ax.set_yticklabels(corr.columns, color='#00ffcc')
        plt.colorbar(im)
        st.pyplot(fig)

# --- PREPROCESSING ---
elif menu == "Preprocessing":
    st.header("PREPROCESSING DATA DIGITAL")
    st.write(f"Ukuran Data Awal: **{df.shape[0]} baris, {df.shape[1]} kolom**")

    X, y, target_encoder, feature_encoders, scaler, feature_columns = prepare_training_data(df)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    st.success("Proses Preprocessing dan Transformasi Sinyal Berhasil.")

    st.write("""
    Data telah dibagi menggunakan metode Stratified Train-Test Split (Rasio 80:20):
    - Data Training (80%) : Pelatihan model.
    - Data Testing (20%) : Pengujian performa.
    """)

    st.write(f"- Data Training: {X_train.shape[0]} sampel")
    st.write(f"- Data Testing: {X_test.shape[0]} sampel")
    st.write("Kelas Target terdeteksi:", list(target_encoder.classes_))

# --- TRAINING ---
elif menu == "Training":
    st.header("TRAINING MODEL KOMPUTASI")
    
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

    hasil_df = pd.DataFrame(hasil, columns=["Model", "Accuracy", "Precision", "Recall", "F1 Score"])
    hasil_df = hasil_df.sort_values(by="Accuracy", ascending=False)

    st.success("Proses Komputasi Training Selesai.")

    st.subheader("EVALUASI PERFORMA MODEL")
    st.dataframe(hasil_df.style.format({
        "Accuracy": "{:.2%}", "Precision": "{:.2%}", "Recall": "{:.2%}", "F1 Score": "{:.2%}"
    }), use_container_width=True)

    best_model_name = hasil_df.iloc[0]["Model"]
    st.subheader(f"MODEL TERBAIK: {best_model_name}")

    # Simpan Artefak Model
    joblib.dump(trained_models, "Semua_Model.pkl")
    joblib.dump(scaler, "Scaler.pkl")
    joblib.dump(target_encoder, "Target_Encoder.pkl")
    joblib.dump(feature_encoders, "Feature_Encoders.pkl")
    joblib.dump(feature_columns, "Feature_Columns.pkl")
    st.success("Artefak model berhasil disimpan dalam memori stasiun.")

    # Visualisasi Matplotlib
    plt.style.use('dark_background')
    fig, ax = plt.subplots(figsize=(8, 3.5))
    fig.patch.set_facecolor('#182030')
    ax.set_facecolor('#0b0e14')
    
    ax.bar(hasil_df["Model"], hasil_df["Accuracy"], color='#ff0055', edgecolor='#00ffcc', linewidth=2)
    ax.set_ylabel("Accuracy Score", color='#ffcc00')
    plt.xticks(rotation=15, color='#00ffcc')
    st.pyplot(fig)

# ==========================================================
# PREDIKSI MANUAL
# ==========================================================
elif menu == "Prediksi Manual":
    st.header("SIMULASI PREDIKSI INDIVIDU & BERKAS")
    
    required_files = ["Semua_Model.pkl", "Scaler.pkl", "Target_Encoder.pkl", "Feature_Encoders.pkl", "Feature_Columns.pkl"]
    if not all(os.path.exists(file) for file in required_files):
        st.error("Silakan jalankan menu Training terlebih dahulu agar komponen siap digunakan.")
        st.stop()

    models = joblib.load("Semua_Model.pkl")
    scaler = joblib.load("Scaler.pkl")
    target_encoder = joblib.load("Target_Encoder.pkl")
    feature_encoders = joblib.load("Feature_Encoders.pkl")
    feature_columns = joblib.load("Feature_Columns.pkl")

    model_name = st.selectbox("PILIH ALGORITMA PREDIKSI", list(models.keys()))
    model = models[model_name]

    st.write("---")
    
    tab_form, tab_upload = st.tabs(["INPUT FORM MANDIRI", "UPLOAD BERKAS CSV"])

    # TAB 1: FORM
    with tab_form:
        st.subheader("INPUT DATA INDIVIDU:")
        input_data = {}
        
        label_mapping = {
            "Age": "Masukkan Usia / Umur",
            "Gender": "Pilih Jenis Kelamin",
            "Academic_Level": "Pilih Jenjang Education",
            "Country": "Pilih Negara Asal",
            "Avg_Daily_Usage_Hours": "Rata-rata Penggunaan Ponsel (Jam)",
            "Most_Used_Platform": "Platform Utama",
            "Affects_Academic_Performance": "Dampak Pada Performa Akademik?",
            "Sleep_Hours_Per_Night": "Durasi Tidur per Malam (Jam)",
            "Mental_Health_Score": "Skor Kesehatan Mental (1-10)",
            "Physical_Activity": "Tingkat Aktivitas Fisik"
        }

        option_mapping = {
            "Male": "Laki-laki",
            "Female": "Perempuan",
            "Undergraduate": "Sarjana (S1)",
            "Graduate": "Pascasarjana (S2/S3)",
            "High School": "SMA / Sederajat",
            "School": "Sekolah Dasar/Menengah",
            "Yes": "Ya",
            "No": "Tidak",
            "Low": "Rendah",
            "Medium": "Sedang",
            "High": "Tinggi"
        }

        reverse_option_mapping = {v: k for k, v in option_mapping.items()}
        
        with st.form("form_prediksi_manual"):
            col1, col2 = st.columns(2)
            ui_inputs = {}
            
            for idx, col_name in enumerate(feature_columns):
                form_col = col1 if idx % 2 == 0 else col2
                display_label = label_mapping.get(col_name, f"Masukkan {col_name.replace('_', ' ')}")
                
                if col_name in feature_encoders:
                    labels_kategori = list(feature_encoders[col_name].classes_)
                    translated_options = [option_mapping.get(opt, opt) for opt in labels_kategori]
                    
                    ui_inputs[col_name] = form_col.selectbox(
                        display_label, 
                        options=translated_options,
                        key=f"ui_{col_name}"
                    )
                else:
                    default_val = int(df[col_name].mean()) if col_name in df.columns else 0
                    ui_inputs[col_name] = form_col.number_input(
                        display_label, 
                        value=default_val,
                        step=1,
                        key=f"ui_{col_name}"
                    )
            
            submitted = st.form_submit_button("JALANKAN KALKULASI PREDIKSI")

        if submitted:
            for col_name in feature_columns:
                val = ui_inputs[col_name]
                if col_name in feature_encoders:
                    input_data[col_name] = reverse_option_mapping.get(val, val)
                else:
                    input_data[col_name] = float(val)

            input_df = pd.DataFrame([input_data])
            
            for col, encoder in feature_encoders.items():
                try:
                    val_str = str(input_df.at[0, col]).strip()
                    known_classes = list(encoder.classes_)
                    matched_class = None
                    
                    for c in known_classes:
                        if str(c).lower() == val_str.lower():
                            matched_class = c
                            break
                    
                    if matched_class is not None:
                        input_df[col] = encoder.transform([matched_class])
                    else:
                        input_df[col] = encoder.transform([known_classes[0]])
                        
                except Exception as e:
                    st.error(f"Gagal memproses fitur '{col}': {e}")
                    st.stop()
            
            input_df = input_df[feature_columns]
            input_scaled = scaler.transform(input_df)
            
            prediksi_angka = model.predict(input_scaled)
            hasil_label = target_encoder.inverse_transform(prediksi_angka)[0]
            
            st.write("---")
            st.subheader("HASIL ANALISIS MODEL")
            st.success(f"Berdasarkan Algoritma {model_name}, Tingkat Kecanduan: {hasil_label}")
            
            if hasattr(model, "predict_proba"):
                probabilitas = model.predict_proba(input_scaled)[0]
                prob_df = pd.DataFrame({
                    "Tingkat Kecanduan": target_encoder.classes_,
                    "Probabilitas": [f"{p*100:.2f}%" for p in probabilitas]
                })
                st.dataframe(prob_df, use_container_width=True)

    # TAB 2: UPLOAD
    with tab_upload:
        st.subheader("PREDIKSI MASAL VIA BERKAS CSV")
        uploaded_file = st.file_uploader("Unggah Berkas CSV Anda", type=["csv"], key="manual_upload_csv")
        
        if uploaded_file is not None:
            user_data = pd.read_csv(uploaded_file)
            st.dataframe(user_data.head(), use_container_width=True)
            
            aligned_data = pd.DataFrame(index=user_data.index)

            for col in feature_columns:
                if col in user_data.columns:
                    aligned_data[col] = user_data[col].copy()
                else:
                    matched_col = None
                    simplified_target = col.lower().replace("_", "").replace(" ", "")
                    for user_col in user_data.columns:
                        if user_col.lower().replace("_", "").replace(" ", "") == simplified_target:
                            matched_col = user_col
                            break
                    
                    if matched_col:
                        aligned_data[col] = user_data[matched_col].copy()
                    else:
                        if col in feature_encoders:
                            aligned_data[col] = feature_encoders[col].classes_[0]
                        else:
                            aligned_data[col] = float(df[col].mean()) if col in df.columns else 0.0

            for col in feature_columns:
                if aligned_data[col].isnull().any():
                    if col in feature_encoders:
                        aligned_data[col] = aligned_data[col].fillna(feature_encoders[col].classes_[0])
                    else:
                        aligned_data[col] = aligned_data[col].fillna(float(df[col].mean()) if col in df.columns else 0.0)

            for col, encoder in feature_encoders.items():
                known_classes = set(encoder.classes_)
                default_class = encoder.classes_[0]
                
                aligned_data[col] = aligned_data[col].astype(str).apply(
                    lambda x: x if x in known_classes else default_class
                )
                aligned_data[col] = encoder.transform(aligned_data[col])

            aligned_data = aligned_data[feature_columns]
            aligned_scaled = scaler.transform(aligned_data)
            csv_preds = model.predict(aligned_scaled)
            csv_labels = target_encoder.inverse_transform(csv_preds)

            final_result = user_data.copy()
            final_result[f"Prediksi ({model_name})"] = csv_labels

            st.subheader("HASIL PREDIKSI BERKAS")
            st.dataframe(final_result, use_container_width=True)

            csv_output = final_result.to_csv(index=False).encode("utf-8")
            st.download_button(
                label=f"UNDUH HASIL PREDIKSI ({model_name})",
                data=csv_output,
                file_name=f"hasil_prediksi_{model_name.lower().replace(' ', '_')}.csv",
                mime="text/csv"
            )

# ==========================================================
# PREDIKSI DATASET UPLOAD
# ==========================================================
elif menu == "Prediksi Dataset Upload":
    st.header("ANALISIS MASSAL BERKAS EKSTERNAL")
    uploaded_prediction = st.file_uploader("Unggah Dataset Eksternal (CSV)", type=["csv"], key="dataset_prediction")
    
    if uploaded_prediction is None:
        st.warning("Silakan unggah berkas CSV terlebih dahulu.")
        st.stop()

    new_data = pd.read_csv(uploaded_prediction)
    st.dataframe(new_data, use_container_width=True)

    required_files = ["Semua_Model.pkl", "Scaler.pkl", "Target_Encoder.pkl", "Feature_Encoders.pkl", "Feature_Columns.pkl"]
    if not all(os.path.exists(file) for file in required_files):
        st.error("Model durung disiapkan. Jalankan menu Training terlebih dahulu.")
        st.stop()

    models = joblib.load("Semua_Model.pkl")
    scaler = joblib.load("Scaler.pkl")
    target_encoder = joblib.load("Target_Encoder.pkl")
    feature_encoders = joblib.load("Feature_Encoders.pkl")
    feature_columns = joblib.load("Feature_Columns.pkl")

    has_target = False
    actual_labels = None
    target_candidates = ["Addiction_Level", "addiction_level", "Addiction Level", "Tingkat Kecanduan", "tingkat kecanduan"]
    
    for candidate in target_candidates:
        if candidate in new_data.columns:
            has_target = True
            actual_labels = new_data[candidate].copy()
            break

    prediction_data = pd.DataFrame(index=new_data.index)

    for col in feature_columns:
        if col in new_data.columns:
            prediction_data[col] = new_data[col].copy()
        else:
            matched_col = None
            simplified_target = col.lower().replace("_", "").replace(" ", "")
            for user_col in new_data.columns:
                if user_col.lower().replace("_", "").replace(" ", "") == simplified_target:
                    matched_col = user_col
                    break
            
            if matched_col:
                prediction_data[col] = new_data[matched_col].copy()
            else:
                if col in feature_encoders:
                    prediction_data[col] = feature_encoders[col].classes_[0]
                else:
                    prediction_data[col] = 0.0

    for col in feature_columns:
        if prediction_data[col].isnull().any():
            if col in feature_encoders:
                prediction_data[col] = prediction_data[col].fillna(feature_encoders[col].classes_[0])
            else:
                prediction_data[col] = prediction_data[col].fillna(0.0)

    for col, encoder in feature_encoders.items():
        known_classes = set(encoder.classes_)
        default_class = encoder.classes_[0]
        
        prediction_data[col] = prediction_data[col].astype(str).apply(
            lambda x: x if x in known_classes else default_class
        )
        prediction_data[col] = encoder.transform(prediction_data[col])

    prediction_data = prediction_data[feature_columns]
    prediction_scaled = scaler.transform(prediction_data)
    
    perbandingan_list = []
    result_data_all = new_data.copy()

    for name, model_obj in models.items():
        preds = model_obj.predict(prediction_scaled)
        pred_labels = target_encoder.inverse_transform(preds)
        result_data_all[f"Prediksi ({name})"] = pred_labels
        
        if has_target:
            acc = accuracy_score(actual_labels.astype(str), pred_labels.astype(str))
            pre = precision_score(actual_labels.astype(str), pred_labels.astype(str), average="weighted", zero_division=0)
            rec = recall_score(actual_labels.astype(str), pred_labels.astype(str), average="weighted", zero_division=0)
            f1 = f1_score(actual_labels.astype(str), pred_labels.astype(str), average="weighted", zero_division=0)
            perbandingan_list.append([name, acc, pre, rec, f1])

    if has_target:
        st.subheader("PERBANDINGAN AKURASI SELURUH MODEL")
        df_compare = pd.DataFrame(
            perbandingan_list, 
            columns=["Model", "Accuracy", "Precision", "Recall", "F1 Score"]
        ).sort_values(by="Accuracy", ascending=False)

        st.dataframe(df_compare.style.format({
            "Accuracy": "{:.2%}", "Precision": "{:.2%}", "Recall": "{:.2%}", "F1 Score": "{:.2%}"
        }), use_container_width=True)

        plt.style.use('dark_background')
        fig, ax = plt.subplots(figsize=(9, 4))
        fig.patch.set_facecolor('#182030')
        ax.set_facecolor('#0b0e14')
        
        bars = ax.bar(df_compare["Model"], df_compare["Accuracy"], color='#00ffcc', edgecolor='#ff0055', linewidth=2)
        ax.set_ylabel("Accuracy Score", color='#ffcc00')
        plt.xticks(rotation=15, color='#00ffcc')
        st.pyplot(fig)

    st.subheader("TABEL HASIL PREDIKSI MASSAL")
    st.dataframe(result_data_all, use_container_width=True)

    csv_data = result_data_all.to_csv(index=False).encode("utf-8")
    st.download_button(
        "UNDUH HASIL PREDIKSI MASSAL (CSV)",
        data=csv_data,
        file_name="hasil_prediksi_massal.csv",
        mime="text/csv"
    )
    import io
import numpy as np
import streamlit as st
from PIL import Image

# ==========================================
# KONFIGURASI HALAMAN STREAMLIT
# ==========================================
st.set_page_config(
    page_title="Pixel Chibi World Cup Generator",
    page_icon="⚽",
    layout="centered",
)

st.title("⚽ Pixel Chibi World Cup Generator")
st.write(
    "Buat dan kustomisasi karakter chibi berpiksel bertema Piala Dunia milikmu!"
)


# ==========================================
# PALET WARNA (HEX / RGB)
# ==========================================
COLOR_TRANSPARENT = (0, 0, 0, 0)
COLOR_OUTLINE = (20, 20, 20, 255)

SKIN_COLORS = {
    "Cerah": (255, 220, 177, 255),
    "Sawo Matang": (224, 172, 105, 255),
    "Gelap": (141, 85, 36, 255),
}

HAIR_COLORS = {
    "Hitam": (30, 30, 30, 255),
    "Cokelat": (101, 67, 33, 255),
    "Pirang": (240, 200, 80, 255),
    "Merah": (184, 51, 42, 255),
}

JERSEY_TEAMS = {
    "Indonesia 🇮🇩": {"base": (200, 16, 46, 255), "shorts": (255, 255, 255, 255)},
    "Jepang 🇯🇵": {"base": (0, 43, 127, 255), "shorts": (255, 255, 255, 255)},
    "Brasil 🇧🇷": {"base": (254, 221, 0, 255), "shorts": (1, 33, 105, 255)},
    "Argentina 🇦🇷": {"base": (117, 170, 219, 255), "shorts": (0, 0, 0, 255)},
    "Jerman 🇩🇪": {"base": (240, 240, 240, 255), "shorts": (20, 20, 20, 255)},
    "Prancis 🇫🇷": {"base": (5, 16, 75, 255), "shorts": (255, 255, 255, 255)},
}


# ==========================================
# FUNGSI MENGGAMBAR PIXEL ART (CANVAS 16x16)
# ==========================================
def draw_pixel_chibi(skin_tone, hair_style, hair_color, team_name):
    # Buat grid 16x16 transparan (RGBA)
    grid = np.zeros((16, 16, 4), dtype=np.uint8)

    skin_rgb = SKIN_COLORS[skin_tone]
    hair_rgb = HAIR_COLORS[hair_color]
    jersey_rgb = JERSEY_TEAMS[team_name]["base"]
    shorts_rgb = JERSEY_TEAMS[team_name]["shorts"]

    # Helper untuk mewarnai piksel
    def set_p(x, y, color):
        if 0 <= x < 16 and 0 <= y < 16:
            grid[y, x] = color

    # --- 1. KEPALA & WAJAH (Piksel Chibi Besar) ---
    # Bentuk Kepala
    for x in range(4, 12):
        for y in range(2, 9):
            set_p(x, y, skin_rgb)

    # Mata Chibi (Hitam dengan kilau putih)
    set_p(5, 5, (0, 0, 0, 255))
    set_p(5, 6, (0, 0, 0, 255))
    set_p(10, 5, (0, 0, 0, 255))
    set_p(10, 6, (0, 0, 0, 255))

    # Pipi Merah (Blush)
    set_p(4, 7, (255, 150, 150, 255))
    set_p(11, 7, (255, 150, 150, 255))

    # --- 2. RAMBUT ---
    if hair_style == "Pendek Standard":
        for x in range(4, 12):
            set_p(x, 1, hair_rgb)
            set_p(x, 2, hair_rgb)
        set_p(3, 2, hair_rgb)
        set_p(3, 3, hair_rgb)
        set_p(12, 2, hair_rgb)
        set_p(12, 3, hair_rgb)
        # Poni
        set_p(6, 3, hair_rgb)
        set_p(9, 3, hair_rgb)

    elif hair_style == "Spiky / Jabrik":
        for x in range(4, 12):
            set_p(x, 2, hair_rgb)
        set_p(4, 0, hair_rgb)
        set_p(6, 0, hair_rgb)
        set_p(8, 0, hair_rgb)
        set_p(10, 0, hair_rgb)
        set_p(5, 1, hair_rgb)
        set_p(7, 1, hair_rgb)
        set_p(9, 1, hair_rgb)
        set_p(11, 1, hair_rgb)

    elif hair_style == "Gondrong":
        for x in range(4, 12):
            set_p(x, 1, hair_rgb)
            set_p(x, 2, hair_rgb)
        for y in range(2, 9):
            set_p(3, y, hair_rgb)
            set_p(12, y, hair_rgb)

    # --- 3. BADAN & JERSEY ---
    # Baju Jersey
    for x in range(5, 11):
        for y in range(9, 12):
            set_p(x, y, jersey_rgb)

    # Lengan Baju
    set_p(4, 9, jersey_rgb)
    set_p(4, 10, skin_rgb)  # Tangan
    set_p(11, 9, jersey_rgb)
    set_p(11, 10, skin_rgb)  # Tangan

    # Celana
    for x in range(5, 11):
        set_p(x, 12, shorts_rgb)

    # Kaki & Sepatu
    set_p(6, 13, skin_rgb)
    set_p(9, 13, skin_rgb)
    set_p(6, 14, (30, 30, 30, 255))  # Sepatu Hitam
    set_p(5, 14, (30, 30, 30, 255))
    set_p(9, 14, (30, 30, 30, 255))
    set_p(10, 14, (30, 30, 30, 255))

    # --- 4. BOLA SEPATU PIXEL DI SEBALAH KARAKTER ---
    ball_white = (240, 240, 240, 255)
    ball_black = (20, 20, 20, 255)
    for bx in range(12, 15):
        for by in range(13, 16):
            set_p(bx, by, ball_white)
    set_p(13, 14, ball_black)

    # Convert array ke PIL Image dan perbesar (Scale Up) tanpa blur (NEAREST)
    img = Image.fromarray(grid, mode="RGBA")
    img = img.resize((320, 320), resample=Image.NEAREST)
    return img


# ==========================================
# INTERFACE KONTROL / SIDEBAR
# ==========================================
st.sidebar.header("🎨 Kustomisasi Karakter")

selected_team = st.sidebar.selectbox("Tim / Negara Jersey", list(JERSEY_TEAMS.keys()))
selected_skin = st.sidebar.selectbox("Warna Kulit", list(SKIN_COLORS.keys()))
selected_hair_style = st.sidebar.selectbox(
    "Gaya Rambut", ["Pendek Standard", "Spiky / Jabrik", "Gondrong"]
)
selected_hair_color = st.sidebar.selectbox(
    "Warna Rambut", list(HAIR_COLORS.keys())
)

# Render Karakter
chibi_image = draw_pixel_chibi(
    selected_skin, selected_hair_style, selected_hair_color, selected_team
)

# Tampilkan Gambar
col1, col2, col3 = st.columns([1, 2, 1])
with col2:
    st.image(
        chibi_image,
        caption=f"Pixel Chibi - {selected_team}",
        use_container_width=True,
    )

# ==========================================
# FITUR UNDUH GAMBAR
# ==========================================
buf = io.BytesIO()
chibi_image.save(buf, format="PNG")
byte_im = buf.getvalue()

st.download_button(
    label="📥 Download Avatar Pixel Chibi (PNG)",
    data=byte_im,
    file_name=f"chibi_worldcup_{selected_team.split()[0].lower()}.png",
    mime="image/png",
)
