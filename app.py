import os
import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    confusion_matrix,
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
# KONFIGURASI HALAMAN
# ==========================================================
st.set_page_config(
    page_title="Prediksi Kecanduan Media Sosial",
    page_icon="https://dianisa.com/_next/image?url=https%3A%2F%2Fassets.dianisa.com%2Fmedia%2F2024%2F09%2FLogo-Universitas-Negeri-Makassar.webp&w=3840&q=75",
    layout="wide"
)

st.title(" Prediksi Tingkat Kecanduan Media Sosial")
st.write("---")

# ==========================================================
# SIDEBAR MENU
# ==========================================================
menu = st.sidebar.selectbox(
    "Pilih Menu",
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
    st.error(f"Dataset training '{TRAINING_DATASET}' tidak ditemukan.")
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
        st.error("Kolom 'Addiction_Level' tidak ditemukan pada dataset training.")
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
    st.header("Machine Learning Project")
    st.image("https://i0.wp.com/metrum.co.id/kanal/uploads/2022/01/pict-19-kecanduan-medsos.jpg?w=800&ssl=1")
    st.write(
        """
        ## Prediksi Tingkat Kecanduan Media Sosial
        Aplikasi ini menggunakan Machine Learning untuk memprediksi tingkat kecanduan media sosial seseorang.

        ### Model yang Digunakan:
        ✔ Logistic Regression | ✔ Decision Tree | ✔ Random Forest | ✔ KNN | ✔ Naive Bayes | ✔ SVM | ✔ XGBoost

        ### Fitur Utama:
        * 📊 **Dataset** - Melihat ringkasan data training.
        * 📈 **EDA** - Exploratory Data Analysis & Visualisasi Korelasi.
        * ⚙️ **Preprocessing** - Pembersihan data & pembagian dataset.
        * 🤖 **Training Model** - Melatih data ke semua algoritma secara simultan.
        * 🔍 **Prediksi Manual** - Input data mandiri via form input.
        * 📁 **Prediksi Dataset Upload** - Upload file CSV eksternal untuk diprediksi massal.
        """
    )

# --- DATASET ---
elif menu == "Dataset":
    st.header("📊 Dataset Training")
    st.info(f"Dataset training yang digunakan: {TRAINING_DATASET}")
    st.dataframe(df, use_container_width=True)

    col1, col2, col3 = st.columns(3)
    col1.metric("Jumlah Baris", df.shape[0])
    col2.metric("Jumlah Kolom", df.shape[1])
    
    st.subheader("Statistik Deskriptif")
    st.write(df.describe(include="all"))

    st.subheader("Tipe Data")
    st.write(df.dtypes.astype(str))

# --- EDA ---
elif menu == "EDA":
    st.header("📈 Exploratory Data Analysis")
    st.subheader("5 Data Pertama")
    st.dataframe(df.head())

    st.subheader("Informasi Dataset")
    info = pd.DataFrame({
        "Kolom": df.columns,
        "Tipe Data": df.dtypes.astype(str),
        "Missing Value": df.isnull().sum(),
        "Unique": df.nunique()
    })
    st.dataframe(info, use_container_width=True)

    if "Addiction_Level" in df.columns:
        st.subheader("Distribusi Addiction Level")
        fig, ax = plt.subplots(figsize=(6, 3))
        df["Addiction_Level"].value_counts().plot(kind="bar", ax=ax, color='#1E88E5')
        ax.set_xlabel("Tingkat Kecanduan")
        ax.set_ylabel("Jumlah")
        st.pyplot(fig)

    numeric = df.select_dtypes(include=np.number)
    if not numeric.empty:
        st.subheader("Distribusi Variabel Numerik")
        for col in numeric.columns:
            fig, ax = plt.subplots(figsize=(5, 2.5))
            ax.hist(numeric[col], bins=20, color='#4CAF50')
            ax.set_title(f"Distribusi {col}")
            st.pyplot(fig)

        st.subheader("Correlation Matrix")
        corr = numeric.corr()
        fig, ax = plt.subplots(figsize=(8, 6))
        im = ax.imshow(corr, cmap='coolwarm')
        ax.set_xticks(range(len(corr.columns)))
        ax.set_xticklabels(corr.columns, rotation=90)
        ax.set_yticks(range(len(corr.columns)))
        ax.set_yticklabels(corr.columns)
        plt.colorbar(im)
        st.pyplot(fig)

# --- PREPROCESSING ---
elif menu == "Preprocessing":
    st.header("⚙️ Preprocessing Data")
    st.write(f"Dataset awal: **{df.shape[0]} baris, {df.shape[1]} kolom**")

    X, y, target_encoder, feature_encoders, scaler, feature_columns = prepare_training_data(df)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    st.success("Preprocessing berhasil dilakukan!")
    st.write(f"✔ Jumlah Data Training: {X_train.shape[0]} sampel")
    st.write(f"✔ Jumlah Data Testing: {X_test.shape[0]} sampel")
    st.write("Target Classes:", list(target_encoder.classes_))

# --- TRAINING ---
elif menu == "Training":
    st.header("🤖 Training Model")
    
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

    st.success("Training selesai!")
    st.dataframe(hasil_df.style.format({
        "Accuracy": "{:.2%}", "Precision": "{:.2%}", "Recall": "{:.2%}", "F1 Score": "{:.2%}"
    }), use_container_width=True)

    best_model_name = hasil_df.iloc[0]["Model"]
    st.subheader(f"🏆 Model Terbaik: {best_model_name}")

    # Simpan Artefak Model
    joblib.dump(trained_models, "Semua_Model.pkl")
    joblib.dump(scaler, "Scaler.pkl")
    joblib.dump(target_encoder, "Target_Encoder.pkl")
    joblib.dump(feature_encoders, "Feature_Encoders.pkl")
    joblib.dump(feature_columns, "Feature_Columns.pkl")
    st.success("Semua model dan encoder berhasil diekspor!")

    # Visualisasi
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(hasil_df["Model"], hasil_df["Accuracy"], color='#26A69A')
    ax.set_ylabel("Accuracy Score")
    plt.xticks(rotation=15)
    st.pyplot(fig)

# --- PREDIKSI MANUAL ---
# ==========================================================
# PREDIKSI MANUAL
# ==========================================================
# --- PREDIKSI MANUAL ---
# ==========================================================
# PREDIKSI MANUAL (MENDUKUNG INPUT FORM & UPLOAD CSV)
# ==========================================================
elif menu == "Prediksi Manual":
    st.header("🔍 Prediksi Manual & Unggah Data")
    
    required_files = [
        "Semua_Model.pkl",
        "Scaler.pkl",
        "Target_Encoder.pkl",
        "Feature_Encoders.pkl",
        "Feature_Columns.pkl"
    ]
    
    if not all(os.path.exists(file) for file in required_files):
        st.error("Silakan jalankan menu **Training** terlebih dahulu agar model dan komponen siap digunakan.")
        st.stop()

    # Load semua komponen model
    models = joblib.load("Semua_Model.pkl")
    scaler = joblib.load("Scaler.pkl")
    target_encoder = joblib.load("Target_Encoder.pkl")
    feature_encoders = joblib.load("Feature_Encoders.pkl")
    feature_columns = joblib.load("Feature_Columns.pkl")

    # Pilih Model
    model_name = st.selectbox("Pilih Model untuk Prediksi", list(models.keys()))
    model = models[model_name]

    st.write("---")
    
    # Navigasi Metode Input menggunakan Tabs
    tab_form, tab_upload = st.tabs(["📝 Input Form Mandiri", "📁 Upload File CSV"])

    # ------------------------------------------------------
    # TAB 1: INPUT FORM MANDIRI
    # ------------------------------------------------------
    # ------------------------------------------------------
    # TAB 1: INPUT FORM MANDIRI
    # ------------------------------------------------------
    # ------------------------------------------------------
    # TAB 1: INPUT FORM MANDIRI
    # ------------------------------------------------------
    # ------------------------------------------------------
    # TAB 1: INPUT FORM MANDIRI
    # ------------------------------------------------------
    with tab_form:
        st.subheader("Masukkan Data Pengguna Baru:")
        input_data = {}
        
        # 1. Kamus translasi label visual (Bahasa Indonesia Akademis)
        label_mapping = {
            "Age": "Masukkan Usia / Umur",
            "Gender": "Pilih Jenis Kelamin",
            "Academic_Level": "Pilih Jenjang / Tingkat Pendidikan",
            "Country": "Pilih Negara Asal",
            "Avg_Daily_Usage_Hours": "Masukkan Rata-rata Durasi Penggunaan Harian (Jam)",
            "Most_Used_Platform": "Pilih Platform yang Paling Sering Digunakan",
            "Affects_Academic_Performance": "Apakah Memengaruhi Performa Akademik?",
            "Sleep_Hours_Per_Night": "Masukkan Durasi Tidur per Malam (Jam)",
            "Mental_Health_Score": "Masukkan Skor Kesehatan Mental",
            "Physical_Activity": "Pilih Tingkat Aktivitas Fisik"
        }

        # 2. Kamus translasi opsi dropdown ke Bahasa Indonesia
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

        # Kebalikan mapping untuk mengembalikan nilai asli ke Model ML
        reverse_option_mapping = {v: k for k, v in option_mapping.items()}
        
        with st.form("form_prediksi_manual"):
            col1, col2 = st.columns(2)
            
            # Simpan referensi objek input UI agar nilainya bisa dibaca saat submit
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
                    min_val = float(df[col_name].min()) if col_name in df.columns else 0.0
                    max_val = float(df[col_name].max()) if col_name in df.columns else 100.0
                    mean_val = float(df[col_name].mean()) if col_name in df.columns else 0.0
                    
                    ui_inputs[col_name] = form_col.number_input(
                        display_label, 
                        min_value=min_val,
                        max_value=max_val,
                        value=mean_val,
                        key=f"ui_{col_name}"
                    )
            
            submitted = st.form_submit_button("🔮 Lakukan Prediksi Form")

        # PROSES EKSEKUSI PREDIKSI SETELAH TOMBOL DIKLIK
        if submitted:
            # Pindahkan & konversi data dari UI ke dictionary input data asli
            for col_name in feature_columns:
                val = ui_inputs[col_name]
                # Jika nilainya ada di kamus Indonesia, kembalikan ke teks English aslinya
                if col_name in feature_encoders:
                    input_data[col_name] = reverse_option_mapping.get(val, val)
                else:
                    input_data[col_name] = val

            # Ubah ke DataFrame
            input_df = pd.DataFrame([input_data])
            
            # Lakukan Encoding menggunakan encoder bawaan model
            for col, encoder in feature_encoders.items():
                try:
                    # Normalisasi teks ke string biasa
                    val_str = str(input_df.at[0, col]).strip()
                    
                    # Cek pencocokan string secara fleksibel (tidak sensitif huruf besar/kecil)
                    known_classes = list(encoder.classes_)
                    matched_class = None
                    
                    for c in known_classes:
                        if str(c).lower() == val_str.lower():
                            matched_class = c
                            break
                    
                    # Jika cocok, pakai label asli hasil training, jika tidak, pakai default kelas pertama
                    if matched_class is not None:
                        input_df[col] = encoder.transform([matched_class])
                    else:
                        input_df[col] = encoder.transform([known_classes[0]])
                        
                except Exception as e:
                    st.error(f"Terjadi kesalahan saat encoding fitur '{col}': {e}")
                    st.stop()
            
            # Selaraskan urutan kolom dan scaling
            input_df = input_df[feature_columns]
            input_scaled = scaler.transform(input_df)
            
            # Prediksi
            prediksi_angka = model.predict(input_scaled)
            hasil_label = target_encoder.inverse_transform(prediksi_angka)[0]
            
            # TAMPILKAN HASILNYA DI BAWAH FORM
            st.write("---")
            st.subheader("📊 Hasil Prediksi Form")
            st.success(f"Berdasarkan model **{model_name}**, tingkat kecanduan pengguna ini adalah: **{hasil_label}**")
            
            if hasattr(model, "predict_proba"):
                probabilitas = model.predict_proba(input_scaled)[0]
                st.write("**Probabilitas Keyakinan Model:**")
                prob_df = pd.DataFrame({
                    "Tingkat Kecanduan": target_encoder.classes_,
                    "Keyakinan (Persentase)": [f"{p*100:.2f}%" for p in probabilitas]
                })
                st.dataframe(prob_df, use_container_width=True)
    # ------------------------------------------------------
    # TAB 2: UPLOAD FILE CSV (ADAPTIF MENYESUAIKAN DATASET)
    # ------------------------------------------------------
    with tab_upload:
        st.subheader("Predict via CSV Upload")
        st.info("Sistem akan otomatis mendeteksi, menyelaraskan, dan menyesuaikan kolom dataset yang Anda unggah.")
        
        uploaded_file = st.file_uploader("Upload Dataset CSV Anda", type=["csv"], key="manual_upload_csv")
        
        if uploaded_file is not None:
            user_data = pd.read_csv(uploaded_file)
            st.write("📄 **Preview Data yang Di-upload:**")
            st.dataframe(user_data.head(), use_container_width=True)
            
            # Kerangka DataFrame baru untuk diselaraskan dengan dataset training
            aligned_data = pd.DataFrame(index=user_data.index)

            # Proses Penyelarasan Otomatis
            for col in feature_columns:
                # 1. Jika nama cocok persis
                if col in user_data.columns:
                    aligned_data[col] = user_data[col].copy()
                # 2. Jika nama mirip (tidak sensitif huruf besar/kecil dan spasi)
                else:
                    matched_col = None
                    simplified_target = col.lower().replace("_", "").replace(" ", "")
                    for user_col in user_data.columns:
                        if user_col.lower().replace("_", "").replace(" ", "") == simplified_target:
                            matched_col = user_col
                            break
                    
                    if matched_col:
                        aligned_data[col] = user_data[matched_col].copy()
                    # 3. Jika kolom tidak ada sama sekali, buat nilai default
                    else:
                        if col in feature_encoders:
                            aligned_data[col] = feature_encoders[col].classes_[0]
                        else:
                            aligned_data[col] = float(df[col].mean()) if col in df.columns else 0.0

            # Penanganan data kosong (NaN)
            for col in feature_columns:
                if aligned_data[col].isnull().any():
                    if col in feature_encoders:
                        aligned_data[col] = aligned_data[col].fillna(feature_encoders[col].classes_[0])
                    else:
                        aligned_data[col] = aligned_data[col].fillna(float(df[col].mean()) if col in df.columns else 0.0)

            # Transformasi / Label Encoding untuk kolom kategorikal
            for col, encoder in feature_encoders.items():
                known_classes = set(encoder.classes_)
                default_class = encoder.classes_[0]
                
                # Ubah teks baru yang tidak dikenal saat training ke kelas default
                aligned_data[col] = aligned_data[col].astype(str).apply(
                    lambda x: x if x in known_classes else default_class
                )
                aligned_data[col] = encoder.transform(aligned_data[col])

            # Susun ulang urutan kolom agar identik dengan data training
            aligned_data = aligned_data[feature_columns]

            # Lakukan Scaling dan Prediksi menggunakan model terpilih
            aligned_scaled = scaler.transform(aligned_data)
            csv_preds = model.predict(aligned_scaled)
            csv_labels = target_encoder.inverse_transform(csv_preds)

            # Gabungkan hasil prediksi ke data asli user
            final_result = user_data.copy()
            final_result[f"Hasil Prediksi ({model_name})"] = csv_labels

            st.write("---")
            st.subheader("📊 Hasil Prediksi Dataset Upload")
            st.dataframe(final_result, use_container_width=True)

            # Tombol Download Hasil
            csv_output = final_result.to_csv(index=False).encode("utf-8")
            st.download_button(
                label=f"⬇️ Download Hasil Prediksi {model_name} (CSV)",
                data=csv_output,
                file_name=f"hasil_prediksi_{model_name.lower().replace(' ', '_')}.csv",
                mime="text/csv"
            )
# --- PREDIKSI DATASET UPLOAD ---
# ==========================================================
# PREDIKSI DATASET UPLOAD (ADAPTIF & DUKUNG SEMUA FORMAT)
# ==========================================================
elif menu == "Prediksi Dataset Upload":
    st.header("📁 Upload Dataset Baru & Analisis Model")
    st.info(
        """
        ### 💡 Fitur Deteksi Kolom Otomatis Aktif
        Anda dapat mengunggah file CSV dengan format kolom apa pun. Sistem akan otomatis menyelaraskan 
        dan mengisi kolom yang kurang agar model tidak error dan prediksi tetap bisa berjalan!
        """
    )

    uploaded_prediction = st.file_uploader("Upload Dataset Baru (CSV)", type=["csv"], key="dataset_prediction")
    if uploaded_prediction is None:
        st.warning("Silakan unggah berkas CSV terlebih dahulu.")
        st.stop()

    new_data = pd.read_csv(uploaded_prediction)
    st.subheader("📄 Dataset Asli yang Di-upload User")
    st.dataframe(new_data, use_container_width=True)

    # Validasi keberadaan file model eksternal
    required_files = ["Semua_Model.pkl", "Scaler.pkl", "Target_Encoder.pkl", "Feature_Encoders.pkl", "Feature_Columns.pkl"]
    if not all(os.path.exists(file) for file in required_files):
        st.error("Model/Komponen scaler belum tersedia. Silakan jalankan menu Training terlebih dahulu.")
        st.stop()

    # Load komponen training awal
    models = joblib.load("Semua_Model.pkl")
    scaler = joblib.load("Scaler.pkl")
    target_encoder = joblib.load("Target_Encoder.pkl")
    feature_encoders = joblib.load("Feature_Encoders.pkl")
    feature_columns = joblib.load("Feature_Columns.pkl")

    # Cek apakah target aktual ada di dataset baru (untuk mode evaluasi)
    has_target = False
    actual_labels = None
    target_candidates = ["Addiction_Level", "addiction_level", "Addiction Level", "Tingkat Kecanduan", "tingkat kecanduan"]
    
    for candidate in target_candidates:
        if candidate in new_data.columns:
            has_target = True
            actual_labels = new_data[candidate].copy()
            break

    # ======================================================
    # PROSES PENYELARASAN DATA SECARA OTOMATIS (APAPUN FORMATNYA)
    # ======================================================
    prediction_data = pd.DataFrame(index=new_data.index)

    for col in feature_columns:
        # 1. Jika nama kolom COCOK PERSIS
        if col in new_data.columns:
            prediction_data[col] = new_data[col].copy()
        
        # 2. Jika nama mirip (tidak sensitif huruf besar/kecil atau spasi/underscore)
        else:
            matched_col = None
            simplified_target = col.lower().replace("_", "").replace(" ", "")
            for user_col in new_data.columns:
                if user_col.lower().replace("_", "").replace(" ", "") == simplified_target:
                    matched_col = user_col
                    break
            
            if matched_col:
                prediction_data[col] = new_data[matched_col].copy()
            
            # 3. Jika kolom benar-benar tidak ada di data user (Buat kolom tiruan otomatis)
            else:
                if col in feature_encoders:
                    # Kolom kategori diisi dengan nilai modus (terbanyak) atau nilai pertama dari encoder
                    default_cat = feature_encoders[col].classes_[0]
                    prediction_data[col] = default_cat
                else:
                    # Kolom numerik diisi dengan angka 0 atau rata-rata dummy
                    prediction_data[col] = 0.0

    # Menangani missing value (NaN) jika data buatan user ada yang kosong
    for col in feature_columns:
        if prediction_data[col].isnull().any():
            if col in feature_encoders:
                prediction_data[col] = prediction_data[col].fillna(feature_encoders[col].classes_[0])
            else:
                prediction_data[col] = prediction_data[col].fillna(0.0)

    # 4. Transformasi/Encoding Kategori (Aman dari data baru tak dikenal)
    for col, encoder in feature_encoders.items():
        # Dapatkan nilai dasar yang diketahui oleh encoder saat training
        known_classes = set(encoder.classes_)
        default_class = encoder.classes_[0]
        
        # Jika user memasukkan teks baru yang tidak ada saat training, ubah otomatis ke default_class
        prediction_data[col] = prediction_data[col].astype(str).apply(
            lambda x: x if x in known_classes else default_class
        )
        
        # Jalankan Label Encoder
        prediction_data[col] = encoder.transform(prediction_data[col])

    # Pastikan urutan dan bentuk kolom 100% konsisten dengan training
    prediction_data = prediction_data[feature_columns]

    # Scaling
    prediction_scaled = scaler.transform(prediction_data)
    
    # ======================================================
    # PROSES PREDIKSI MULTI-MODEL
    # ======================================================
    st.write("---")
    perbandingan_list = []
    result_data_all = new_data.copy()

    for name, model_obj in models.items():
        preds = model_obj.predict(prediction_scaled)
        pred_labels = target_encoder.inverse_transform(preds)
        result_data_all[f"Prediksi ({name})"] = pred_labels
        
        if has_target:
            # Pastikan label aktual di-string-kan agar cocok saat evaluasi skor
            acc = accuracy_score(actual_labels.astype(str), pred_labels.astype(str))
            pre = precision_score(actual_labels.astype(str), pred_labels.astype(str), average="weighted", zero_division=0)
            rec = recall_score(actual_labels.astype(str), pred_labels.astype(str), average="weighted", zero_division=0)
            f1 = f1_score(actual_labels.astype(str), pred_labels.astype(str), average="weighted", zero_division=0)
            perbandingan_list.append([name, acc, pre, rec, f1])

    # ======================================================
    # TAMPILKAN OUTPUT/EVALUASI
    # ======================================================
    if has_target:
        st.subheader("📈 Perbandingan Performa Semua Model (Berdasarkan Nilai Aktual)")
        df_compare = pd.DataFrame(
            perbandingan_list, 
            columns=["Model", "Accuracy", "Precision", "Recall", "F1 Score"]
        ).sort_values(by="Accuracy", ascending=False)

        st.dataframe(df_compare.style.format({
            "Accuracy": "{:.2%}", "Precision": "{:.2%}", "Recall": "{:.2%}", "F1 Score": "{:.2%}"
        }), use_container_width=True)

        fig, ax = plt.subplots(figsize=(10, 4))
        bars = ax.bar(df_compare["Model"], df_compare["Accuracy"], color='#1E88E5')
        ax.set_ylabel("Accuracy Score")
        plt.xticks(rotation=15)
        
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.01, f"{height*100:.1f}%", ha='center', va='bottom', fontsize=9)
        st.pyplot(fig)
    else:
        st.subheader("📊 Hasil Prediksi Massal Model")
        st.info("Dataset eksternal berhasil diproses! Metrik akurasi/grafik komparasi tidak dimunculkan karena data tidak berisi kolom target aktual.")

    st.subheader("📋 Data Lengkap Hasil Prediksi Seluruh Model")
    st.dataframe(result_data_all, use_container_width=True)

    csv_data = result_data_all.to_csv(index=False).encode("utf-8")
    st.download_button(
        "⬇️ Download Hasil Prediksi Lengkap (CSV)",
        data=csv_data,
        file_name="hasil_prediksi_massal.csv",
        mime="text/csv"
    )
