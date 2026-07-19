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
# KONFIGURASI HALAMAN
# ==========================================================
st.set_page_config(
    page_title="Prediksi Kecanduan Media Sosial",
    page_icon="🤖",
    layout="wide"
)

st.title("Prediksi Tingkat Kecanduan Media Sosial")
st.write("---")

# ==========================================================
# SIDEBAR MENU
# ==========================================================
menu = st.sidebar.selectbox(
    "Pilih Menu",
    ["Home", "Dataset", "EDA", "Preprocessing", "Training", "Prediksi Manual"]
)

# ==========================================================
# DATASET TRAINING & UTILITY FUNCTIONS
# ==========================================================
TRAINING_DATASET = "Dataset_Revisi_Indikator_Kecanduan.csv"

if not os.path.exists(TRAINING_DATASET):
    st.error(f"Dataset training '{TRAINING_DATASET}' tidak ditemukan.")
    st.stop()

# Load data utama
df = pd.read_csv(TRAINING_DATASET)

def prepare_training_data(data):
    """Membersihkan data dan melakukan encoding awal."""
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
    categorical_columns = X.select_dtypes(include=["object", "category"]).columns
    for col in categorical_columns:
        encoder = LabelEncoder()
        X[col] = encoder.fit_transform(X[col].astype(str))
        feature_encoders[col] = encoder

    feature_columns = list(X.columns)
    
    # Sediakan scaler kosongan untuk di-fit saat training
    scaler = StandardScaler()
    
    return X, y_encoded, target_encoder, feature_encoders, scaler, feature_columns

# Inisialisasi pipeline preproses global berdasarkan dataset dasar
X_prep, y_prep, target_encoder_prep, feature_encoders_prep, scaler_prep, feature_columns_prep = prepare_training_data(df)

def align_and_predict(user_data, feature_columns, feature_encoders, scaler, model, target_encoder, global_df):
    """Menyelaraskan kolom dataset baru (upload) dan melakukan prediksi."""
    aligned_data = pd.DataFrame(index=user_data.index)

    for col in feature_columns:
        if col in user_data.columns:
            aligned_data[col] = user_data[col].copy()
        else:
            # Pengecekan nama kolom alternatif (case & space insensitive)
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
                    aligned_data[col] = float(global_df[col].mean()) if col in global_df.columns else 0.0

    # Penanganan Missing Value
    for col in feature_columns:
        if aligned_data[col].isnull().any():
            if col in feature_encoders:
                aligned_data[col] = aligned_data[col].fillna(feature_encoders[col].classes_[0])
            else:
                aligned_data[col] = aligned_data[col].fillna(float(global_df[col].mean()) if col in global_df.columns else 0.0)

    # Transformasi Kategorikal menggunakan encoder terstandarisasi
    for col, encoder in feature_encoders.items():
        known_classes = set(encoder.classes_)
        default_class = encoder.classes_[0]
        aligned_data[col] = aligned_data[col].astype(str).apply(
            lambda x: x if x in known_classes else default_class
        )
        aligned_data[col] = encoder.transform(aligned_data[col])

    # Pastikan urutan kolom presisi
    aligned_data = aligned_data[feature_columns]
    aligned_scaled = scaler.transform(aligned_data)
    
    preds = model.predict(aligned_scaled)
    labels = target_encoder.inverse_transform(preds)
    return labels, aligned_scaled

# ==========================================================
# TAMPILAN MENU VIA IF-ELIF
# ==========================================================

# --- HOME ---
if menu == "Home":
    st.header("🏠 Machine Learning Project")
    st.write("### Prediksi Tingkat Kecanduan Media Sosial")
    st.write("Aplikasi ini menggunakan berbagai model Machine Learning untuk memprediksi tingkat kecanduan media sosial seseorang.")
    
    st.markdown("""
    ### 🤖 Model yang Didukung:
    * Logistic Regression | Decision Tree | Random Forest | KNN | Naive Bayes | SVM | XGBoost

    ### 📊 Fitur Utama:
    1. **Dataset** - Ringkasan data mentah pelatihan.
    2. **EDA** - Analisis visual dan matriks korelasi data.
    3. **Preprocessing** - Pembersihan data otomatis dan splitting dataset.
    4. **Training Model** - Melatih seluruh model secara simultan dan mencari model terbaik.
    5. **Prediksi Manual** - Form isian interaktif & unggah file CSV untuk prediksi massal.
    """)

# --- DATASET ---
elif menu == "Dataset":
    st.header("📊 Dataset Training")
    st.info(f"Dataset aktif: `{TRAINING_DATASET}`")
    st.dataframe(df, use_container_width=True)

    col1, col2 = st.columns(2)
    col1.metric("Jumlah Baris Data", df.shape[0])
    col2.metric("Jumlah Fitur/Kolom", df.shape[1])
    
    st.subheader("📋 Statistik Deskriptif")
    st.dataframe(df.describe(include="all"), use_container_width=True)

# --- EDA ---
elif menu == "EDA":
    st.header("📈 Exploratory Data Analysis")
    
    st.subheader("🔍 Informasi Kualitas Data")
    info_df = pd.DataFrame({
        "Tipe Data": df.dtypes.astype(str),
        "Missing Value": df.isnull().sum(),
        "Jumlah Unique": df.nunique()
    })
    st.dataframe(info_df, use_container_width=True)

    if "Addiction_Level" in df.columns:
        st.subheader("🎯 Distribusi Kelas Target (Addiction Level)")
        fig, ax = plt.subplots(figsize=(6, 3.5))
        df["Addiction_Level"].value_counts().plot(kind="bar", ax=ax, color='#1E88E5')
        ax.set_ylabel("Jumlah Sampel")
        plt.xticks(rotation=0)
        st.pyplot(fig)

    numeric_cols = df.select_dtypes(include=np.number)
    if not numeric_cols.empty:
        st.subheader("🔥 Matriks Korelasi Fitur Numerik")
        corr = numeric_cols.corr()
        fig, ax = plt.subplots(figsize=(7, 5))
        im = ax.imshow(corr, cmap='coolwarm')
        ax.set_xticks(range(len(corr.columns)))
        ax.set_xticklabels(corr.columns, rotation=45, ha='right')
        ax.set_yticks(range(len(corr.columns)))
        ax.set_yticklabels(corr.columns)
        plt.colorbar(im)
        st.pyplot(fig)

# --- PREPROCESSING ---
elif menu == "Preprocessing":
    st.header("⚙️ Preprocessing Data")
    st.write(f"Ukuran awal dataset: **{df.shape[0]} baris, {df.shape[1]} kolom**")

    X_train, X_test, y_train, y_test = train_test_split(
        X_prep, y_prep, test_size=0.2, random_state=42, stratify=y_prep
    )

    st.success("Preprocessing & Data Splitting Selesai dilakukan!")
    st.markdown(f"""
    * **Jumlah Data Training (80%):** {X_train.shape[0]} sampel
    * **Jumlah Data Testing (20%):** {X_test.shape[0]} sampel
    * **Daftar Target Output:** {list(target_encoder_prep.classes_)}
    """)

# --- TRAINING ---
elif menu == "Training":
    st.header("🤖 Training Model")
    
    X_train, X_test, y_train, y_test = train_test_split(
        X_prep, y_prep, test_size=0.2, random_state=42, stratify=y_prep
    )

    # Validasi & Fitting Scaler secara konsisten
    X_train_scaled = scaler_prep.fit_transform(X_train)
    X_test_scaled = scaler_prep.transform(X_test)

    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000),
        "Decision Tree": DecisionTreeClassifier(random_state=42),
        "Random Forest": RandomForestClassifier(n_estimators=200, random_state=42),
        "KNN": KNeighborsClassifier(),
        "Naive Bayes": GaussianNB(),
        "SVM": SVC(probability=True, random_state=42)
    }
    if xgb:
        models["XGBoost"] = XGBClassifier(eval_metric="mlogloss", random_state=42)

    hasil = []
    trained_models = {}
    
    progress_bar = st.progress(0)
    for idx, (name, model) in enumerate(models.items()):
        model.fit(X_train_scaled, y_train)
        preds = model.predict(X_test_scaled)
        
        acc = accuracy_score(y_test, preds)
        prec = precision_score(y_test, preds, average="weighted", zero_division=0)
        rec = recall_score(y_test, preds, average="weighted", zero_division=0)
        f1 = f1_score(y_test, preds, average="weighted", zero_division=0)
        
        hasil.append([name, acc, prec, rec, f1])
        trained_models[name] = model
        progress_bar.progress((idx + 1) / len(models))

    hasil_df = pd.DataFrame(hasil, columns=["Model", "Accuracy", "Precision", "Recall", "F1 Score"])
    hasil_df = hasil_df.sort_values(by="Accuracy", ascending=False).reset_index(drop=True)

    st.success("Proses Training Berhasil Terlaksana!")
    st.dataframe(hasil_df.style.format({
        "Accuracy": "{:.2%}", "Precision": "{:.2%}", "Recall": "{:.2%}", "F1 Score": "{:.2%}"
    }), use_container_width=True)

    best_model_name = hasil_df.iloc[0]["Model"]
    st.subheader(f"🏆 Model Terbaik saat ini: {best_model_name}")

    # Simpan Seluruh Objek Model & Pipeline Ekstraksi
    joblib.dump(trained_models, "Semua_Model.pkl")
    joblib.dump(scaler_prep, "Scaler.pkl")
    joblib.dump(target_encoder_prep, "Target_Encoder.pkl")
    joblib.dump(feature_encoders_prep, "Feature_Encoders.pkl")
    joblib.dump(feature_columns_prep, "Feature_Columns.pkl")
    st.info("Artefak model dan preprocessing scaler berhasil disimpan lokal.")

# --- PREDIKSI MANUAL ---
elif menu == "Prediksi Manual":
    st.header("🔍 Prediksi Manual & Unggah Data")
    
    required_files = ["Semua_Model.pkl", "Scaler.pkl", "Target_Encoder.pkl", "Feature_Encoders.pkl", "Feature_Columns.pkl"]
    if not all(os.path.exists(f) for f in required_files):
        st.warning("⚠️ Komponen Model Belum Siap. Silakan masuk ke menu **Training** terlebih dahulu untuk melatih model.")
        st.stop()

    # Load Komponen dari Storage lokal
    models = joblib.load("Semua_Model.pkl")
    scaler = joblib.load("Scaler.pkl")
    target_encoder = joblib.load("Target_Encoder.pkl")
    feature_encoders = joblib.load("Feature_Encoders.pkl")
    feature_columns = joblib.load("Feature_Columns.pkl")

    model_name = st.selectbox("Pilih Model untuk Menjalankan Prediksi", list(models.keys()))
    model = models[model_name]
    
    tab_form, tab_upload = st.tabs(["📝 Input Form Mandiri", "📁 Upload File CSV"])

    # TAB 1: FORM MANDIRI
    with tab_form:
        st.subheader("Karakteristik Pengguna Baru:")
        
        label_mapping = {
            "Age": "Usia / Umur", "Gender": "Jenis Kelamin",
            "Academic_Level": "Jenjang / Tingkat Pendidikan", "Country": "Negara Asal",
            "Avg_Daily_Usage_Hours": "Rata-rata Durasi Penggunaan Harian (Jam)",
            "Most_Used_Platform": "Platform Paling Sering Digunakan",
            "Affects_Academic_Performance": "Apakah Memengaruhi Performa Akademik?",
            "Sleep_Hours_Per_Night": "Durasi Tidur per Malam (Jam)",
            "Mental_Health_Score": "Skor Kesehatan Mental", "Physical_Activity": "Tingkat Aktivitas Fisik"
        }

        option_mapping = {
            "Male": "Laki-laki", "Female": "Perempuan", "Undergraduate": "Sarjana (S1)",
            "Graduate": "Pascasarjana (S2/S3)", "High School": "SMA / Sederajat",
            "School": "Sekolah Dasar/Menengah", "Yes": "Ya", "No": "Tidak",
            "Low": "Rendah", "Medium": "Sedang", "High": "Tinggi"
        }
        reverse_option_mapping = {v: k for k, v in option_mapping.items()}
        
        with st.form("form_prediksi_manual"):
            col1, col2 = st.columns(2)
            ui_inputs = {}
            
            for idx, col_name in enumerate(feature_columns):
                form_col = col1 if idx % 2 == 0 else col2
                display_label = label_mapping.get(col_name, f"Input {col_name}")
                
                if col_name in feature_encoders:
                    labels_kategori = list(feature_encoders[col_name].classes_)
                    translated_options = [option_mapping.get(opt, opt) for opt in labels_kategori]
                    ui_inputs[col_name] = form_col.selectbox(display_label, options=translated_options, key=f"ui_{col_name}")
                else:
                    min_val = float(df[col_name].min()) if col_name in df.columns else 0.0
                    max_val = float(df[col_name].max()) if col_name in df.columns else 100.0
                    mean_val = float(df[col_name].mean()) if col_name in df.columns else 0.0
                    ui_inputs[col_name] = form_col.number_input(display_label, min_value=min_val, max_value=max_val, value=mean_val, key=f"ui_{col_name}")
            
            submitted = st.form_submit_button("🔮 Hitung Prediksi")

        if submitted:
            input_data = {}
            for col_name in feature_columns:
                val = ui_inputs[col_name]
                input_data[col_name] = reverse_option_mapping.get(val, val)

            input_df = pd.DataFrame([input_data])
            
            # Ubah data kategorikal menggunakan encoder
            for col, encoder in feature_encoders.items():
                val_str = str(input_df.at[0, col]).strip()
                matched_class = next((c for c in encoder.classes_ if str(c).lower() == val_str.lower()), encoder.classes_[0])
                input_df[col] = encoder.transform([matched_class])
            
            input_df = input_df[feature_columns]
            input_scaled = scaler.transform(input_df)
            
            prediksi_angka = model.predict(input_scaled)
            hasil_label = target_encoder.inverse_transform(prediksi_angka)[0]
            
            st.success(f"Hasil Klasifikasi Model ({model_name}): **{hasil_label}**")
            
            if hasattr(model, "predict_proba"):
                probabilitas = model.predict_proba(input_scaled)[0]
                prob_df = pd.DataFrame({
                    "Tingkat Kecanduan": target_encoder.classes_,
                    "Confidence Rate": [f"{p*100:.2f}%" for p in probabilitas]
                })
                st.dataframe(prob_df, use_container_width=True)

    # TAB 2: UPLOAD CSV
    with tab_upload:
        st.subheader("Bulk Predict dengan CSV")
        uploaded_file = st.file_uploader("Pilih file CSV", type=["csv"], key="file_pred_csv")
        
        if uploaded_file is not None:
            user_data = pd.read_csv(uploaded_file)
            st.write("📂 **Pratinjau Data:**")
            st.dataframe(user_data.head(), use_container_width=True)
            
            csv_labels, aligned_scaled = align_and_predict(
                user_data, feature_columns, feature_encoders, scaler, model, target_encoder, df
            )

            final_result = user_data.copy()
            final_result[f"Prediksi_{model_name}"] = csv_labels

            # Validasi evaluasi jika terdapat kolom target asli
            has_target = False
            for target_cand in ["Addiction_Level", "addiction_level"]:
                if target_cand in user_data.columns:
                    has_target = True
                    actual_labels = user_data[target_cand].astype(str)
                    break

            if has_target:
                st.subheader("📊 Metrik Akurasi File Upload")
                acc = accuracy_score(actual_labels, csv_labels.astype(str))
                metrics_df = pd.DataFrame([[model_name, acc]], columns=["Model", "Accuracy"])
                st.dataframe(metrics_df.style.format({"Accuracy": "{:.2%}"}))

            st.subheader("📋 Hasil Akhir Tabel Prediksi")
            st.dataframe(final_result, use_container_width=True)
