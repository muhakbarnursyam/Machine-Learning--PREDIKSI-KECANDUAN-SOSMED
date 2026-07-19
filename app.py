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
    page_icon="https://i.pinimg.com/originals/13/a1/56/13a15667ed324c559ca73f6d2108392b.jpg",
    layout="wide"
)

st.title("Prediksi Tingkat Kecanduan Media Sosial")
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
        "Prediksi Manual"
    ]
)

# ==========================================================
# DATASET TRAINING & UTILITY FUNCTIONS
# ==========================================================
TRAINING_DATASET = "Dataset_Revisi_Indikator_Kecanduan.csv"

if not os.path.exists(TRAINING_DATASET):
    st.error(f"Dataset training '{TRAINING_DATASET}' tidak ditemukan.")
    st.stop()

df = pd.read_csv(TRAINING_DATASET)

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
    scaler = StandardScaler()
    
    return X, y_encoded, target_encoder, feature_encoders, scaler, feature_columns

def align_and_predict(user_data, feature_columns, feature_encoders, scaler, model, target_encoder, model_name):
    """Fungsi pembantu untuk menyelaraskan kolom dataset baru dan melakukan prediksi."""
    aligned_data = pd.DataFrame(index=user_data.index)

    for col in feature_columns:
        if col in user_data.columns:
            aligned_data[col] = user_data[col].copy()
        else:
            # Pengecekan nama kolom yang mirip (case-insensitive & space-insensitive)
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

    # Penanganan Missing Value
    for col in feature_columns:
        if aligned_data[col].isnull().any():
            if col in feature_encoders:
                aligned_data[col] = aligned_data[col].fillna(feature_encoders[col].classes_[0])
            else:
                aligned_data[col] = aligned_data[col].fillna(float(df[col].mean()) if col in df.columns else 0.0)

    # Transformasi Kategorikal
    for col, encoder in feature_encoders.items():
        known_classes = set(encoder.classes_)
        default_class = encoder.classes_[0]
        aligned_data[col] = aligned_data[col].astype(str).apply(
            lambda x: x if x in known_classes else default_class
        )
        aligned_data[col] = encoder.transform(aligned_data[col])

    aligned_data = aligned_data[feature_columns]
    aligned_scaled = scaler.transform(aligned_data)
    
    preds = model.predict(aligned_scaled)
    labels = target_encoder.inverse_transform(preds)
    return labels, aligned_scaled

# Cache data preprocessing agar tidak running berulang kali pada menu statis
X_prep, y_prep, target_encoder_prep, feature_encoders_prep, scaler_prep, feature_columns_prep = prepare_training_data(df)

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
        * 🔍 **Prediksi Manual** - Input data mandiri via form input atau unggah CSV massal.
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

    X_train, X_test, y_train, y_test = train_test_split(X_prep, y_prep, test_size=0.2, random_state=42, stratify=y_prep)

    st.success("Preprocessing berhasil dilakukan!")
    st.write(f"✔ Jumlah Data Training: {X_train.shape[0]} sampel")
    st.write(f"✔ Jumlah Data Testing: {X_test.shape[0]} sampel")
    st.write("Target Classes:", list(target_encoder_prep.classes_))

# --- TRAINING ---
elif menu == "Training":
    st.header("🤖 Training Model")
    
    X_train, X_test, y_train, y_test = train_test_split(X_prep, y_prep, test_size=0.2, random_state=42, stratify=y_prep)

    X_train_scaled = scaler_prep.fit_transform(X_train)
    X_test_scaled = scaler_prep.transform(X_test)

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
    joblib.dump(scaler_prep, "Scaler.pkl")
    joblib.dump(target_encoder_prep, "Target_Encoder.pkl")
    joblib.dump(feature_encoders_prep, "Feature_Encoders.pkl")
    joblib.dump(feature_columns_prep, "Feature_Columns.pkl")
    st.success("Semua model dan encoder berhasil diekspor!")

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(hasil_df["Model"], hasil_df["Accuracy"], color='#26A69A')
    ax.set_ylabel("Accuracy Score")
    plt.xticks(rotation=15)
    st.pyplot(fig)

# --- PREDIKSI MANUAL ---
elif menu == "Prediksi Manual":
    st.header("🔍 Prediksi Manual & Unggah Data")
    
    required_files = ["Semua_Model.pkl", "Scaler.pkl", "Target_Encoder.pkl", "Feature_Encoders.pkl", "Feature_Columns.pkl"]
    if not all(os.path.exists(file) for file in required_files):
        st.error("Silakan jalankan menu **Training** terlebih dahulu agar model dan komponen siap digunakan.")
        st.stop()

    # Load semua komponen model
    models = joblib.load("Semua_Model.pkl")
    scaler = joblib.load("Scaler.pkl")
    target_encoder = joblib.load("Target_Encoder.pkl")
    feature_encoders = joblib.load("Feature_Encoders.pkl")
    feature_columns = joblib.load("Feature_Columns.pkl")

    model_name = st.selectbox("Pilih Model untuk Prediksi", list(models.keys()))
    model = models[model_name]
    st.write("---")
    
    tab_form, tab_upload = st.tabs(["📝 Input Form Mandiri", "📁 Upload File CSV"])

    # TAB 1: INPUT FORM MANDIRI
    with tab_form:
        st.subheader("Masukkan Data Pengguna Baru:")
        
        label_mapping = {
            "Age": "Masukkan Usia / Umur", "Gender": "Pilih Jenis Kelamin",
            "Academic_Level": "Pilih Jenjang / Tingkat Pendidikan", "Country": "Pilih Negara Asal",
            "Avg_Daily_Usage_Hours": "Masukkan Rata-rata Durasi Penggunaan Harian (Jam)",
            "Most_Used_Platform": "Pilih Platform yang Paling Sering Digunakan",
            "Affects_Academic_Performance": "Apakah Memengaruhi Performa Akademik?",
            "Sleep_Hours_Per_Night": "Masukkan Durasi Tidur per Malam (Jam)",
            "Mental_Health_Score": "Masukkan Skor Kesehatan Mental", "Physical_Activity": "Pilih Tingkat Aktivitas Fisik"
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
                display_label = label_mapping.get(col_name, f"Masukkan {col_name.replace('_', ' ')}")
                
                if col_name in feature_encoders:
                    labels_kategori = list(feature_encoders[col_name].classes_)
                    translated_options = [option_mapping.get(opt, opt) for opt in labels_kategori]
                    ui_inputs[col_name] = form_col.selectbox(display_label, options=translated_options, key=f"ui_{col_name}")
                else:
                    min_val = float(df[col_name].min()) if col_name in df.columns else 0.0
                    max_val = float(df[col_name].max()) if col_name in df.columns else 100.0
                    mean_val = float(df[col_name].mean()) if col_name in df.columns else 0.0
                    ui_inputs[col_name] = form_col.number_input(display_label, min_value=min_val, max_value=max_val, value=mean_val, key=f"ui_{col_name}")
            
            submitted = st.form_submit_button("🔮 Lakukan Prediksi Form")

        if submitted:
            input_data = {}
            for col_name in feature_columns:
                val = ui_inputs[col_name]
                input_data[col_name] = reverse_option_mapping.get(val, val)

            input_df = pd.DataFrame([input_data])
            
            for col, encoder in feature_encoders.items():
                try:
                    val_str = str(input_df.at[0, col]).strip()
                    matched_class = next((c for c in encoder.classes_ if str(c).lower() == val_str.lower()), encoder.classes_[0])
                    input_df[col] = encoder.transform([matched_class])
                except Exception as e:
                    st.error(f"Terjadi kesalahan saat encoding fitur '{col}': {e}")
                    st.stop()
            
            input_df = input_df[feature_columns]
            input_scaled = scaler.transform(input_df)
            
            prediksi_angka = model.predict(input_scaled)
            hasil_label = target_encoder.inverse_transform(prediksi_angka)[0]
            
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

    # TAB 2: UPLOAD FILE CSV
    with tab_upload:
        st.subheader("Predict via CSV Upload")
        st.info("Sistem akan otomatis mendeteksi, menyelaraskan, dan menyesuaikan kolom dataset yang Anda unggah.")
        
        uploaded_file = st.file_uploader("Upload Dataset CSV Anda", type=["csv"], key="manual_upload_csv")
        
        if uploaded_file is not None:
            user_data = pd.read_csv(uploaded_file)
            st.write("📄 **Preview Data yang Di-upload:**")
            st.dataframe(user_data.head(), use_container_width=True)
            
            csv_labels, aligned_scaled = align_and_predict(user_data, feature_columns, feature_encoders, scaler, model, target_encoder, model_name)

            final_result = user_data.copy()
            final_result[f"Hasil Prediksi ({model_name})"] = csv_labels

            # Mengecek apakah data memiliki kolom target untuk memunculkan evaluasi kinerja otomatis
            has_target = False
            target_candidates = ["Addiction_Level", "addiction_level", "Addiction Level", "Tingkat Kecanduan"]
            for candidate in target_candidates:
                if candidate in user_data.columns:
                    has_target = True
                    actual_labels = user_data[candidate].copy()
                    break

            st.write("---")
            if has_target:
                st.subheader("📈 Metrik Performa Hasil Upload")
                acc = accuracy_score(actual_labels.astype(str), csv_labels.astype(str))
                pre = precision_score(actual_labels.astype(str), csv_labels.astype(str), average="weighted", zero_division=0)
                rec = recall_score(actual_labels.astype(str), csv_labels.astype(str), average="weighted", zero_division=0)
                f1 = f1_score(actual_labels.astype(str), csv_labels.astype(str), average="weighted", zero_division=0)
                
                metrics_df = pd.DataFrame([[model_name, acc, pre, rec, f1]], columns=["Model", "Accuracy", "Precision", "Recall", "F1 Score"])
                st.dataframe(metrics_df.style.format({"Accuracy": "{:.2%}", "Precision": "{:.2%}", "Recall": "{:.2%}", "F1 Score": "{:.2%}"}), use_container_width=True)

            st.subheader("📊 Hasil Prediksi Dataset Upload")
            st.dataframe(final_result, use_container_width=True)

            csv_output = final_result.to_csv(index=False).encode("utf-8")
            st.download_button(
                label=f"⬇️ Download Hasil Prediksi {model_name} (CSV)",
                data=csv_output,
                file_name=f"hasil_prediksi_{model_name.lower().replace(' ', '_')}.csv",
                mime="text/csv"
            )

# Optional Lottie animation support
try:
    import requests
    import importlib
    streamlit_lottie = importlib.import_module("streamlit_lottie")
    st_lottie = getattr(streamlit_lottie, "st_lottie", None)
except (ImportError, ModuleNotFoundError):
    requests = None
    st_lottie = None


def load_lottieurl(url):
    if requests is None:
        return None
    try:
        r = requests.get(url)
        if r.status_code != 200:
            return None
        return r.json()
    except Exception:
        return None

# Contoh animasi bertema teknologi/analisis
lottie_coding = load_lottieurl("https://assets5.lottiefiles.com/packages/lf20_fcfjwiyb.json")
if st_lottie is not None and lottie_coding is not None:
    st_lottie(lottie_coding, height=300, key="coding")
