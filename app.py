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

# ==========================================================
# TEMA / CSS KUSTOM — "Editorial Portfolio Gelap"
# Terinspirasi nuansa: latar teal gelap, aksen olive/khaki,
# tipografi tebal huruf kapital, tombol outline minimalis.
# ==========================================================
CUSTOM_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Archivo:wght@500;600;700;800;900&family=Poppins:wght@300;400;500;600&display=swap');

html, body, [class*="css"] {
    font-family: 'Poppins', sans-serif;
}

/* Latar belakang utama: teal gelap ala editorial portfolio */
.stApp {
    background: #0C3236;
    color: #EDEAD9;
}

header[data-testid="stHeader"] {
    background: transparent;
}

/* ==================== HERO / JUDUL ==================== */
.hero-box {
    background: linear-gradient(180deg, #12474D 0%, #0C3236 100%);
    padding: 3rem 2.8rem 2.4rem 2.8rem;
    border-radius: 4px;
    margin-bottom: 1.8rem;
    border: 1px solid rgba(201, 203, 126, 0.25);
    position: relative;
}
.hero-nav {
    display: flex;
    justify-content: space-between;
    align-items: center;
    padding-bottom: 1.6rem;
    margin-bottom: 1.8rem;
    border-bottom: 1px solid rgba(237,234,217,0.15);
    font-family: 'Archivo', sans-serif;
    font-weight: 600;
    letter-spacing: 1px;
    color: #C9CB7E;
    font-size: 0.8rem;
    text-transform: uppercase;
}
.hero-box h1 {
    font-family: 'Archivo', sans-serif;
    color: #C9CB7E;
    font-weight: 900;
    font-size: 2.6rem;
    margin: 0;
    letter-spacing: 1px;
    text-transform: uppercase;
    line-height: 1.15;
}
.hero-box p {
    color: #B9C4C2;
    font-size: 1rem;
    margin-top: 0.9rem;
    font-weight: 300;
    max-width: 640px;
}
.hero-btn {
    display: inline-block;
    margin-top: 1.6rem;
    padding: 0.65rem 1.6rem;
    border: 1px solid #C9CB7E;
    border-radius: 999px;
    color: #C9CB7E;
    font-family: 'Archivo', sans-serif;
    font-size: 0.78rem;
    letter-spacing: 1.5px;
    text-transform: uppercase;
    font-weight: 600;
}

/* ==================== SIDEBAR - GAYA NAVBAR OLIVE ==================== */
section[data-testid="stSidebar"] {
    background: #A9A87C;
    border-right: 1px solid rgba(0,0,0,0.15);
}
section[data-testid="stSidebar"] * {
    color: #17312F !important;
    font-family: 'Archivo', sans-serif !important;
}
section[data-testid="stSidebar"] .stSelectbox label {
    font-weight: 700 !important;
    font-size: 0.85rem;
    text-transform: uppercase;
    letter-spacing: 1px;
}
section[data-testid="stSidebar"] div[data-baseweb="select"] > div {
    background-color: rgba(255,255,255,0.35) !important;
    border-radius: 4px !important;
    border: 1px solid rgba(23,49,47,0.4) !important;
}

/* ==================== KARTU KONTEN ==================== */
.card {
    background: #12474D;
    border-radius: 4px;
    padding: 1.6rem 1.8rem;
    margin-bottom: 1.3rem;
    border: 1px solid rgba(201, 203, 126, 0.25);
    color: #EDEAD9;
}

/* ==================== TEKS UMUM ==================== */
p, li, span, div, label {
    color: #EDEAD9;
}
.stMarkdown, .stMarkdown p {
    color: #EDEAD9 !important;
}

/* ==================== HEADER / SUBHEADER ==================== */
h1, h2, h3 {
    font-family: 'Archivo', sans-serif;
    font-weight: 800;
    color: #EDEAD9;
    text-transform: uppercase;
    letter-spacing: 0.5px;
}
h2 { color: #C9CB7E; }
h3 { color: #9FD0C8; }

/* ==================== TOMBOL - OUTLINE MINIMALIS ==================== */
.stButton > button, .stFormSubmitButton > button, .stDownloadButton > button {
    background: transparent;
    color: #C9CB7E;
    border: 1px solid #C9CB7E;
    border-radius: 999px;
    padding: 0.55rem 1.6rem;
    font-family: 'Archivo', sans-serif;
    font-weight: 600;
    letter-spacing: 1px;
    text-transform: uppercase;
    font-size: 0.8rem;
    transition: all 0.15s ease;
}
.stButton > button:hover, .stFormSubmitButton > button:hover, .stDownloadButton > button:hover {
    background: #C9CB7E;
    color: #0C3236;
}

/* ==================== METRIC ==================== */
div[data-testid="stMetric"] {
    background: #12474D;
    border-radius: 4px;
    padding: 1rem 1.2rem;
    border: 1px solid rgba(201, 203, 126, 0.25);
}
div[data-testid="stMetric"] label, div[data-testid="stMetric"] div {
    color: #EDEAD9 !important;
}

/* ==================== TABS ==================== */
button[data-baseweb="tab"] {
    font-family: 'Archivo', sans-serif !important;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 1px;
    color: #B9C4C2 !important;
}
div[data-baseweb="tab-highlight"] {
    background-color: #C9CB7E !important;
}

/* ==================== ALERT & DATAFRAME ==================== */
div[data-testid="stAlert"] {
    border-radius: 4px;
    background: #12474D !important;
}
div[data-testid="stDataFrame"] {
    border-radius: 4px;
    overflow: hidden;
    border: 1px solid rgba(201, 203, 126, 0.25);
}

/* ==================== FITUR GRID (Home) ==================== */
.feature-grid {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(220px, 1fr));
    gap: 1rem;
    margin-top: 1rem;
}
.feature-item {
    background: #12474D;
    border-radius: 4px;
    padding: 1.1rem 1.2rem;
    border-left: 3px solid #C9CB7E;
    font-family: 'Poppins', sans-serif;
    color: #EDEAD9;
}
.feature-item b {
    color: #C9CB7E;
    font-family: 'Archivo', sans-serif;
    text-transform: uppercase;
    letter-spacing: 0.5px;
}

/* ==================== BADGE MODEL (Home) ==================== */
.badge-row { display:flex; flex-wrap:wrap; gap:0.5rem; margin: 0.8rem 0 1.2rem 0; }
.badge {
    background: transparent;
    color: #C9CB7E;
    font-weight:600; font-size:0.75rem;
    font-family: 'Archivo', sans-serif;
    text-transform: uppercase;
    letter-spacing: 1px;
    padding: 0.4rem 1rem; border-radius: 999px;
    border: 1px solid #C9CB7E;
}
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

st.markdown(
    """
    <div class="hero-box">
        <div class="hero-nav">
            <span>P. &nbsp; Home &nbsp; Editorial &nbsp; About</span>
            <span>Contact</span>
        </div>
        <h1>Prediksi Tingkat<br>Kecanduan Media Sosial</h1>
        <p>Aplikasi Machine Learning untuk menganalisis, melatih model, dan memprediksi tingkat kecanduan media sosial berdasarkan data & kebiasaan penggunanya.</p>
        <span class="hero-btn">Lihat Semua Fitur ↗</span>
    </div>
    """,
    unsafe_allow_html=True
)

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
    col_img, col_text = st.columns([1, 1.3], gap="large")

    with col_img:
        st.image(
            "https://i0.wp.com/metrum.co.id/kanal/uploads/2022/01/pict-19-kecanduan-medsos.jpg?w=800&ssl=1",
            use_container_width=True
        )

    with col_text:
        st.markdown(
            """
            <div class="card">
                <h3 style="margin-top:0;">👋 Selamat Datang!</h3>
                <p style="font-family:'Poppins', sans-serif; font-size:1.02rem; color:#EDEAD9; font-weight:300;">
                    Aplikasi ini memakai <b>Machine Learning</b> untuk membantu memprediksi tingkat
                    kecanduan media sosial seseorang berdasarkan kebiasaan sehari-hari, kondisi
                    akademik, hingga kesehatan mental. Yuk eksplor datanya! 🚀
                </p>
            </div>
            """,
            unsafe_allow_html=True
        )

        st.markdown("**🤖 Model yang Digunakan:**")
        st.markdown(
            """
            <div class="badge-row">
                <span class="badge">Logistic Regression</span>
                <span class="badge">Decision Tree</span>
                <span class="badge">Random Forest</span>
                <span class="badge">KNN</span>
                <span class="badge">Naive Bayes</span>
                <span class="badge">SVM</span>
                <span class="badge">XGBoost</span>
            </div>
            """,
            unsafe_allow_html=True
        )

    st.markdown("### ✨ Fitur Utama")
    st.markdown(
        """
        <div class="feature-grid">
            <div class="feature-item">📊 <b>Dataset</b><br>Melihat ringkasan data training.</div>
            <div class="feature-item">📈 <b>EDA</b><br>Exploratory Data Analysis & visualisasi korelasi.</div>
            <div class="feature-item">⚙️ <b>Preprocessing</b><br>Pembersihan data & pembagian dataset.</div>
            <div class="feature-item">🤖 <b>Training Model</b><br>Melatih data ke semua algoritma sekaligus.</div>
            <div class="feature-item">🔍 <b>Prediksi Manual</b><br>Input data mandiri via form.</div>
            <div class="feature-item">📁 <b>Prediksi Dataset Upload</b><br>Upload CSV untuk prediksi massal.</div>
        </div>
        """,
        unsafe_allow_html=True
    )

# --- DATASET ---
elif menu == "Dataset":
    st.header("📊 Dataset Training")
    st.info(f"Dataset training yang digunakan: {TRAINING_DATASET}")
    
    st.markdown("""
    Halaman ini menampilkan seluruh isi dataset mentah yang digunakan sebagai dasar pembuatan model *Machine Learning*. 
    Tabel di bawah memuat seluruh atribut responden mulai dari informasi demografis, kebiasaan penggunaan perangkat, hingga label tingkat kecanduan media sosial (*Addiction Level*).
    """)

    st.dataframe(df, use_container_width=True)

    col1, col2, col3 = st.columns(3)
    col1.metric("Jumlah Baris", df.shape[0])
    col2.metric("Jumlah Kolom", df.shape[1])
    
    st.subheader("Statistik Deskriptif")
    st.markdown("""
    Tabel ringkasan statistik berikut menyajikan nilai-nilai parameter numerik (seperti *mean*, *std*, *min*, dan *max*) serta sebaran frekuensi untuk variabel kategorikal dari data responden.
    """)
    st.write(df.describe(include="all"))

    st.subheader("Tipe Data")
    st.markdown("Berikut adalah daftar variabel/kolom yang ada dalam dataset beserta tipe datanya masing-masing:")
    st.write(df.dtypes.astype(str))

# --- EDA ---
elif menu == "EDA":
    st.header("📈 Exploratory Data Analysis")
    
    st.markdown("""
    Menu Exploratory Data Analysis (EDA) bertujuan untuk menganalisis karakteristik data melalui eksplorasi visual dan ringkasan struktur. 
    Langkah ini membantu dalam memahami pola, mengecek keberadaan data yang hilang (*missing values*), serta melihat korelasi antar variabel.
    """)

    st.subheader("5 Data Pertama")
    st.write("Berikut adalah sampel 5 baris pertama dari dataset untuk memberikan gambaran cepat struktur data:")
    st.dataframe(df.head())

    st.subheader("Informasi Dataset")
    st.write("Tabel ringkasan struktur kolom, jenis tipe data, jumlah *missing value*, serta jumlah nilai unik (*unique values*) dari dataset:")
    info = pd.DataFrame({
        "Kolom": df.columns,
        "Tipe Data": df.dtypes.astype(str),
        "Missing Value": df.isnull().sum(),
        "Unique": df.nunique()
    })
    st.dataframe(info, use_container_width=True)

    if "Addiction_Level" in df.columns:
        st.subheader("Distribusi Addiction Level")
        st.write("Grafik batang di bawah menunjukkan sebaran frekuensi label tingkat kecanduan (*Addiction Level*) yang menjadi variabel target dalam model prediksi:")
        fig, ax = plt.subplots(figsize=(6, 3))
        df["Addiction_Level"].value_counts().plot(kind="bar", ax=ax, color='#1E88E5')
        ax.set_xlabel("Tingkat Kecanduan")
        ax.set_ylabel("Jumlah")
        st.pyplot(fig)

    numeric = df.select_dtypes(include=np.number)
    if not numeric.empty:
        st.subheader("Distribusi Variabel Numerik")
        st.write("Visualisasi histogram untuk melihat bagaimana sebaran data pada tiap-tiap kolom numerik:")
        for col in numeric.columns:
            fig, ax = plt.subplots(figsize=(5, 2.5))
            ax.hist(numeric[col], bins=20, color='#4CAF50')
            ax.set_title(f"Distribusi {col}")
            st.pyplot(fig)

        st.subheader("Correlation Matrix")
        st.write("Matriks korelasi untuk mengukur derajat hubungan linier antar variabel numerik. Nilai mendekati 1 atau -1 menunjukkan korelasi yang semakin kuat:")
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
    
    st.markdown("""
    Pada tahap *preprocessing*, data mentah dibersihkan dan ditransformasi agar siap diproses oleh algoritma *Machine Learning*. 
    Proses ini mencakup penghapusan data duplikat, eliminasi kolom yang tidak relevan (seperti *Student_ID*), serta encoding pada variabel kategorikal.
    """)

    st.write(f"Dataset awal: **{df.shape[0]} baris, {df.shape[1]} kolom**")

    X, y, target_encoder, feature_encoders, scaler, feature_columns = prepare_training_data(df)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    st.success("Preprocessing berhasil dilakukan!")

    st.markdown("""
    Data telah dibagi menjadi 2 bagian utama menggunakan teknik *Stratified Train-Test Split* (rasio 80:20):
    * **Data Training (80%)**: Digunakan untuk melatih algoritma *Machine Learning*.
    * **Data Testing (20%)**: Digunakan untuk menguji dan mengevaluasi akurasi model yang telah dilatih.
    """)

    st.write(f"✔ Jumlah Data Training: {X_train.shape[0]} sampel")
    st.write(f"✔ Jumlah Data Testing: {X_test.shape[0]} sampel")
    st.write("Target Classes:", list(target_encoder.classes_))

# --- TRAINING ---
elif menu == "Training":
    st.header("🤖 Training Model")
    
    st.markdown("""
    Halaman ini melakukan pelatihan (*training*) secara bersamaan pada beberapa algoritma *Machine Learning* menggunakan data yang telah di-*preprocessing*. 
    Performa tiap algoritma kemudian dievaluasi menggunakan data testing berdasarkan metrik **Accuracy**, **Precision**, **Recall**, dan **F1 Score**.
    """)

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

    st.subheader("📊 Tabel Perbandingan Performa Model")
    st.markdown("""
    Tabel di bawah mengurutkan algoritma berdasarkan tingkat **Accuracy** tertinggi. Model dengan performa terbaik disiapkan secara otomatis sebagai model rujukan utama.
    """)
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
    st.subheader("📈 Grafik Akurasi Model")
    st.markdown("Perbandingan performa tingkat akurasi antar algoritma dalam bentuk grafik batang:")
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(hasil_df["Model"], hasil_df["Accuracy"], color='#26A69A')
    ax.set_ylabel("Accuracy Score")
    plt.xticks(rotation=15)
    st.pyplot(fig)

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
    
    tab_form, tab_upload = st.tabs(["📝 Input Form Mandiri", "📁 Upload File CSV"])

    # ------------------------------------------------------
    # TAB 1: INPUT FORM MANDIRI
    # ------------------------------------------------------
    with tab_form:
        st.subheader("Masukkan Data Pengguna Baru:")
        input_data = {}
        
        # 1. Kamus translasi label visual (Judul Disesuaikan)
        label_mapping = {
            "Age": "Masukkan Usia / Umur",
            "Gender": "Pilih Jenis Kelamin",
            "Academic_Level": "Pilih Jenjang / Tingkat Pendidikan",
            "Country": "Pilih Negara Asal",
            "Avg_Daily_Usage_Hours": "Masukkan rata-rata penggunaan handphone",
            "Most_Used_Platform": "Pilih Platform yang Paling Sering Digunakan",
            "Affects_Academic_Performance": "Apakah Memengaruhi Performa Akademik?",
            "Sleep_Hours_Per_Night": "Masukkan Durasi Tidur per Malam (Jam)",
            "Mental_Health_Score": "Masukkan Skor Kesehatan Mental (1-5 rendah, 6-10 tinggi)",
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
            ui_inputs = {}
            
            for idx, col_name in enumerate(feature_columns):
                form_col = col1 if idx % 2 == 0 else col2
                display_label = label_mapping.get(col_name, f"Masukkan {col_name.replace('_', ' ')}")
                
                # Input untuk Fitur Kategorikal
                if col_name in feature_encoders:
                    labels_kategori = list(feature_encoders[col_name].classes_)
                    translated_options = [option_mapping.get(opt, opt) for opt in labels_kategori]
                    
                    ui_inputs[col_name] = form_col.selectbox(
                        display_label, 
                        options=translated_options,
                        key=f"ui_{col_name}"
                    )
                # Input Numerik Bebas (Tanpa Batasan Min/Max dan Bebas Koma)
                else:
                    # Menghitung default value sesuai rata-rata data training jika ada
                    default_val = int(df[col_name].mean()) if col_name in df.columns else 0
                    
                    ui_inputs[col_name] = form_col.number_input(
                        display_label, 
                        value=default_val,
                        step=1,
                        key=f"ui_{col_name}"
                    )
            
            submitted = st.form_submit_button("🔮 Lakukan Prediksi Form")

        # PROSES EKSEKUSI PREDIKSI SETELAH TOMBOL DIKLIK
        if submitted:
            for col_name in feature_columns:
                val = ui_inputs[col_name]
                if col_name in feature_encoders:
                    input_data[col_name] = reverse_option_mapping.get(val, val)
                else:
                    # Konversi angka berapapun ke float agar algoritma dapat memprosesnya
                    input_data[col_name] = float(val)

            # Ubah ke DataFrame
            input_df = pd.DataFrame([input_data])
            
            # Lakukan Encoding menggunakan encoder bawaan model
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
    # TAB 2: UPLOAD FILE CSV
    # ------------------------------------------------------
    with tab_upload:
        st.subheader("Predict via CSV Upload")
        st.info("Sistem akan otomatis mendeteksi, menyelaraskan, dan menyesuaikan kolom dataset yang Anda unggah.")
        
        uploaded_file = st.file_uploader("Upload Dataset CSV Anda", type=["csv"], key="manual_upload_csv")
        
        if uploaded_file is not None:
            user_data = pd.read_csv(uploaded_file)
            st.write("📄 **Preview Data yang Di-upload:**")
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
            final_result[f"Hasil Prediksi ({model_name})"] = csv_labels

            st.write("---")
            st.subheader("📊 Hasil Prediksi Dataset Upload")
            st.dataframe(final_result, use_container_width=True)

            csv_output = final_result.to_csv(index=False).encode("utf-8")
            st.download_button(
                label=f"⬇️ Download Hasil Prediksi {model_name} (CSV)",
                data=csv_output,
                file_name=f"hasil_prediksi_{model_name.lower().replace(' ', '_')}.csv",
                mime="text/csv"
            )

# ==========================================================
# PREDIKSI DATASET UPLOAD
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

    required_files = ["Semua_Model.pkl", "Scaler.pkl", "Target_Encoder.pkl", "Feature_Encoders.pkl", "Feature_Columns.pkl"]
    if not all(os.path.exists(file) for file in required_files):
        st.error("Model/Komponen scaler belum tersedia. Silakan jalankan menu Training terlebih dahulu.")
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
                    default_cat = feature_encoders[col].classes_[0]
                    prediction_data[col] = default_cat
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
    
    st.write("---")
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
