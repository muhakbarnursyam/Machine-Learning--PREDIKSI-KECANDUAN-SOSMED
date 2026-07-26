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
# TEMA / CSS KUSTOM — "Biru-Putih Profesional" (modern, tidak kaku)
# ==========================================================
CUSTOM_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&family=Inter:wght@300;400;500;600&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}

/* Latar belakang utama: putih kebiruan lembut dengan aksen bulat */
.stApp {
    background:
        radial-gradient(circle at 6% 8%, rgba(30,136,229,0.08) 0, transparent 22%),
        radial-gradient(circle at 96% 18%, rgba(66,165,245,0.08) 0, transparent 24%),
        radial-gradient(circle at 90% 92%, rgba(21,101,192,0.06) 0, transparent 24%),
        #F5F8FC;
    color: #1E2A3A;
}

header[data-testid="stHeader"] {
    background: transparent;
}

/* ==================== PERBAIKAN IKON COLLAPSE SIDEBAR ====================
   Membuat ikon buka/tutup sidebar selalu terlihat jelas baik saat tema
   Streamlit disetel Light maupun Dark. */
[data-testid="collapsedControl"],
[data-testid="stSidebarCollapsedControl"],
button[data-testid="baseButton-headerNoPadding"] {
    background: #1565C0 !important;
    border-radius: 10px !important;
    box-shadow: 0 4px 10px -2px rgba(21,101,192,0.5);
}
[data-testid="collapsedControl"] svg,
[data-testid="stSidebarCollapsedControl"] svg,
button[data-testid="baseButton-headerNoPadding"] svg {
    fill: #FFFFFF !important;
    color: #FFFFFF !important;
}
[data-testid="collapsedControl"] span,
[data-testid="stSidebarCollapsedControl"] span {
    color: #FFFFFF !important;
    font-size: 1.3rem !important;
}
[data-testid="stSidebarCollapseButton"] button,
[data-testid="stSidebarCollapseButton"] svg {
    color: #FFFFFF !important;
    fill: #FFFFFF !important;
}
[data-testid="stSidebarCollapseButton"] button {
    background: rgba(255,255,255,0.18) !important;
    border-radius: 8px !important;
}

/* Jika font ikon Material Symbols gagal dimuat, browser menampilkan nama
   ikonnya sebagai teks mentah (mis. "keyboard_double_arrow_left"). Sembunyikan
   teks tersebut dan gantikan dengan tanda panah sederhana agar tombol tetap
   jelas menandakan kontrol buka/tutup sidebar. */
[data-testid="collapsedControl"] [data-testid="stIconMaterial"],
[data-testid="stSidebarCollapsedControl"] [data-testid="stIconMaterial"],
[data-testid="stSidebarCollapseButton"] [data-testid="stIconMaterial"] {
    font-size: 0 !important;
    color: transparent !important;
    position: relative;
    display: inline-block;
    width: 1.3rem;
    height: 1.3rem;
}
[data-testid="collapsedControl"] [data-testid="stIconMaterial"]::after,
[data-testid="stSidebarCollapsedControl"] [data-testid="stIconMaterial"]::after {
    content: "»";
    position: absolute;
    left: 50%; top: 50%;
    transform: translate(-50%, -55%);
    font-size: 1.4rem !important;
    line-height: 1;
    color: #FFFFFF !important;
    font-family: 'Plus Jakarta Sans', sans-serif !important;
}
[data-testid="stSidebarCollapseButton"] [data-testid="stIconMaterial"]::after {
    content: "«";
    position: absolute;
    left: 50%; top: 50%;
    transform: translate(-50%, -55%);
    font-size: 1.4rem !important;
    line-height: 1;
    color: #FFFFFF !important;
    font-family: 'Plus Jakarta Sans', sans-serif !important;
}

/* ==================== HERO / JUDUL ==================== */
.hero-box {
    background: linear-gradient(120deg, #0D47A1 0%, #1565C0 45%, #42A5F5 100%);
    padding: 2.6rem 2.8rem;
    border-radius: 24px;
    margin-bottom: 1.6rem;
    box-shadow: 0 18px 38px -14px rgba(13,71,161,0.45);
    position: relative;
    overflow: hidden;
}
.hero-box::after {
    content: "";
    position: absolute;
    top: -70px; right: -70px;
    width: 220px; height: 220px;
    background: rgba(255,255,255,0.10);
    border-radius: 50%;
}
.hero-box::before {
    content: "";
    position: absolute;
    bottom: -90px; left: -40px;
    width: 240px; height: 240px;
    background: rgba(255,255,255,0.07);
    border-radius: 50%;
}
.hero-box h1 {
    font-family: 'Plus Jakarta Sans', sans-serif;
    color: #FFFFFF;
    font-weight: 800;
    font-size: 2.3rem;
    margin: 0;
    line-height: 1.2;
    letter-spacing: -0.5px;
}
.hero-box p {
    color: rgba(255,255,255,0.92);
    font-size: 1rem;
    margin-top: 0.8rem;
    font-weight: 400;
    max-width: 640px;
}
.hero-btn-wrap { margin-top: 1.4rem; }
.hero-btn-wrap .stButton > button {
    background: #FFFFFF !important;
    color: #0D47A1 !important;
    border: none !important;
    font-weight: 700 !important;
    box-shadow: 0 8px 18px -6px rgba(0,0,0,0.25) !important;
}
.hero-btn-wrap .stButton > button:hover {
    background: #E3F2FD !important;
    color: #0D47A1 !important;
}

/* ==================== SIDEBAR ==================== */
section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0D47A1 0%, #1565C0 100%);
    border-right: 1px solid rgba(0,0,0,0.06);
}
section[data-testid="stSidebar"] * {
    color: #FFFFFF !important;
    font-family: 'Plus Jakarta Sans', sans-serif !important;
}
section[data-testid="stSidebar"] .stSelectbox label {
    font-weight: 700 !important;
    font-size: 0.85rem;
    text-transform: uppercase;
    letter-spacing: 1px;
}
section[data-testid="stSidebar"] div[data-baseweb="select"] > div {
    background-color: rgba(255,255,255,0.16) !important;
    border-radius: 12px !important;
    border: 1.5px solid rgba(255,255,255,0.35) !important;
}

/* ==================== KARTU KONTEN ==================== */
.card {
    background: #FFFFFF;
    border-radius: 20px;
    padding: 1.6rem 1.8rem;
    margin-bottom: 1.3rem;
    box-shadow: 0 10px 26px -14px rgba(21,101,192,0.22);
    border: 1px solid rgba(21,101,192,0.08);
    color: #1E2A3A;
}

/* ==================== HEADER / SUBHEADER ==================== */
h1, h2, h3 {
    font-family: 'Plus Jakarta Sans', sans-serif;
    font-weight: 700;
    color: #14213D;
}
h2 { color: #1565C0; }
h3 { color: #0D47A1; }

/* ==================== TOMBOL ==================== */
.stButton > button, .stFormSubmitButton > button, .stDownloadButton > button {
    background: linear-gradient(120deg, #1565C0, #42A5F5);
    color: #FFFFFF;
    border: none;
    border-radius: 999px;
    padding: 0.6rem 1.7rem;
    font-family: 'Plus Jakarta Sans', sans-serif;
    font-weight: 700;
    box-shadow: 0 8px 18px -6px rgba(21,101,192,0.45);
    transition: transform 0.15s ease, box-shadow 0.15s ease;
}
.stButton > button:hover, .stFormSubmitButton > button:hover, .stDownloadButton > button:hover {
    transform: translateY(-2px);
    box-shadow: 0 12px 22px -6px rgba(21,101,192,0.55);
    color: #FFFFFF;
}

/* ==================== METRIC ==================== */
div[data-testid="stMetric"] {
    background: linear-gradient(135deg, #FFFFFF, #EAF3FC);
    border-radius: 16px;
    padding: 1rem 1.2rem;
    border: 1px solid rgba(21,101,192,0.15);
    box-shadow: 0 8px 18px -12px rgba(21,101,192,0.25);
}

/* ==================== TABS ==================== */
button[data-baseweb="tab"] {
    font-family: 'Plus Jakarta Sans', sans-serif !important;
    font-weight: 600;
    border-radius: 12px 12px 0 0 !important;
}
div[data-baseweb="tab-highlight"] {
    background-color: #1565C0 !important;
}

/* ==================== ALERT & DATAFRAME ==================== */
div[data-testid="stAlert"] {
    border-radius: 14px;
}
div[data-testid="stDataFrame"] {
    border-radius: 14px;
    overflow: hidden;
    border: 1px solid rgba(21,101,192,0.12);
}

/* ==================== FITUR GRID (Home) ==================== */
.feature-grid {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(220px, 1fr));
    gap: 1rem;
    margin-top: 1rem;
}
.feature-item {
    background: #FFFFFF;
    border-radius: 16px;
    padding: 1.1rem 1.2rem;
    border-left: 5px solid #1565C0;
    box-shadow: 0 8px 18px -14px rgba(21,101,192,0.3);
    font-family: 'Inter', sans-serif;
    color: #1E2A3A;
}
.feature-item b {
    color: #0D47A1;
    font-family: 'Plus Jakarta Sans', sans-serif;
}

/* ==================== BADGE MODEL (Home) ==================== */
.badge-row { display:flex; flex-wrap:wrap; gap:0.5rem; margin: 0.8rem 0 1.2rem 0; }
.badge {
    background: linear-gradient(120deg, #1565C0, #42A5F5);
    color: #FFFFFF;
    font-weight: 600; font-size: 0.82rem;
    font-family: 'Plus Jakarta Sans', sans-serif;
    padding: 0.4rem 1rem; border-radius: 999px;
}

/* ==================== MENU SIDEBAR DENGAN IKON ==================== */
.sidebar-menu-title {
    display: flex;
    align-items: center;
    gap: 0.5rem;
    font-family: 'Plus Jakarta Sans', sans-serif;
    font-weight: 800;
    font-size: 1.05rem;
    color: #FFFFFF;
    margin-bottom: 0.6rem;
}
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

# ==========================================================
# DATA MENU (didefinisikan lebih awal agar bisa dipakai bersama
# oleh kartu fitur "Lihat Semua Fitur" dan sidebar navigasi)
# ==========================================================
MENU_ICONS = {
    "Home": "🏠",
    "Dataset": "📊",
    "EDA": "📈",
    "Preprocessing": "⚙️",
    "Training": "🤖",
    "Prediksi Manual": "🔍",
    "Prediksi Dataset Upload": "📁",
}
MENU_DESC = {
    "Dataset": "Melihat ringkasan data training.",
    "EDA": "Exploratory Data Analysis & visualisasi korelasi.",
    "Preprocessing": "Pembersihan data & pembagian dataset.",
    "Training": "Melatih data ke semua algoritma sekaligus.",
    "Prediksi Manual": "Input data mandiri via form.",
    "Prediksi Dataset Upload": "Upload CSV untuk prediksi massal.",
}
display_to_menu = {f"{icon}  {label}": label for label, icon in MENU_ICONS.items()}
menu_to_display = {label: disp for disp, label in display_to_menu.items()}

SELECTBOX_KEY = "sb_menu_display"
if SELECTBOX_KEY not in st.session_state:
    st.session_state[SELECTBOX_KEY] = menu_to_display["Home"]

st.markdown(
    """
    <div class="hero-box">
        <h1>🧠 Prediksi Tingkat Kecanduan Media Sosial</h1>
        <p>Aplikasi Machine Learning untuk menganalisis, melatih model, dan memprediksi tingkat kecanduan media sosial berdasarkan data & kebiasaan penggunanya.</p>
    </div>
    """,
    unsafe_allow_html=True
)

# Tombol "Lihat Semua Fitur" — nyata & bisa diklik (bukan sekadar hiasan),
# menampilkan ringkasan fitur di halaman manapun. Setiap kartu di bawahnya
# juga bisa langsung diklik untuk berpindah ke menu tersebut.
if "show_fitur" not in st.session_state:
    st.session_state.show_fitur = False

st.markdown('<div class="hero-btn-wrap">', unsafe_allow_html=True)
if st.button("✨ Lihat Semua Fitur ↗", key="btn_lihat_fitur"):
    st.session_state.show_fitur = not st.session_state.show_fitur
st.markdown('</div>', unsafe_allow_html=True)

if st.session_state.show_fitur:
    fitur_list = [label for label in MENU_ICONS if label != "Home"]
    kolom_per_baris = 3
    for i in range(0, len(fitur_list), kolom_per_baris):
        baris_items = fitur_list[i:i + kolom_per_baris]
        cols = st.columns(len(baris_items))
        for col, label in zip(cols, baris_items):
            with col:
                with st.container(border=True):
                    st.markdown(f"**{MENU_ICONS[label]} {label}**")
                    st.caption(MENU_DESC[label])
                    if st.button("Buka →", key=f"card_buka_{label}", use_container_width=True):
                        st.session_state[SELECTBOX_KEY] = menu_to_display[label]
                        st.session_state.show_fitur = False
                        st.rerun()

# ==========================================================
# SIDEBAR MENU
# ==========================================================
st.sidebar.markdown(
    '<div class="sidebar-menu-title">🧭 Navigasi Aplikasi</div>',
    unsafe_allow_html=True
)

selected_display = st.sidebar.selectbox(
    "Pilih Menu",
    list(display_to_menu.keys()),
    key=SELECTBOX_KEY
)
menu = display_to_menu[selected_display]

# ==========================================================
# DATASET TRAINING
# ==========================================================
TRAINING_DATASET = "Dataset_Revisi_Indikator_Kecanduan.csv"

if not os.path.exists(TRAINING_DATASET):
    st.error(f"Dataset training '{TRAINING_DATASET}' tidak ditemukan.")
    st.stop()

df = pd.read_csv(TRAINING_DATASET)

# ==========================================================
# FOOTER SIDEBAR — pengisi bagian bawah agar tampilan lebih ramai
# ==========================================================
st.sidebar.markdown(
    "<hr style='border: none; border-top: 1px solid rgba(255,255,255,0.25); margin: 1.5rem 0 1.1rem 0;'>",
    unsafe_allow_html=True
)

st.sidebar.markdown(
    '<div class="sidebar-menu-title" style="font-size:0.92rem;">📊 Ringkasan Dataset</div>',
    unsafe_allow_html=True
)
st.sidebar.markdown(
    f"""
    <div style="display:flex; gap:0.6rem; margin-bottom:1rem;">
        <div style="flex:1; background: rgba(255,255,255,0.14); border: 1.5px solid rgba(255,255,255,0.3);
                    border-radius: 14px; padding: 0.7rem 0.5rem; text-align:center;">
            <div style="font-size:1.25rem; font-weight:800; line-height:1.2;">{df.shape[0]:,}</div>
            <div style="font-size:0.68rem; opacity:0.85; text-transform:uppercase; letter-spacing:0.5px;">Baris Data</div>
        </div>
        <div style="flex:1; background: rgba(255,255,255,0.14); border: 1.5px solid rgba(255,255,255,0.3);
                    border-radius: 14px; padding: 0.7rem 0.5rem; text-align:center;">
            <div style="font-size:1.25rem; font-weight:800; line-height:1.2;">{df.shape[1]}</div>
            <div style="font-size:0.68rem; opacity:0.85; text-transform:uppercase; letter-spacing:0.5px;">Kolom</div>
        </div>
        <div style="flex:1; background: rgba(255,255,255,0.14); border: 1.5px solid rgba(255,255,255,0.3);
                    border-radius: 14px; padding: 0.7rem 0.5rem; text-align:center;">
            <div style="font-size:1.25rem; font-weight:800; line-height:1.2;">7</div>
            <div style="font-size:0.68rem; opacity:0.85; text-transform:uppercase; letter-spacing:0.5px;">Model ML</div>
        </div>
    </div>
    """,
    unsafe_allow_html=True
)

st.sidebar.markdown(
    """
    <div style="background: rgba(255,255,255,0.14); border: 1.5px solid rgba(255,255,255,0.3);
                border-radius: 14px; padding: 0.9rem 1rem; margin-bottom: 1.1rem; font-size: 0.83rem; line-height:1.55;">
        💡 <b>Mau cek dirimu sendiri?</b><br>
        Buka menu <b>🔍 Prediksi Manual</b> untuk mengisi form singkat dan lihat estimasi tingkat kecanduan media sosialmu.
    </div>
    """,
    unsafe_allow_html=True
)

st.sidebar.markdown(
    """
    <div style="text-align:center; padding-top:0.8rem; border-top:1px solid rgba(255,255,255,0.2);
                font-size:0.76rem; opacity:0.85; line-height:1.6;">
        🧠 Dibuat dengan <span style="color:#FF6B81;">❤️</span> memakai Streamlit<br>
        Versi 1.0.0 • © 2026
    </div>
    """,
    unsafe_allow_html=True
)

# ==========================================================
# FUNGSI PREPROCESSING & TRAINING (VERSI BEBAS DATA LEAKAGE)
# ==========================================================
# Prinsip yang dipakai di sini:
# 1. train_test_split dilakukan PALING AWAL, sebelum proses apa pun
#    yang "belajar" dari data (fit LabelEncoder, fit StandardScaler, dll).
# 2. Semua encoder/scaler HANYA di-fit pada data TRAIN.
#    Data TEST hanya ditransformasi (transform), tidak pernah dipakai fit.
# 3. Kategori pada data test yang tidak pernah muncul di data train
#    ditangani dengan fallback (bukan error), sama seperti pada menu
#    prediksi upload CSV.
# Dengan urutan ini, tidak ada informasi dari data test yang "bocor"
# ke proses training, sehingga skor akurasi antar model jadi jujur
# (bisa berbeda-beda sesuai karakteristik masing-masing algoritma).
def prepare_training_data(data):
    """Membersihkan data lalu membagi menjadi train/test SEBELUM encoding & scaling."""
    data = data.copy().drop_duplicates()

    # Hapus kolom yang tidak boleh dipakai sebagai fitur:
    # - Student_ID: identifier, tidak relevan
    # - Addicted_Score, Sleep_Addiction_Indicator, Physical_Activity_Indicator:
    #   kolom ini adalah turunan/indikator dari target itu sendiri, sehingga
    #   kalau ikut dipakai sebagai fitur akan jadi TARGET LEAKAGE (model
    #   "curi start" karena diam-diam sudah tahu jawabannya).
    drop_cols = ["Student_ID", "Addicted_Score", "Sleep_Addiction_Indicator", "Physical_Activity_Indicator"]
    drop_cols = [col for col in drop_cols if col in data.columns]
    data = data.drop(columns=drop_cols)

    # Validasi target
    target = "Addiction_Level"
    if target not in data.columns:
        st.error("Kolom 'Addiction_Level' tidak ditemukan pada dataset training.")
        st.stop()

    X_raw = data.drop(columns=[target])
    y_raw = data[target]

    # ---- 1) SPLIT DULU, SEBELUM ENCODING/SCALING APA PUN ----
    X_train_raw, X_test_raw, y_train_raw, y_test_raw = train_test_split(
        X_raw, y_raw, test_size=0.2, random_state=42, stratify=y_raw
    )

    # ---- 2) Encode target: fit hanya di train, transform di test ----
    target_encoder = LabelEncoder()
    y_train = target_encoder.fit_transform(y_train_raw.astype(str))
    y_test = target_encoder.transform(y_test_raw.astype(str))

    # ---- 3) Encode fitur kategorikal: fit hanya di train ----
    feature_encoders = {}
    categorical_columns = X_train_raw.select_dtypes(include="object").columns

    X_train_enc = X_train_raw.copy()
    X_test_enc = X_test_raw.copy()

    for col in categorical_columns:
        encoder = LabelEncoder()
        X_train_enc[col] = encoder.fit_transform(X_train_raw[col].astype(str))

        # Kategori di data test yang tidak pernah terlihat saat fit (train)
        # difallback ke kelas pertama, bukan menyebabkan error/leakage.
        known_classes = set(encoder.classes_)
        default_class = encoder.classes_[0]
        test_col_str = X_test_raw[col].astype(str).apply(lambda x: x if x in known_classes else default_class)
        X_test_enc[col] = encoder.transform(test_col_str)

        feature_encoders[col] = encoder

    feature_columns = list(X_train_enc.columns)

    # ---- 4) Scaling: fit hanya di train, test hanya di-transform ----
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_enc)
    X_test_scaled = scaler.transform(X_test_enc)

    return {
        "X_train_scaled": X_train_scaled,
        "X_test_scaled": X_test_scaled,
        "y_train": y_train,
        "y_test": y_test,
        "target_encoder": target_encoder,
        "feature_encoders": feature_encoders,
        "scaler": scaler,
        "feature_columns": feature_columns,
        "n_train": X_train_enc.shape[0],
        "n_test": X_test_enc.shape[0],
    }

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
                <p style="font-family:'Inter', sans-serif; font-size:1.02rem; color:#3A4657; font-weight:400;">
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

    st.info("💡 Ingin lihat semua fitur? Klik tombol **✨ Lihat Semua Fitur ↗** di bagian atas halaman — setiap kartu bisa langsung diklik untuk berpindah menu.")

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

    prep = prepare_training_data(df)

    st.success("Preprocessing berhasil dilakukan!")

    st.markdown("""
    Data dibagi menjadi 2 bagian menggunakan *Stratified Train-Test Split* (rasio 80:20) **sebelum** proses
    encoding dan scaling dilakukan. Urutan ini penting: encoder dan scaler hanya "belajar" (fit) dari data
    training, sedangkan data testing hanya ditransformasi memakai aturan yang sudah dipelajari dari data
    training tersebut — sehingga tidak ada informasi dari data testing yang bocor ke proses training.
    * **Data Training (80%)**: Digunakan untuk melatih algoritma *Machine Learning*.
    * **Data Testing (20%)**: Digunakan untuk menguji dan mengevaluasi akurasi model yang telah dilatih.
    """)

    st.write(f"✔ Jumlah Data Training: {prep['n_train']} sampel")
    st.write(f"✔ Jumlah Data Testing: {prep['n_test']} sampel")
    st.write("Target Classes:", list(prep["target_encoder"].classes_))

# --- TRAINING ---
elif menu == "Training":
    st.header("🤖 Training Model")
    
    st.markdown("""
    Halaman ini melakukan pelatihan (*training*) secara bersamaan pada beberapa algoritma *Machine Learning* menggunakan data yang telah di-*preprocessing*. 
    Performa tiap algoritma kemudian dievaluasi menggunakan data testing berdasarkan metrik **Accuracy**, **Precision**, **Recall**, dan **F1 Score**.
    """)

    prep = prepare_training_data(df)
    X_train_scaled = prep["X_train_scaled"]
    X_test_scaled = prep["X_test_scaled"]
    y_train = prep["y_train"]
    y_test = prep["y_test"]
    target_encoder = prep["target_encoder"]
    feature_encoders = prep["feature_encoders"]
    scaler = prep["scaler"]
    feature_columns = prep["feature_columns"]

    st.caption(
        "🔒 Encoder & scaler hanya dilatih (fit) dari data training. Data testing hanya "
        "ditransformasi, sehingga skor di bawah ini adalah performa jujur tiap model — "
        "wajar jika angkanya tidak seragam antar algoritma."
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
