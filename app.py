import io
import os
import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image
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
# KONFIGURASI HALAMAN & TEMA RETRO PIXEL PIALA DUNIA CHIBI
# ==========================================================
st.set_page_config(
    page_title="Prediksi Kecanduan Media Sosial - World Cup Chibi Pixel Edition",
    page_icon="⚽",
    layout="wide"
)

# Custom CSS Inject untuk Tema Pixel World Cup Chibi
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Press+Start+2P&family=VT323&display=swap');

    /* Global Theme - Retro Stadium Turf Background */
    .stApp {
        background-color: #0d2818;
        background-image: 
            radial-gradient(#16db65 1px, transparent 1px),
            radial-gradient(#058c42 1px, #0d2818 1px);
        background-size: 40px 40px;
        background-position: 0 0, 20px 20px;
        color: #e0f2fe;
        font-family: 'VT323', monospace;
        font-size: 20px;
    }

    /* Headings Style - Gold & Neon Green Trophy Look */
    h1, h2, h3, h4 {
        font-family: 'Press Start 2P', cursive !important;
        color: #ffb703 !important;
        text-shadow: 3px 3px 0px #d90429, 6px 6px 0px #000000;
        letter-spacing: 1px;
    }

    /* Sidebar Pixel Styling - Dugout Theme */
    [data-testid="stSidebar"] {
        background-color: #051923 !important;
        border-right: 4px solid #ffb703;
    }
    
    [data-testid="stSidebar"] * {
        font-family: 'VT323', monospace !important;
        font-size: 22px !important;
        color: #00f5d4 !important;
    }

    /* Pixel Art Container Cards */
    div.stDataFrame, div.stForm, .stAlert {
        background: #003566 !important;
        border: 4px solid #ffb703 !important;
        box-shadow: 6px 6px 0px #d90429;
        border-radius: 0px !important;
    }

    /* Retro Pixel Buttons */
    .stButton>button, div[data-testid="stFormSubmitButton"]>button, .stDownloadButton>button {
        font-family: 'Press Start 2P', cursive !important;
        font-size: 12px !important;
        color: #ffffff !important;
        background-color: #d90429 !important;
        border: 3px solid #ffffff !important;
        box-shadow: 4px 4px 0px #000000;
        border-radius: 0px !important;
        padding: 10px 20px !important;
        transition: all 0.1s ease-in-out;
    }

    .stButton>button:hover, div[data-testid="stFormSubmitButton"]>button:hover, .stDownloadButton>button:hover {
        background-color: #00f5d4 !important;
        color: #000000 !important;
        transform: translate(-2px, -2px);
        box-shadow: 6px 6px 0px #000000;
    }

    /* Selectbox & Input Customization */
    div[data-baseweb="select"] > div, input {
        background-color: #051923 !important;
        color: #00f5d4 !important;
        border: 2px solid #00f5d4 !important;
        border-radius: 0px !important;
        font-family: 'VT323', monospace !important;
        font-size: 20px !important;
    }

    /* Metric Boxes Pixel Style */
    [data-testid="stMetricValue"] {
        font-family: 'Press Start 2P', cursive !important;
        color: #00f5d4 !important;
    }
    
    [data-testid="stMetricLabel"] {
        font-family: 'VT323', monospace !important;
        font-size: 22px !important;
        color: #ffb703 !important;
    }

    /* Tabs Retro Styling */
    button[data-baseweb="tab"] {
        font-family: 'Press Start 2P', cursive !important;
        font-size: 10px !important;
        color: #ffffff !important;
        background-color: #003566 !important;
        border: 2px solid #00f5d4 !important;
        border-radius: 0px !important;
    }

    button[aria-selected="true"] {
        background-color: #d90429 !important;
        color: #ffffff !important;
    }

    /* Line separator */
    hr {
        border-top: 4px dashed #00f5d4 !important;
    }
    </style>
""", unsafe_allow_html=True)

# Header Utama
st.title("⚽ PREDIKSI KECANDUAN MEDIA SOSIAL")
st.caption("🏆 STADION PIXEL WORLD CUP CHIBI - DATA CENTER")
st.write("---")

# ==========================================================
# CONSTANTS & ASSETS PIXEL CHIBI WORLD CUP
# ==========================================================
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

def draw_pixel_chibi(skin_tone, hair_style, hair_color, team_name):
    """Membuat avatar pixel chibi berpiksel 16x16 scaled up ke 320x320."""
    grid = np.zeros((16, 16, 4), dtype=np.uint8)

    skin_rgb = SKIN_COLORS[skin_tone]
    hair_rgb = HAIR_COLORS[hair_color]
    jersey_rgb = JERSEY_TEAMS[team_name]["base"]
    shorts_rgb = JERSEY_TEAMS[team_name]["shorts"]

    def set_p(x, y, color):
        if 0 <= x < 16 and 0 <= y < 16:
            grid[y, x] = color

    # 1. KEPALA & WAJAH
    for x in range(4, 12):
        for y in range(2, 9):
            set_p(x, y, skin_rgb)

    # Mata Chibi
    set_p(5, 5, (0, 0, 0, 255))
    set_p(5, 6, (0, 0, 0, 255))
    set_p(10, 5, (0, 0, 0, 255))
    set_p(10, 6, (0, 0, 0, 255))

    # Blush
    set_p(4, 7, (255, 150, 150, 255))
    set_p(11, 7, (255, 150, 150, 255))

    # 2. RAMBUT
    if hair_style == "Pendek Standard":
        for x in range(4, 12):
            set_p(x, 1, hair_rgb)
            set_p(x, 2, hair_rgb)
        set_p(3, 2, hair_rgb)
        set_p(3, 3, hair_rgb)
        set_p(12, 2, hair_rgb)
        set_p(12, 3, hair_rgb)
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

    # 3. BADAN & JERSEY
    for x in range(5, 11):
        for y in range(9, 12):
            set_p(x, y, jersey_rgb)

    # Lengan
    set_p(4, 9, jersey_rgb)
    set_p(4, 10, skin_rgb)
    set_p(11, 9, jersey_rgb)
    set_p(11, 10, skin_rgb)

    # Celana
    for x in range(5, 11):
        set_p(x, 12, shorts_rgb)

    # Sepatu
    set_p(6, 13, skin_rgb)
    set_p(9, 13, skin_rgb)
    set_p(6, 14, (30, 30, 30, 255))
    set_p(5, 14, (30, 30, 30, 255))
    set_p(9, 14, (30, 30, 30, 255))
    set_p(10, 14, (30, 30, 30, 255))

    # 4. BOLA PIXEL PIALA DUNIA
    ball_white = (240, 240, 240, 255)
    ball_black = (20, 20, 20, 255)
    for bx in range(12, 15):
        for by in range(13, 16):
            set_p(bx, by, ball_white)
    set_p(13, 14, ball_black)

    img = Image.fromarray(grid, mode="RGBA")
    img = img.resize((320, 320), resample=Image.NEAREST)
    return img

# ==========================================================
# SIDEBAR MENU
# ==========================================================
menu = st.sidebar.selectbox(
    "MENU NAVIGASI UTAMA",
    [
        "Home",
        "Chibi Generator",
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
    st.header("🏆 MODUL ML PIALA DUNIA PIXEL CHIBI")
    
    col_img, col_txt = st.columns([1, 2])
    with col_img:
        sample_chibi = draw_pixel_chibi("Cerah", "Spiky / Jabrik", "Hitam", "Indonesia 🇮🇩")
        st.image(sample_chibi, caption="Maskot Chibi Pixel", use_container_width=True)
    
    with col_txt:
        st.write(
            """
            ### PREDIKSI TINGKAT KECANDUAN MEDIA SOSIAL
            Sistem cerdas bertema Piala Dunia Berpiksel yang menganalisis tingkat kecanduan media sosial dengan algoritma Machine Learning unggulan.

            ### ALGORITMA DUKUNGAN SISTEM:
            - Logistic Regression & Decision Tree
            - Random Forest & KNN
            - Naive Bayes, SVM & XGBoost

            ### FITUR EKSKLUSIF:
            - **Chibi Generator**: Buat avatar pemain sepak bola piksel kamu sendiri!
            - **Analisis & Prediksi ML**: Prediksi individu & file CSV secara akurat.
            """
        )

# --- CHIBI GENERATOR ---
elif menu == "Chibi Generator":
    st.header("⚽ PIXEL CHIBI WORLD CUP GENERATOR")
    st.write("Kustomisasi dan unduh karakter chibi berpiksel tim favoritmu!")

    col_ctrl, col_prev = st.columns([1, 1])

    with col_ctrl:
        selected_team = st.selectbox("Tim / Negara Jersey", list(JERSEY_TEAMS.keys()))
        selected_skin = st.selectbox("Warna Kulit", list(SKIN_COLORS.keys()))
        selected_hair_style = st.selectbox("Gaya Rambut", ["Pendek Standard", "Spiky / Jabrik", "Gondrong"])
        selected_hair_color = st.selectbox("Warna Rambut", list(HAIR_COLORS.keys()))

    chibi_img = draw_pixel_chibi(selected_skin, selected_hair_style, selected_hair_color, selected_team)

    with col_prev:
        st.image(chibi_img, caption=f"Pemain Chibi Pixel - {selected_team}", use_container_width=False, width=280)

        buf = io.BytesIO()
        chibi_img.save(buf, format="PNG")
        byte_im = buf.getvalue()

        st.download_button(
            label="📥 Unduh Avatar Chibi (PNG)",
            data=byte_im,
            file_name=f"chibi_worldcup_{selected_team.split()[0].lower()}.png",
            mime="image/png",
        )

# --- DATASET ---
elif menu == "Dataset":
    st.header("📊 DATASET PIPELINE")
    
    if not os.path.exists(TRAINING_DATASET):
        st.error(f"Berkas dataset training '{TRAINING_DATASET}' tidak ditemukan.")
        st.stop()

    df = pd.read_csv(TRAINING_DATASET)
    st.info(f"Sumber Berkas Data: {TRAINING_DATASET}")
    st.dataframe(df, use_container_width=True)

    col1, col2 = st.columns(2)
    col1.metric("JUMLAH SAMPEL", df.shape[0])
    col2.metric("JUMLAH ATRIBUT", df.shape[1])
    
    st.subheader("STATISTIK DESKRIPTIF")
    st.write(df.describe(include="all"))

# --- EDA ---
elif menu == "EDA":
    st.header("🔍 EXPLORATORY DATA ANALYSIS")
    
    if not os.path.exists(TRAINING_DATASET):
        st.error(f"Berkas dataset training '{TRAINING_DATASET}' tidak ditemukan.")
        st.stop()

    df = pd.read_csv(TRAINING_DATASET)
    st.subheader("5 SAMPEL PERTAMA")
    st.dataframe(df.head())

    plt.style.use('dark_background')

    if "Addiction_Level" in df.columns:
        st.subheader("DISTRIBUSI ADDICTION LEVEL")
        fig, ax = plt.subplots(figsize=(7, 3.5))
        fig.patch.set_facecolor('#003566')
        ax.set_facecolor('#051923')
        
        df["Addiction_Level"].value_counts().plot(kind="bar", ax=ax, color='#00f5d4', edgecolor='#ffb703', linewidth=2)
        ax.set_xlabel("Tingkat Kecanduan", color='#ffb703')
        ax.set_ylabel("Jumlah", color='#ffb703')
        st.pyplot(fig)

    numeric = df.select_dtypes(include=np.number)
    if not numeric.empty:
        st.subheader("MATRIKS KORELASI PARAMETER")
        corr = numeric.corr()
        fig, ax = plt.subplots(figsize=(8, 5))
        fig.patch.set_facecolor('#003566')
        ax.set_facecolor('#051923')
        
        im = ax.imshow(corr, cmap='viridis')
        ax.set_xticks(range(len(corr.columns)))
        ax.set_xticklabels(corr.columns, rotation=90, color='#00f5d4')
        ax.set_yticks(range(len(corr.columns)))
        ax.set_yticklabels(corr.columns, color='#00f5d4')
        plt.colorbar(im)
        st.pyplot(fig)

# --- PREPROCESSING ---
elif menu == "Preprocessing":
    st.header("⚙️ PREPROCESSING DATASET")
    if not os.path.exists(TRAINING_DATASET):
        st.error(f"Berkas dataset training '{TRAINING_DATASET}' tidak ditemukan.")
        st.stop()

    df = pd.read_csv(TRAINING_DATASET)
    X, y, target_encoder, feature_encoders, scaler, feature_columns = prepare_training_data(df)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    st.success("Proses Preprocessing Berhasil!")
    st.write(f"- Data Training: **{X_train.shape[0]}** sampel")
    st.write(f"- Data Testing: **{X_test.shape[0]}** sampel")
    st.write("Kelas Target terdeteksi:", list(target_encoder.classes_))

# --- TRAINING ---
elif menu == "Training":
    st.header("🏋️ TRAINING MODEL KOMPUTASI")
    if not os.path.exists(TRAINING_DATASET):
        st.error(f"Berkas dataset training '{TRAINING_DATASET}' tidak ditemukan.")
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

    st.success("Training Selesai!")
    st.dataframe(hasil_df.style.format({
        "Accuracy": "{:.2%}", "Precision": "{:.2%}", "Recall": "{:.2%}", "F1 Score": "{:.2%}"
    }), use_container_width=True)

    joblib.dump(trained_models, "Semua_Model.pkl")
    joblib.dump(scaler, "Scaler.pkl")
    joblib.dump(target_encoder, "Target_Encoder.pkl")
    joblib.dump(feature_encoders, "Feature_Encoders.pkl")
    joblib.dump(feature_columns, "Feature_Columns.pkl")
    st.success("Model dan scaler berhasil disimpan!")

# --- PREDIKSI MANUAL ---
elif menu == "Prediksi Manual":
    st.header("🎯 SIMULASI PREDIKSI INDIVIDU")
    
    required_files = ["Semua_Model.pkl", "Scaler.pkl", "Target_Encoder.pkl", "Feature_Encoders.pkl", "Feature_Columns.pkl"]
    if not all(os.path.exists(f) for f in required_files):
        st.error("Silakan jalankan menu Training terlebih dahulu.")
        st.stop()

    models = joblib.load("Semua_Model.pkl")
    scaler = joblib.load("Scaler.pkl")
    target_encoder = joblib.load("Target_Encoder.pkl")
    feature_encoders = joblib.load("Feature_Encoders.pkl")
    feature_columns = joblib.load("Feature_Columns.pkl")

    model_name = st.selectbox("PILIH ALGORITMA PREDIKSI", list(models.keys()))
    model = models[model_name]

    st.subheader("INPUT DATA PARAMETER:")
    input_data = {}
    with st.form("form_prediksi"):
        col1, col2 = st.columns(2)
        ui_inputs = {}
        
        for idx, col_name in enumerate(feature_columns):
            form_col = col1 if idx % 2 == 0 else col2
            if col_name in feature_encoders:
                labels_kategori = list(feature_encoders[col_name].classes_)
                ui_inputs[col_name] = form_col.selectbox(col_name, options=labels_kategori, key=f"ui_{col_name}")
            else:
                ui_inputs[col_name] = form_col.number_input(col_name, value=0, step=1, key=f"ui_{col_name}")
        
        submitted = st.form_submit_button("JALANKAN KALKULASI PREDIKSI")

    if submitted:
        input_df = pd.DataFrame([ui_inputs])
        for col, encoder in feature_encoders.items():
            input_df[col] = encoder.transform([input_df.at[0, col]])
        
        input_scaled = scaler.transform(input_df[feature_columns])
        prediksi_angka = model.predict(input_scaled)
        hasil_label = target_encoder.inverse_transform(prediksi_angka)[0]
        
        st.success(f"Berdasarkan Algoritma **{model_name}**, Tingkat Kecanduan: **{hasil_label}**")

# --- PREDIKSI DATASET UPLOAD ---
elif menu == "Prediksi Dataset Upload":
    st.header("📁 ANALISIS MASSAL FILE CSV")
    uploaded_file = st.file_uploader("Unggah File CSV", type=["csv"])
    
    if uploaded_file is not None:
        user_data = pd.read_csv(uploaded_file)
        st.dataframe(user_data.head(), use_container_width=True)
