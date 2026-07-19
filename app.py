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

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    ConfusionMatrixDisplay
)

try:
    from xgboost import XGBClassifier
    xgb = True
except:
    xgb = False


# ==========================================================
# KONFIGURASI
# ==========================================================

st.set_page_config(
    page_title="Prediksi Kecanduan Media Sosial",
    page_icon="📱",
    layout="wide"
)


st.title("📱 Prediksi Tingkat Kecanduan Media Sosial")
st.write("---")


# ==========================================================
# MENU
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

    st.error(
        f"Dataset training '{TRAINING_DATASET}' tidak ditemukan."
    )

    st.stop()


df = pd.read_csv(
    TRAINING_DATASET
)


# ==========================================================
# FUNGSI PREPROCESSING
# ==========================================================

def prepare_training_data(data):
    data = data.copy()

    # Hapus duplikat
    data = data.drop_duplicates()

    # Kolom yang tidak digunakan
    drop_cols = [
        "Student_ID",
        "Addicted_Score",
        "Sleep_Addiction_Indicator",
        "Physical_Activity_Indicator"
    ]

    drop_cols = [
        col for col in drop_cols
        if col in data.columns
    ]

    data = data.drop(
        columns=drop_cols
    )

    # Target
    target = "Addiction_Level"

    if target not in data.columns:

        st.error(
            "Kolom 'Addiction_Level' tidak ditemukan pada dataset training."
        )

        st.stop()

    # Pisahkan fitur dan target
    X = data.drop(
        columns=[target]
    )

    y = data[target]

    # Encode target
    target_encoder = LabelEncoder()

    y_encoded = target_encoder.fit_transform(
        y.astype(str)
    )

    # Encoder fitur
    feature_encoders = {}

    categorical_columns = X.select_dtypes(
        include="object"
    ).columns

    for col in categorical_columns:

        encoder = LabelEncoder()

        X[col] = encoder.fit_transform(
            X[col].astype(str)
        )

        feature_encoders[col] = encoder

    # Simpan urutan kolom
    feature_columns = list(
        X.columns
    )

    # Scaling
    scaler = StandardScaler()
    
def train_models_automatically():
    (
        X,
        y,
        target_encoder,
        feature_encoders,
        scaler,
        feature_columns
    ) = prepare_training_data(df)

    X_train, X_test, y_train, y_test = train_test_split(

        X,

        y,

        test_size=0.2,

        random_state=42,

        stratify=y
    )

    # Scaling
    X_train_scaled = scaler.fit_transform(
        X_train
    )

    X_test_scaled = scaler.transform(
        X_test
    )

    models = {

        "Logistic Regression":
            LogisticRegression(
                max_iter=1000
            ),

        "Decision Tree":
            DecisionTreeClassifier(
                random_state=42
            ),

        "Random Forest":
            RandomForestClassifier(
                n_estimators=200,
                random_state=42
            ),

        "KNN":
            KNeighborsClassifier(),

        "Naive Bayes":
            GaussianNB(),

        "SVM":
            SVC(
                probability=True
            )
    }

    if xgb:

        models["XGBoost"] = XGBClassifier(

            eval_metric="mlogloss",

            random_state=42
        )

    trained_models = {}

    for nama, model in models.items():

        model.fit(

            X_train_scaled,

            y_train
        )

        trained_models[nama] = model

    return (

        trained_models,

        scaler,

        target_encoder,

        feature_encoders,

        feature_columns

    )
    data = data.copy()

    # Hapus duplikat
    data = data.drop_duplicates()

    # Kolom yang tidak digunakan
    drop_cols = [
        "Student_ID",
        "Addicted_Score",
        "Sleep_Addiction_Indicator",
        "Physical_Activity_Indicator"
    ]

    drop_cols = [
        col for col in drop_cols
        if col in data.columns
    ]

    data = data.drop(
        columns=drop_cols
    )

    # Target
    target = "Addiction_Level"

    if target not in data.columns:

        st.error(
            "Kolom 'Addiction_Level' tidak ditemukan pada dataset training."
        )

        st.stop()

    # Pisahkan fitur dan target
    X = data.drop(
        columns=[target]
    )

    y = data[target]

    # Encode target
    target_encoder = LabelEncoder()

    y_encoded = target_encoder.fit_transform(
        y.astype(str)
    )

    # Encoder fitur
    feature_encoders = {}

    categorical_columns = X.select_dtypes(
        include="object"
    ).columns

    for col in categorical_columns:

        encoder = LabelEncoder()

        X[col] = encoder.fit_transform(
            X[col].astype(str)
        )

        feature_encoders[col] = encoder

    # Simpan urutan kolom
    feature_columns = list(
        X.columns
    )

    # Scaling
    scaler = StandardScaler()

    X_scaled = scaler.fit_transform(
        X
    )

    return (
        X,
        y_encoded,
        target_encoder,
        feature_encoders,
        scaler,
        feature_columns
    )


# ==========================================================
# HOME
# ==========================================================

if menu == "Home":

    st.header(
        "Machine Learning Project"
    )

    st.image(
        "https://images.unsplash.com/photo-1611162618071-b39a2ec055fb"
    )

    st.write(
        """
        ## Prediksi Tingkat Kecanduan Media Sosial

        Aplikasi ini menggunakan Machine Learning
        untuk memprediksi tingkat kecanduan media sosial.

        ### Model yang Digunakan:

        ✔ Logistic Regression

        ✔ Decision Tree

        ✔ Random Forest

        ✔ KNN

        ✔ Naive Bayes

        ✔ SVM

        ✔ XGBoost

        ### Fitur:

        📊 Analisis Dataset

        📈 EDA

        ⚙️ Preprocessing

        🤖 Training Model

        👤 Prediksi Manual

        📁 Upload Dataset Baru untuk Prediksi

        📊 Perbandingan Aktual vs Prediksi
        """
    )


# ==========================================================
# DATASET
# ==========================================================

elif menu == "Dataset":

    st.header(
        "📊 Dataset Training"
    )

    st.info(
        f"Dataset training yang digunakan: {TRAINING_DATASET}"
    )

    st.dataframe(
        df,
        use_container_width=True
    )

    st.subheader(
        "Ukuran Dataset"
    )

    st.write(
        f"Jumlah Baris: {df.shape[0]}"
    )

    st.write(
        f"Jumlah Kolom: {df.shape[1]}"
    )

    st.subheader(
        "Statistik Deskriptif"
    )

    st.write(
        df.describe(
            include="all"
        )
    )

    st.subheader(
        "Tipe Data"
    )

    st.write(
        df.dtypes
    )


# ==========================================================
# EDA
# ==========================================================

elif menu == "EDA":

    st.header(
        "📈 Exploratory Data Analysis"
    )

    st.subheader(
        "5 Data Pertama"
    )

    st.dataframe(
        df.head()
    )

    st.subheader(
        "Informasi Dataset"
    )

    info = pd.DataFrame({

        "Kolom": df.columns,

        "Tipe Data": df.dtypes.astype(str),

        "Missing Value": df.isnull().sum(),

        "Unique": df.nunique()

    })

    st.dataframe(
        info,
        use_container_width=True
    )

    st.subheader(
        "Distribusi Addiction Level"
    )

    if "Addiction_Level" in df.columns:

        fig, ax = plt.subplots(
            figsize=(6, 4)
        )

        df[
            "Addiction_Level"
        ].value_counts().plot(
            kind="bar",
            ax=ax
        )

        ax.set_xlabel(
            "Tingkat Kecanduan"
        )

        ax.set_ylabel(
            "Jumlah"
        )

        ax.set_title(
            "Distribusi Tingkat Kecanduan"
        )

        st.pyplot(
            fig
        )

    numeric = df.select_dtypes(
        include=np.number
    )

    st.subheader(
        "Distribusi Variabel Numerik"
    )

    for col in numeric.columns:

        fig, ax = plt.subplots(
            figsize=(6, 3)
        )

        ax.hist(
            numeric[col],
            bins=20
        )

        ax.set_title(
            col
        )

        st.pyplot(
            fig
        )

    st.subheader(
        "Correlation Matrix"
    )

    if not numeric.empty:

        corr = numeric.corr()

        fig, ax = plt.subplots(
            figsize=(10, 8)
        )

        im = ax.imshow(
            corr
        )

        ax.set_xticks(
            range(len(corr.columns))
        )

        ax.set_xticklabels(
            corr.columns,
            rotation=90
        )

        ax.set_yticks(
            range(len(corr.columns))
        )

        ax.set_yticklabels(
            corr.columns
        )

        plt.colorbar(
            im
        )

        st.pyplot(
            fig
        )


# ==========================================================
# PREPROCESSING
# ==========================================================

elif menu == "Preprocessing":

    st.header(
        "⚙️ Preprocessing Data"
    )

    st.write(
        "Dataset training yang digunakan:"
    )

    st.write(
        TRAINING_DATASET
    )

    st.write(
        "Ukuran Dataset Awal:"
    )

    st.write(
        df.shape
    )

    (
        X,
        y,
        target_encoder,
        feature_encoders,
        scaler,
        feature_columns
    ) = prepare_training_data(
        df
    )

    X_train, X_test, y_train, y_test = train_test_split(

        X,

        y,

        test_size=0.2,

        random_state=42,

        stratify=y
    )

    st.success(
        "Preprocessing berhasil!"
    )

    st.write(
        f"Jumlah Data Training: {X_train.shape}"
    )

    st.write(
        f"Jumlah Data Testing: {X_test.shape}"
    )

    st.write(
        "Kelas Target:"
    )

    st.write(
        target_encoder.classes_
    )


# ==========================================================
# TRAINING
# ==========================================================

elif menu == "Training":

    st.header(
        "🤖 Training Model"
    )

    (
        X,
        y,
        target_encoder,
        feature_encoders,
        scaler,
        feature_columns
    ) = prepare_training_data(
        df
    )

    X_train, X_test, y_train, y_test = train_test_split(

        X,

        y,

        test_size=0.2,

        random_state=42,

        stratify=y
    )

    # Scaling
    X_train_scaled = scaler.fit_transform(
        X_train
    )

    X_test_scaled = scaler.transform(
        X_test
    )

    models = {

        "Logistic Regression":
            LogisticRegression(
                max_iter=1000
            ),

        "Decision Tree":
            DecisionTreeClassifier(
                random_state=42
            ),

        "Random Forest":
            RandomForestClassifier(
                n_estimators=200,
                random_state=42
            ),

        "KNN":
            KNeighborsClassifier(),

        "Naive Bayes":
            GaussianNB(),

        "SVM":
            SVC(
                probability=True
            )
    }

    if xgb:

        models["XGBoost"] = XGBClassifier(

            eval_metric="mlogloss",

            random_state=42
        )

    hasil = []

    trained_models = {}

    progress = st.progress(
        0
    )

    total = len(
        models
    )

    for i, (

        nama,

        model

    ) in enumerate(

        models.items()

    ):

        model.fit(

            X_train_scaled,

            y_train
        )

        pred = model.predict(

            X_test_scaled
        )

        acc = accuracy_score(

            y_test,

            pred
        )

        pre = precision_score(

            y_test,

            pred,

            average="weighted",

            zero_division=0
        )

        rec = recall_score(

            y_test,

            pred,

            average="weighted",

            zero_division=0
        )

        f1 = f1_score(

            y_test,

            pred,

            average="weighted",

            zero_division=0
        )

        hasil.append(

            [

                nama,

                acc,

                pre,

                rec,

                f1

            ]

        )

        trained_models[nama] = model

        progress.progress(

            (i + 1) / total
        )

    hasil_df = pd.DataFrame(

        hasil,

        columns=[

            "Model",

            "Accuracy",

            "Precision",

            "Recall",

            "F1 Score"

        ]

    )

    hasil_df = hasil_df.sort_values(

        by="Accuracy",

        ascending=False
    )

    st.success(
        "Training selesai!"
    )

    st.dataframe(
        hasil_df,
        use_container_width=True
    )

    best_model_name = hasil_df.iloc[0][
        "Model"
    ]

    st.subheader(
        "🏆 Model Terbaik"
    )

    st.success(
        best_model_name
    )

    # Simpan file model
    joblib.dump(

        trained_models,

        "Semua_Model.pkl"
    )

    joblib.dump(

        scaler,

        "Scaler.pkl"
    )

    joblib.dump(

        target_encoder,

        "Target_Encoder.pkl"
    )

    joblib.dump(

        feature_encoders,

        "Feature_Encoders.pkl"
    )

    joblib.dump(

        feature_columns,

        "Feature_Columns.pkl"
    )

    st.success(
        "Model berhasil disimpan!"
    )

    st.subheader(
        "📊 Perbandingan Accuracy"
    )

    fig, ax = plt.subplots(
        figsize=(10, 5)
    )

    ax.bar(

        hasil_df["Model"],

        hasil_df["Accuracy"]
    )

    ax.set_ylabel(
        "Accuracy"
    )

    ax.set_title(
        "Perbandingan Akurasi Model"
    )

    plt.xticks(
        rotation=20
    )

    st.pyplot(
        fig
    )


# ==========================================================
# PREDIKSI MANUAL
# ==========================================================

elif menu == "Prediksi Manual":

    st.header(
        "🔍 Prediksi Manual"
    )

    required_files = [

        "Semua_Model.pkl",

        "Scaler.pkl",

        "Target_Encoder.pkl",

        "Feature_Encoders.pkl",

        "Feature_Columns.pkl"

    ]

    if not all(

        os.path.exists(
            file
        )

        for file in required_files

    ):

        st.error(
            "Silakan jalankan menu Training terlebih dahulu."
        )

        st.stop()

    models = joblib.load(
        "Semua_Model.pkl"
    )

    scaler = joblib.load(
        "Scaler.pkl"
    )

    target_encoder = joblib.load(
        "Target_Encoder.pkl"
    )

    feature_encoders = joblib.load(
        "Feature_Encoders.pkl"
    )

    feature_columns = joblib.load(
        "Feature_Columns.pkl"
    )

    model_name = st.selectbox(

        "Pilih Model",

        list(
            models.keys()
        )
    )

    model = models[
        model_name
    ]

    st.info(
        "Form prediksi manual dapat digunakan sesuai fitur dataset training."
    )


# ==========================================================
# PREDIKSI DATASET UPLOAD
# ==========================================================

elif menu == "Prediksi Dataset Upload":

    st.header(
        "📁 Upload Dataset Baru untuk Prediksi"
    )

    st.info(
        """
        Upload dataset CSV baru.

        Dataset ini TIDAK digunakan untuk training ulang.

        Dataset hanya akan digunakan untuk
        melakukan prediksi menggunakan model
        yang sudah dilatih.
        """
    )

    uploaded_prediction = st.file_uploader(

        "Upload Dataset Baru",

        type=["csv"],

        key="dataset_prediction"
    )

    if uploaded_prediction is None:

        st.warning(
            "Silakan upload dataset CSV."
        )

        st.stop()

    new_data = pd.read_csv(

        uploaded_prediction
    )

    st.subheader(
        "📄 Dataset yang Di-upload"
    )

    st.dataframe(

        new_data,

        use_container_width=True
    )

    required_files = [

        "Semua_Model.pkl",

        "Scaler.pkl",

        "Target_Encoder.pkl",

        "Feature_Encoders.pkl",

        "Feature_Columns.pkl"

    ]

    if not all(

        os.path.exists(
            file
        )

        for file in required_files

    ):

        st.error(

            "Model belum tersedia. "

            "Silakan jalankan Training terlebih dahulu."
        )

        st.stop()

    models = joblib.load(
        "Semua_Model.pkl"
    )

    scaler = joblib.load(
        "Scaler.pkl"
    )

    target_encoder = joblib.load(
        "Target_Encoder.pkl"
    )

    feature_encoders = joblib.load(
        "Feature_Encoders.pkl"
    )

    feature_columns = joblib.load(
        "Feature_Columns.pkl"
    )

    model_name = st.selectbox(

        "Pilih Model untuk Prediksi",

        list(
            models.keys()
        )
    )

    model = models[
        model_name
    ]

    prediction_data = new_data.copy()

    # Simpan label asli jika tersedia
    actual_labels = None

    if "Addiction_Level" in prediction_data.columns:

        actual_labels = prediction_data[
            "Addiction_Level"
        ].copy()

        prediction_data = prediction_data.drop(

            columns=["Addiction_Level"]
        )

    # Hapus kolom yang tidak digunakan
    drop_cols = [

        "Student_ID",

        "Addicted_Score",

        "Sleep_Addiction_Indicator",

        "Physical_Activity_Indicator"

    ]

    drop_cols = [

        col

        for col in drop_cols

        if col in prediction_data.columns

    ]

    prediction_data = prediction_data.drop(

        columns=drop_cols
    )

    # Cek kolom yang hilang
    missing_columns = [

        col

        for col in feature_columns

        if col not in prediction_data.columns

    ]

    if missing_columns:

        st.error(
            "Kolom dataset upload tidak lengkap."
        )

        st.write(
            "Kolom yang hilang:"
        )

        st.write(
            missing_columns
        )

        st.stop()

    # Ambil fitur sesuai training
    prediction_data = prediction_data[
        feature_columns
    ]

    # Encode kategori
    for col, encoder in feature_encoders.items():

        try:

            prediction_data[col] = encoder.transform(

                prediction_data[col].astype(str)
            )

        except:

            st.error(

                f"Nilai pada kolom '{col}' "
                "tidak sesuai dengan dataset training."
            )

            st.stop()

    # Scaling
    prediction_scaled = scaler.transform(

        prediction_data
    )

    # Prediksi
    predictions = model.predict(

        prediction_scaled
    )

    predicted_labels = target_encoder.inverse_transform(

        predictions
    )

    # Hasil
    result_data = new_data.copy()

    result_data[
        "Prediction"
    ] = predicted_labels

    st.success(
        "Prediksi berhasil!"
    )

    st.subheader(
        "📊 Hasil Prediksi"
    )

    st.dataframe(

        result_data,

        use_container_width=True
    )

    # ======================================================
    # PERBANDINGAN
    # ======================================================

    if actual_labels is not None:

        st.header(
            "📈 Perbandingan Aktual vs Prediksi"
        )

        comparison = pd.DataFrame({

            "Aktual":
                actual_labels.values,

            "Prediksi":
                predicted_labels

        })

        st.dataframe(

            comparison,

            use_container_width=True
        )

        accuracy = accuracy_score(

            actual_labels,

            predicted_labels
        )

        precision = precision_score(

            actual_labels,

            predicted_labels,

            average="weighted",

            zero_division=0
        )

        recall = recall_score(

            actual_labels,

            predicted_labels,

            average="weighted",

            zero_division=0
        )

        f1 = f1_score(

            actual_labels,

            predicted_labels,

            average="weighted",

            zero_division=0
        )

        col1, col2, col3, col4 = st.columns(4)

        col1.metric(
            "Accuracy",
            f"{accuracy * 100:.2f}%"
        )

        col2.metric(
            "Precision",
            f"{precision * 100:.2f}%"
        )

        col3.metric(
            "Recall",
            f"{recall * 100:.2f}%"
        )

        col4.metric(
            "F1 Score",
            f"{f1 * 100:.2f}%"
        )

        st.subheader(
            "📊 Confusion Matrix"
        )

        labels = target_encoder.classes_

        # Ubah label aktual dan prediksi menjadi angka
        actual_encoded = target_encoder.transform(
            actual_labels.astype(str)
        )

        predicted_encoded = target_encoder.transform(
            predicted_labels.astype(str)
        )

        cm = confusion_matrix(

            actual_encoded,

            predicted_encoded,

            labels=range(
                len(labels)
            )
        )

        fig, ax = plt.subplots(

            figsize=(7, 5)
        )

        disp = ConfusionMatrixDisplay(

            confusion_matrix=cm,

            display_labels=labels
        )

        disp.plot(

            ax=ax
        )

        st.pyplot(
            fig
        )

    else:

        st.info(

            """
            Dataset tidak memiliki kolom
            Addiction_Level.

            Sistem hanya melakukan prediksi.

            Accuracy, Precision, Recall, dan F1 Score
            tidak dapat dihitung karena tidak ada
            label aktual.
            """
        )

    # Download
    csv = result_data.to_csv(

        index=False
    ).encode(
        "utf-8"
    )

    st.download_button(

        "⬇️ Download Hasil Prediksi CSV",

        data=csv,

        file_name="hasil_prediksi.csv",

        mime="text/csv"
    )
