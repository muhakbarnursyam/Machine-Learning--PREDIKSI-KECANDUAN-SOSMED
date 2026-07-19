elif menu == "Prediksi Dataset Upload":
    st.header("📁 Upload Dataset Baru & Analisis Model")
    st.info(
        """
        Silakan upload dataset CSV baru. 
        - Jika dataset memiliki kolom **'Addiction_Level'**, sistem akan menampilkan **Fitur Perbandingan Performa** antar model.
        - Jika tidak ada, sistem akan langsung memberikan hasil prediksi untuk semua model.
        """
    )

    uploaded_prediction = st.file_uploader("Upload Dataset Baru", type=["csv"], key="dataset_prediction")

    if uploaded_prediction is None:
        st.warning("Silakan upload dataset CSV.")
        st.stop()

    new_data = pd.read_csv(uploaded_prediction)

    st.subheader("📄 Dataset yang Di-upload")
    st.dataframe(new_data, use_container_width=True)

    required_files = ["Semua_Model.pkl", "Scaler.pkl", "Target_Encoder.pkl", "Feature_Encoders.pkl", "Feature_Columns.pkl"]
    if not all(os.path.exists(file) for file in required_files):
        st.error("Model belum tersedia. Silakan jalankan menu Training terlebih dahulu.")
        st.stop()

    models = joblib.load("Semua_Model.pkl")
    scaler = joblib.load("Scaler.pkl")
    target_encoder = joblib.load("Target_Encoder.pkl")
    feature_encoders = joblib.load("Feature_Encoders.pkl")
    feature_columns = joblib.load("Feature_Columns.pkl")

    # Cek apakah ada kolom target aktual di dataset baru
    has_target = "Addiction_Level" in new_data.columns

    if has_target:
        actual_labels = new_data["Addiction_Level"].copy()
        prediction_data = new_data.copy().drop(columns=["Addiction_Level"])
    else:
        actual_labels = None
        prediction_data = new_data.copy()

    # Hapus kolom yang tidak digunakan sewaktu training awal
    drop_cols = ["Student_ID", "Addicted_Score", "Sleep_Addiction_Indicator", "Physical_Activity_Indicator"]
    drop_cols = [col for col in drop_cols if col in prediction_data.columns]
    prediction_data = prediction_data.drop(columns=drop_cols)

    # Cek kecocokan kolom fitur
    missing_columns = [col for col in feature_columns if col not in prediction_data.columns]
    if missing_columns:
        st.error("Kolom dataset upload tidak lengkap sesuai format training.")
        st.write("Kolom yang hilang:", missing_columns)
        st.stop()

    # Sesuaikan urutan kolom
    prediction_data = prediction_data[feature_columns]

    # Encode kategori dengan encoder lama
    for col, encoder in feature_encoders.items():
        try:
            prediction_data[col] = encoder.transform(prediction_data[col].astype(str))
        except:
            st.error(f"Nilai kategori pada kolom '{col}' tidak sesuai dengan data training.")
            st.stop()

    # Scaling
    prediction_scaled = scaler.transform(prediction_data)

    # ======================================================
    # PROSES PREDIKSI MULTI-MODEL
    # ======================================================
    st.write("---")
    
    perbandingan_list = []
    result_data_all = new_data.copy()

    # Looping untuk memprediksi data lewat seluruh model yang ada
    for name, model in models.items():
        preds = model.predict(prediction_scaled)
        pred_labels = target_encoder.inverse_transform(preds)
        
        # Tambahkan kolom prediksi model ke data akhir
        result_data_all[f"Prediksi ({name})"] = pred_labels
        
        # Hitung metrik evaluasi HANYA JIKA kolom aktualnya ada
        if has_target:
            acc = accuracy_score(actual_labels, pred_labels)
            pre = precision_score(actual_labels, pred_labels, average="weighted", zero_division=0)
            rec = recall_score(actual_labels, pred_labels, average="weighted", zero_division=0)
            f1 = f1_score(actual_labels, pred_labels, average="weighted", zero_division=0)
            perbandingan_list.append([name, acc, pre, rec, f1])

    # ======================================================
    # TAMPILKAN FITUR PERBANDINGAN (JIKA ADA LABELS AKTUAL)
    # ======================================================
    if has_target:
        st.subheader("📈 Perbandingan Performa Semua Model (Berdasarkan Nilai Aktual)")
        
        # Buat Dataframe hasil komparasi
        df_compare = pd.DataFrame(
            perbandingan_list, 
            columns=["Model", "Accuracy", "Precision", "Recall", "F1 Score"]
        ).sort_values(by="Accuracy", ascending=False)

        # Tampilkan tabel perbandingan dengan format persentase
        st.dataframe(df_compare.style.format({
            "Accuracy": "{:.2%}",
            "Precision": "{:.2%}",
            "Recall": "{:.2%}",
            "F1 Score": "{:.2%}"
        }), use_container_width=True)

        # Visualisasi Grafik Batang Perbandingan Akurasi
        fig, ax = plt.subplots(figsize=(10, 4))
        bars = ax.bar(df_compare["Model"], df_compare["Accuracy"], color='#1E88E5')
        ax.set_ylabel("Accuracy Score")
        ax.set_title("Visualisasi Akurasi Model pada Dataset Baru")
        plt.xticks(rotation=15)
        
        # Berikan label persentase di ujung grafik batang
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.01, f"{height*100:.1f}%", ha='center', va='bottom', fontsize=9)
            
        st.pyplot(fig)
    else:
        st.subheader("📊 Hasil Prediksi Massal Model")
        st.info("Dataset tidak memiliki kolom 'Addiction_Level'. Nilai metrik evaluasi (Akurasi/Grafik) tidak dimunculkan karena tidak ada data aktual pembanding.")

    # Tampilkan Detail Data Hasil Prediksi
    st.subheader("📋 Data Lengkap Hasil Prediksi Seluruh Model")
    st.dataframe(result_data_all, use_container_width=True)

    # Fitur Download Hasil
    csv = result_data_all.to_csv(index=False).encode("utf-8")
    st.download_button(
        "⬇️ Download Hasil Prediksi Lengkap (CSV)",
        data=csv,
        file_name="hasil_prediksi_massal.csv",
        mime="text/csv"
    )
