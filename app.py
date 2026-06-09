import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    accuracy_score
)
import plotly.express as px
from wordcloud import WordCloud

st.set_page_config(
    page_title="Analisis Sentimen M-Pajak",
    page_icon="📊",
    layout="wide"
)

# =====================================
# CUSTOM CSS
# =====================================
st.markdown("""
<style>
section[data-testid="stSidebar"] {
    background:linear-gradient(
        180deg,
        #10b981,
        #059669
    );
}
#MainMenu {visibility:hidden;}
footer {visibility:hidden;}
header {visibility:hidden;}

.stApp{
    background: linear-gradient(
        135deg,
        #ecfdf5,
        #d1fae5,
        #a7f3d0
    );

.sidebar-title {
    text-align: center;
    font-size: 22px;
    font-weight: 700;
    color: black;
    padding-bottom: 10px;
}
    .card{
    background:white;
    border-radius:20px;
    padding:25px;
    box-shadow:0 8px 20px rgba(0,0,0,0.08);
}

.menu-group {
    font-size: 12px;
    font-weight: 700;
    color: #64748b;
    margin-top: 15px;
    margin-bottom: 5px;
    letter-spacing: 1px;
}

.submenu {
    margin-left: 15px;
}
</style>
""", unsafe_allow_html=True)


# =====================================
# SESSION
# =====================================
if "page" not in st.session_state:
    st.session_state.page = "Beranda"

    # =====================================
# LOAD MODEL
# =====================================
svm_model = joblib.load("svm_model.pkl")
kmeans_model = joblib.load("kmeans_model.pkl")
tfidf = joblib.load("tfidf.pkl")
test_data = pd.read_csv("test_data.csv")

# =====================================
# SIDEBAR
# =====================================
with st.sidebar:

    st.markdown(
        '<div class="sidebar-title">HALAMAN MENU</div>',
        unsafe_allow_html=True
    )

    if st.button("Beranda", use_container_width=True):
        st.session_state.page = "Beranda"

    if st.button("Prediksi", use_container_width=True):
        st.session_state.page = "Prediksi"

    if st.button("Riwayat", use_container_width=True):
        st.session_state.page = "Riwayat"

    st.divider()

    if st.button("Preprocessing", use_container_width=True):
        st.session_state.page = "Preprocessing"

    st.divider()

    if st.button("Pemodelan Supervised", use_container_width=True):
        st.session_state.page = "Pemodelan Supervised"

    if st.button("Pemodelan Unsupervised", use_container_width=True):
        st.session_state.page = "Pemodelan Unsupervised"


    st.divider()

    if st.button("Tentang", use_container_width=True):
        st.session_state.page = "Tentang"


# =====================================
# HALAMAN
# =====================================
page = st.session_state.page


if page == "Beranda":
    st.markdown("""
    <div style="
        background: linear-gradient(90deg,#11998e,#38ef7d);
        padding:30px;
        border-radius:20px;
        text-align:center;
    ">
        <h1 style="color:white;">
        ANALISIS SENTIMEN ULASAN PENGGUNA APLIKASI M-PAJAK
    </h1>

    <p style="color:white;font-size:20px;">
        Sistem Analisis Sentimen Menggunakan
        Support Vector Machine (SVM)
        dan K-Means Clustering
    </p>
    </div>
    """, unsafe_allow_html=True)

    st.write("")

    # =====================================
    # CARD FITUR UTAMA
    # =====================================
    col1, col2, col3 = st.columns(3)

    col1.markdown("""
    <div style="
        background: linear-gradient(135deg,#00c853,#64dd17);
        padding:25px;
        border-radius:15px;
        text-align:center;
        color:white;
        box-shadow:0px 4px 10px rgba(0,0,0,0.2);
    ">
        <h2>SVM</h2>
        <p>Analisis Sentimen<br>Positif • Negatif • Netral</p>
    </div>
    """, unsafe_allow_html=True)

    col2.markdown("""
    <div style="
        background: linear-gradient(135deg,#2962ff,#448aff);
        padding:25px;
        border-radius:15px;
        text-align:center;
        color:white;
        box-shadow:0px 4px 10px rgba(0,0,0,0.2);
    ">
        <h2>Dashboard</h2>
        <p>Accuracy • Confusion Matrix • WordCloud</p><br>
    </div>
    """, unsafe_allow_html=True)

    col3.markdown("""
    <div style="
        background: linear-gradient(135deg,#ff6d00,#ff9100);
        padding:25px;
        border-radius:15px;
        text-align:center;
        color:white;
        box-shadow:0px 4px 10px rgba(0,0,0,0.2);
    ">
        <h2>K-Means</h2>
        <p>Clustering<br>Persona Pengguna</p>
    </div>
    """, unsafe_allow_html=True)

    st.write("")

    # =====================================
    # TENTANG SISTEM (CARD)
    # =====================================
    st.markdown("""
    <div style="
        background:white;
        padding:20px;
        border-radius:15px;
        box-shadow:0px 4px 10px rgba(0,0,0,0.1);
    ">
        <h3>TENTANG SISTEM</h3>
        <p>
            Sistem ini digunakan untuk menganalisis sentimen ulasan pengguna aplikasi M-Pajak
            menggunakan algoritma Machine Learning yaitu SVM dan K-Means Clustering.
        </p>
    </div>
    """, unsafe_allow_html=True)

    st.write("")

    st.markdown("""
        <div style="
            background:white;
            padding:20px;
            border-radius:15px;
            box-shadow:0 4px 10px rgba(0,0,0,0.1);
        ">
        """, unsafe_allow_html=True)

    # =====================================
    # FITUR SISTEM
    # =====================================
    st.subheader("FITUR SISTEM")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("""
        <div style="
            background:#e8f5e9;
            padding:20px;
            border-radius:15px;
        ">
            <b>🚀 Fitur Utama</b><br><br>
            ✔ Prediksi Sentimen Realtime<br>
            ✔ Upload Dataset CSV<br>
            ✔ Dashboard Akurasi<br>
            ✔ Confusion Matrix<br>
            ✔ Classification Report
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown("""
        <div style="
            background:#e3f2fd;
            padding:20px;
            border-radius:15px;
        ">
            <b>📊 Analisis & Visualisasi</b><br><br>
            ✔ Pie Chart Sentimen<br>
            ✔ WordCloud<br>
            ✔ Cluster Persona<br>
            ✔ Export Excel<br>
            ✔ Riwayat Prediksi
        </div>
        """, unsafe_allow_html=True)

elif page == "Prediksi":

    import numpy as np
    import streamlit as st

    st.markdown("""
    <div style="
        background: linear-gradient(90deg,#11998e,#38ef7d);
        padding:30px;
        border-radius:20px;
        text-align:center;
    ">
        <h1 style="color:white;">
            PREDIKSI SENTIMEN REALTIME
        </h1>
    </div>
    """, unsafe_allow_html=True)

    st.write("")

    teks = st.text_area(
        "Masukkan Ulasan",
        height=150,
        placeholder="Contoh: aplikasi sangat membantu dan mudah digunakan"
    )

    # =====================================
    # BUTTON
    # =====================================

    tombol1, tombol2 = st.columns(2)

    with tombol1:
        prediksi_btn = st.button(
            "🔍 Prediksi Sekarang",
            use_container_width=True
        )

    with tombol2:
        upload_btn = st.button(
            "📂 Upload Dataset",
            use_container_width=True
        )

        if upload_btn:
            st.session_state.page = "Upload CSV"
            st.rerun()

    # =====================================
    # PREDIKSI
    # =====================================

    if prediksi_btn:

        if teks.strip() == "":
            st.warning("Masukkan ulasan terlebih dahulu")

        else:

            vector = tfidf.transform([teks])

            # ===============================
            # SENTIMEN
            # ===============================

            sentimen = svm_model.predict(
                vector
            )[0]

            # ===============================
            # CLUSTER
            # ===============================

            cluster = kmeans_model.predict(
                vector
            )[0]

            # ===============================
            # CONFIDENCE SENTIMEN
            # ===============================

            if hasattr(
                svm_model,
                "predict_proba"
            ):

                probability = (
                    svm_model.predict_proba(
                        vector
                    )
                )

                confidence = (
                    np.max(probability)
                    * 100
                )

            else:

                decision = (
                    svm_model.decision_function(
                        vector
                    )
                )

                confidence = (
                    1 /
                    (
                        1 +
                        np.exp(
                            -np.max(
                                np.abs(decision)
                            )
                        )
                    )
                ) * 100

            confidence = float(
                np.clip(
                    confidence,
                    40,
                    99
                )
            )

            # ===============================
            # CONFIDENCE CLUSTER
            # ===============================

            distances = (
                kmeans_model.transform(
                    vector
                )[0]
            )

            selected_distance = (
                distances[cluster]
            )

            max_distance = np.max(
                distances
            )

            cluster_confidence = (
                1 -
                (
                    selected_distance /
                    (
                        max_distance + 1e-8
                    )
                )
            ) * 100

            cluster_confidence = float(
                np.clip(
                    cluster_confidence,
                    40,
                    99
                )
            )

            # ===============================
            # PERSONA
            # ===============================

            persona_mapping = {
                0: "😊 Pengguna Puas",
                1: "😠 Pengguna Mengalami Kendala",
                2: "😐 Pengguna Netral"
            }

            persona = persona_mapping.get(
                int(cluster),
                f"Cluster {cluster}"
            )

            st.write("")
            st.divider()

            # ===============================
            # HASIL SEJAJAR
            # ===============================

            col1, col2 = st.columns(2)

            # -----------------------------
            # CARD SENTIMEN
            # -----------------------------

            with col1:

                st.subheader(
                    "😊 Hasil Sentimen"
                )

                st.metric(
                    label="Sentimen",
                    value=sentimen.upper()
                )

                st.metric(
                    label="Confidence",
                    value=f"{confidence:.2f}%"
                )

                st.progress(
                    confidence / 100
                )

            # -----------------------------
            # CARD PERSONA
            # -----------------------------

            with col2:

                st.subheader(
                    "🧠 Persona Pengguna"
                )

                st.metric(
                    label="Persona",
                    value=persona
                )

                st.metric(
                    label="Confidence",
                    value=f"{cluster_confidence:.2f}%"
                )

                st.progress(
                    cluster_confidence / 100
                )

            st.write("")
            st.divider()

            # ===============================
            # INTERPRETASI
            # ===============================

            st.subheader(
                "Interpretasi Hasil"
            )

            if sentimen == "positif":
                 interpretasi = ("Positif"
                                 )
                 st.success(interpretasi)

            elif sentimen == "negatif":
                interpretasi = ("Negatif"
                                 )
                st.error(
                    interpretasi
                )

            else:
                interpretasi = ("Netral"
                                 )
                st.warning(interpretasi)

            # ===============================
            # HISTORY
            # ===============================

            if (
                "history"
                not in st.session_state
            ):
                st.session_state[
                    "history"
                ] = []

            st.session_state[
                "history"
            ].append({

                "Ulasan": teks,

                "Sentimen": sentimen,

                "Confidence Sentimen":
                    round(
                        confidence,
                        2
                    ),

                "Cluster":
                    int(cluster),

                "Persona":
                    persona,
                
                "Sentimen Cluster": interpretasi,
                    

                "Confidence Cluster":
                    round(
                        cluster_confidence,
                        2
                    )
            })
        
elif page == "Upload CSV":
    import pandas as pd

    st.markdown("""
    <div style="
        background: linear-gradient(90deg,#11998e,#38ef7d);
        padding:25px;
        border-radius:20px;
        text-align:center;
        color:white;
    ">
        <h1>UPLOAD DATASET</h1>
    </div>
    """, unsafe_allow_html=True)

    st.write("")

    # =====================================
    # UPLOAD FILE CARD
    # =====================================
    st.markdown("""
    <div style="
        background:white;
        padding:20px;
        border-radius:15px;
        box-shadow:0 4px 10px rgba(0,0,0,0.1);
    ">
    """, unsafe_allow_html=True)

    uploaded_file = st.file_uploader(
        "Upload File CSV Anda",
        type=["csv"]
    )

    st.markdown("</div>", unsafe_allow_html=True)

    st.write("")

    # =====================================
    # PROCESS FILE
    # =====================================
    if uploaded_file is not None:

        try:

            data = pd.read_csv(uploaded_file)

            st.subheader("PREVIEW DATASET")
            st.dataframe(data.head(), use_container_width=True)

            # =====================================
            # CEK KOLOM CONTENT
            # =====================================
            if "content" not in data.columns:

                st.error("❌ Kolom 'content' tidak ditemukan pada dataset!")

            else:

                with st.spinner("🔄 Sedang melakukan analisis..."):

                    # =====================================
                    # CLEANING DATA
                    # =====================================

                    data["content"] = (
                        data["content"]
                        .fillna("")
                        .astype(str)
                        .str.strip()
                    )

                    # hapus baris kosong
                    data = data[data["content"] != ""]

                    # reset index
                    data = data.reset_index(drop=True)

                    # cek apakah data masih ada
                    if len(data) == 0:

                        st.error(
                            "❌ Semua data pada kolom 'content' kosong."
                        )
                        st.stop()

                    # =====================================
                    # DEBUG INFO
                    # =====================================
                    st.info(
                        f"Jumlah data yang dianalisis: {len(data)}"
                    )

                    # =====================================
                    # TF-IDF
                    # =====================================
                    vector = tfidf.transform(data["content"])

                    # =====================================
                    # PREDIKSI SENTIMEN
                    # =====================================
                    data["Prediksi_Sentimen"] = svm_model.predict(vector)

                    # =====================================
                    # CLUSTERING
                    # =====================================
                    data["Cluster"] = kmeans_model.predict(vector)

                # =====================================
                # SUCCESS
                # =====================================
                st.success("✅ Analisis berhasil dilakukan!")

                st.write("")

                st.subheader("TABEL HASIL ANALISIS")

                st.dataframe(
                    data,
                    use_container_width=True
                )

                # =====================================
                # DISTRIBUSI SENTIMEN
                # =====================================
                st.subheader("Distribusi Sentimen")

                st.bar_chart(
                    data["Prediksi_Sentimen"].value_counts()
                )

                # =====================================
                # DOWNLOAD EXCEL
                # =====================================
                output = "hasil_prediksi.xlsx"

                data.to_excel(
                    output,
                    index=False
                )

                with open(output, "rb") as file:

                    st.download_button(
                        label="📥 Download Hasil Analisis",
                        data=file,
                        file_name="hasil_prediksi.xlsx",
                        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                        use_container_width=True
                    )

                # =====================================
                # SIMPAN KE SESSION
                # =====================================
                st.session_state["data_upload"] = data

        except Exception as e:

            st.error("Terjadi kesalahan saat memproses dataset.")
            st.exception(e)

elif page == "Riwayat":

    import pandas as pd
    import plotly.express as px

    st.markdown("""
    <div style="
        background: linear-gradient(90deg,#11998e,#38ef7d);
        padding:30px;
        border-radius:20px;
        text-align:center;
    ">
        <h1 style="color:white;">
            RIWAYAT HASIL PREDIKSI SENTIMEN
        </h1>
    </div>
    """, unsafe_allow_html=True)

    st.write("")

    # =====================================
    # CEK HISTORY
    # =====================================

    if "history" not in st.session_state:
        st.warning("Belum ada riwayat prediksi")
        st.stop()

    if len(st.session_state["history"]) == 0:
        st.warning("Belum ada riwayat prediksi")
        st.stop()

    # =====================================
    # DATA HISTORY
    # =====================================

    history = pd.DataFrame(
        st.session_state["history"]
    )

    # =====================================
    # CEK KOLOM
    # =====================================

    if "Cluster" not in history.columns:
        history["Cluster"] = 0

    if "Confidence Cluster" not in history.columns:
        history["Confidence Cluster"] = 0

    if "Confidence Sentimen" not in history.columns:
        history["Confidence Sentimen"] = 0

    

    # =====================================
    # STATISTIK
    # =====================================

    total_prediksi = len(history)

    sentimen_terbanyak = (
        history["Sentimen"]
        .value_counts()
        .idxmax()
    )

    avg_sentiment = (
        history["Confidence Sentimen"]
        .mean()
    )

    avg_cluster = (
        history["Confidence Cluster"]
        .mean()
    )

    # =====================================
    # # CARD STATISTIK DASHBOARD
    # # =====================================
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown(f"""
                    <div style="
                    background:linear-gradient(135deg,#11998e,#38ef7d);
                    padding:25px;
                    border-radius:20px;
                    text-align:center;
                    box-shadow:0 4px 12px rgba(0,0,0,0.15);
                    ">
                    <div style="
                    font-size:16px;
                    color:white;
                    font-weight:600;
                    ">
                    Total Prediksi
                    </div>
                    <div style="
                    font-size:28px;
                    color:white;
                    font-weight:700;
                    margin-top:10px;
                    ">
                    {total_prediksi}
                    </div>
                    </div>
                    """, unsafe_allow_html=True)
        with col2:
            st.markdown(f"""
                        <div style="
                        background:linear-gradient(135deg,#11998e,#38ef7d);
                        padding:25px;
                        border-radius:20px;
                        text-align:center;
                        box-shadow:0 4px 12px rgba(0,0,0,0.15);
                        ">
                        <div style="
                        font-size:16px;
                        color:white;
                        font-weight:600;
                        ">
                        Sentimen Dominan
                        </div>
                        <div style="
                    font-size:28px;
                    color:white;
                    font-weight:700;
                    margin-top:10px;
                    ">
                        {sentimen_terbanyak}
                        </div>
                        </div>
                        """, unsafe_allow_html=True)
            with col3:
                st.markdown(f"""
                            <div style="
                            background:linear-gradient(135deg,#11998e,#38ef7d);
                            padding:25px;
                            border-radius:20px;
                            text-align:center;
                            box-shadow:0 4px 12px rgba(0,0,0,0.15);
                            ">
                            <div style="
                            font-size:16px;
                            color:white;
                            font-weight:600;
                            ">
                            Avg Confidence Sentimen
                            </div>
                            <div style="
                    font-size:28px;
                    color:white;
                    font-weight:700;
                    margin-top:10px;
                    ">
                            {avg_sentiment:.2f}%
                            </div>
                            </div>
                            """, unsafe_allow_html=True)
                with col4:
                    st.markdown(f"""
                                <div style="
                                background:linear-gradient(135deg,#11998e,#38ef7d);
                                padding:25px;
                                border-radius:20px;
                                text-align:center;
                                box-shadow:0 4px 12px rgba(0,0,0,0.15);
                                ">
                                <div style="
                                font-size:16px;
                                color:white;
                                font-weight:600;
                                ">
                                Avg Confidence Cluster
                                </div>
                                <div style="
                    font-size:28px;
                    color:white;
                    font-weight:700;
                    margin-top:10px;
                    ">
                                {avg_cluster:.2f}%
                                </div>
                                </div>
                                """, unsafe_allow_html=True)
                    st.write("")

    # =====================================
    # TABEL
    # =====================================

    st.subheader("📋 Tabel Riwayat Prediksi")

    st.dataframe(
        history[
            [
                "Ulasan",
                "Sentimen",
                "Confidence Sentimen",
                "Persona",
                "Sentimen Cluster",
                "Confidence Cluster"
                
            ]
        ],
        use_container_width=True
    )

    st.write("")

    # =====================================
    # DOWNLOAD + HAPUS
    # =====================================

    output_file = "riwayat_prediksi.xlsx"

    history.to_excel(
        output_file,
        index=False
    )

    col1, col2 = st.columns(2)

    with col1:

        with open(output_file, "rb") as file:

            st.download_button(
                label="📥 Download Riwayat",
                data=file,
                file_name="riwayat_prediksi.xlsx",
                use_container_width=True
            )

    with col2:

        if st.button(
            "🗑 Hapus Riwayat",
            use_container_width=True
        ):

            st.session_state["history"] = []

            st.success(
                "Riwayat berhasil dihapus"
            )

            st.rerun()

    st.write("")

    # =====================================
    # VISUALISASI
    # =====================================

    col1, col2 = st.columns(2)

    with col1:

        st.subheader(
            "📊 Distribusi Sentimen"
        )

        sentimen_counts = (
            history["Sentimen"]
            .value_counts()
            .reset_index()
        )

        sentimen_counts.columns = [
            "Sentimen",
            "Jumlah"
        ]

        fig_sentimen = px.pie(
            sentimen_counts,
            names="Sentimen",
            values="Jumlah",
            hole=0.5
        )

        fig_sentimen.update_traces(
            textposition="inside",
            textinfo="percent+label"
        )

        st.plotly_chart(
            fig_sentimen,
            use_container_width=True
        )

    with col2:

        st.subheader(
            "🧠 Distribusi Persona"
        )

        persona_counts = (
            history["Persona"]
            .value_counts()
            .reset_index()
        )

        persona_counts.columns = [
            "Persona",
            "Jumlah"
        ]

        fig_persona = px.bar(
            persona_counts,
            x="Persona",
            y="Jumlah",
            text_auto=True
        )

        st.plotly_chart(
            fig_persona,
            use_container_width=True
        )

    st.write("")

elif page == "Pemodelan Supervised":
    import matplotlib.pyplot as plt
    from wordcloud import WordCloud
    import pandas as pd

    st.markdown("""
    <div style="
        background: linear-gradient(90deg,#11998e,#38ef7d);
        padding:20px;
        border-radius:15px;
        text-align:center;
        margin-bottom:20px;
        color:white;
    ">
        <h1>VISUALISASI SENTIMEN</h1>
        
    </div>
    """, unsafe_allow_html=True)

    if "data_upload" not in st.session_state:

        st.warning("Silakan upload dataset terlebih dahulu")

    else:

        data = st.session_state["data_upload"]

        # =====================================
        # HITUNG SENTIMEN
        # =====================================
        positif = data[data['Prediksi_Sentimen'] == "positif"].shape[0]
        negatif = data[data['Prediksi_Sentimen'] == "negatif"].shape[0]
        netral  = data[data['Prediksi_Sentimen'] == "netral"].shape[0]

        # =====================================
        # CARD METRIC
        # =====================================
        col1, col2, col3 = st.columns(3)

        col1.markdown(f"""
        <div style="
            background:linear-gradient(135deg,#00c853,#64dd17);
            padding:20px;
            border-radius:15px;
            text-align:center;
            color:white;
        ">
            <h3>😊 Positif</h3>
            <h1>{positif}</h1>
        </div>
        """, unsafe_allow_html=True)

        col2.markdown(f"""
        <div style="
            background:linear-gradient(135deg,#d50000,#ff1744);
            padding:20px;
            border-radius:15px;
            text-align:center;
            color:white;
        ">
            <h3>😠 Negatif</h3>
            <h1>{negatif}</h1>
        </div>
        """, unsafe_allow_html=True)

        col3.markdown(f"""
        <div style="
            background:linear-gradient(135deg,#2962ff,#448aff);
            padding:20px;
            border-radius:15px;
            text-align:center;
            color:white;
        ">
            <h3>😐 Netral</h3>
            <h1>{netral}</h1>
        </div>
        """, unsafe_allow_html=True)

        st.write("")

        # =====================================
        # PIE CHART (RAPI)
        # =====================================
        st.subheader("PERSENTASE SENTIMEN")

        fig1, ax1 = plt.subplots(figsize=(5,5))  # 🔥 diperkecil

        ax1.pie(
            [positif, negatif, netral],
            labels=["Positif", "Negatif", "Netral"],
            autopct='%1.1f%%',
            startangle=90,
            colors=['#00c853', '#d50000', '#2962ff']
        )

        ax1.axis('equal')

        st.pyplot(fig1, use_container_width=False)

        st.write("")

        # =====================================
        # WORDCLOUD (SEJAJAR 3 KOLOM)
        # =====================================
        st.subheader("WORD CLOUD SENTIMEN")

        col1, col2, col3 = st.columns(3)

        # ================= POSITIF =================
        with col1:
            st.markdown("### 😊 Positif")

            text_pos = " ".join(
                data[data['Prediksi_Sentimen']=="positif"]['content'].astype(str)
            )

            if text_pos.strip():
                wc_pos = WordCloud(
                    width=400,
                    height=300,
                    background_color='white',
                    colormap='Greens'
                ).generate(text_pos)

                fig, ax = plt.subplots(figsize=(4,3))
                ax.imshow(wc_pos, interpolation='bilinear')
                ax.axis("off")
                st.pyplot(fig)
            else:
                st.info("Tidak ada data positif")

        # ================= NEGATIF =================
        with col2:
            st.markdown("### 😠 Negatif")

            text_neg = " ".join(
                data[data['Prediksi_Sentimen']=="negatif"]['content'].astype(str)
            )

            if text_neg.strip():
                wc_neg = WordCloud(
                    width=400,
                    height=300,
                    background_color='white',
                    colormap='Reds'
                ).generate(text_neg)

                fig, ax = plt.subplots(figsize=(4,3))
                ax.imshow(wc_neg, interpolation='bilinear')
                ax.axis("off")
                st.pyplot(fig)
            else:
                st.info("Tidak ada data negatif")

        # ================= NETRAL =================
        with col3:
            st.markdown("### 😐 Netral")

            text_net = " ".join(
                data[data['Prediksi_Sentimen']=="netral"]['content'].astype(str)
            )

            if text_net.strip():
                wc_net = WordCloud(
                    width=400,
                    height=300,
                    background_color='white',
                    colormap='Blues'
                ).generate(text_net)

                fig, ax = plt.subplots(figsize=(4,3))
                ax.imshow(wc_net, interpolation='bilinear')
                ax.axis("off")
                st.pyplot(fig)
            else:
                st.info("Tidak ada data netral")

                import plotly.express as px
    import pandas as pd
    from sklearn.metrics import (
        confusion_matrix,
        accuracy_score,
        classification_report
    )

    st.markdown("""
    <div style="
        background: linear-gradient(90deg,#11998e,#38ef7d);
        padding:25px;
        border-radius:20px;
        text-align:center;
        color:white;
    ">
        <h1></h1>
        
    </div>
    """, unsafe_allow_html=True)

    st.write("")

    # =====================================
    # STYLE
    # =====================================
    st.markdown("""
    <style>

    .metric-card {
        background: linear-gradient(135deg, #1e3c72, #2a5298);
        padding: 25px;
        border-radius: 20px;
        color: white;
        text-align: center;
        box-shadow: 0px 4px 15px rgba(0,0,0,0.2);
        margin-bottom: 20px;
    }

    .metric-title {
        font-size:16px;
        font-weight:bold;
    }

    .metric-value {
        font-size:32px;
        font-weight:bold;
    }

    .section-title {
        font-size:24px;
        font-weight:bold;
        color:#11998e;
        margin-top:20px;
        margin-bottom:10px;
    }

    </style>
    """, unsafe_allow_html=True)

    # =====================================
    # CHECK DATA
    # =====================================
    if "data_upload" not in st.session_state:
        st.warning("⚠️ Upload dataset terlebih dahulu")

    else:

        data = st.session_state["data_upload"]

        if "content" not in data.columns:
            st.error("Kolom 'content' tidak ditemukan")

        elif "sentimen" not in data.columns:
            st.error("Kolom 'sentimen' tidak ditemukan")

        else:

            # =====================================
            # PREPROCESS
            # =====================================
            X_test = data["content"].astype(str)
            y_true = data["sentimen"]

            vector = tfidf.transform(X_test)
            y_pred = svm_model.predict(vector)

            # =====================================
            # METRICS
            # =====================================
            accuracy = accuracy_score(y_true, y_pred)
            report = classification_report(y_true, y_pred, output_dict=True)

            precision = report["weighted avg"]["precision"]
            recall = report["weighted avg"]["recall"]
            f1 = report["weighted avg"]["f1-score"]

            data["Prediksi"] = y_pred

            # =====================================
            # METRIC CARD
            # =====================================
            col1, col2, col3, col4 = st.columns(4)

            with col1:
                st.markdown(f"""
                <div class='metric-card'>
                    <div class='metric-title'>Accuracy</div>
                    <div class='metric-value'>{accuracy:.2%}</div>
                </div>
                """, unsafe_allow_html=True)

            with col2:
                st.markdown(f"""
                <div class='metric-card'>
                    <div class='metric-title'>Precision</div>
                    <div class='metric-value'>{precision:.2%}</div>
                </div>
                """, unsafe_allow_html=True)

            with col3:
                st.markdown(f"""
                <div class='metric-card'>
                    <div class='metric-title'>Recall</div>
                    <div class='metric-value'>{recall:.2%}</div>
                </div>
                """, unsafe_allow_html=True)

            with col4:
                st.markdown(f"""
                <div class='metric-card'>
                    <div class='metric-title'>F1-Score</div>
                    <div class='metric-value'>{f1:.2%}</div>
                </div>
                """, unsafe_allow_html=True)

            st.write("")

            # =====================================
            # CONFUSION MATRIX (PLOTLY INTERAKTIF)
            # =====================================
            st.markdown("""
            <div class='section-title'>
                Confusion Matrix
            </div>
            """, unsafe_allow_html=True)

            cm = confusion_matrix(y_true, y_pred)

            labels = ["Negatif", "Netral", "Positif"]

            cm_df = pd.DataFrame(cm, index=labels, columns=labels).reset_index()
            cm_df = cm_df.melt(id_vars="index")

            cm_df.columns = ["Actual", "Predicted", "Value"]

            fig = px.density_heatmap(
                cm_df,
                x="Predicted",
                y="Actual",
                z="Value",
                text_auto=True,
                color_continuous_scale="Blues"
            )

            fig.update_layout(
                height=420,
                margin=dict(t=40, b=20, l=20, r=20)
            )

            st.plotly_chart(fig, use_container_width=True)

            # =====================================
            # CLASSIFICATION REPORT
            # =====================================
            st.markdown("""
            <div class='section-title'>
                Classification Report
            </div>
            """, unsafe_allow_html=True)

            report_df = pd.DataFrame(report).transpose()

            st.dataframe(report_df, use_container_width=True)

            # =====================================
            # HASIL PREDIKSI
            # =====================================
            st.markdown("""
            <div class='section-title'>
                Hasil Prediksi Model
            </div>
            """, unsafe_allow_html=True)

            st.dataframe(
                data[["content", "sentimen", "Prediksi"]],
                use_container_width=True,
                height=350
            )
            
elif page == "Preprocessing":

    import pandas as pd
    import numpy as np
    import re
    import nltk

    from nltk.tokenize import word_tokenize
    from nltk.corpus import stopwords
    from Sastrawi.Stemmer.StemmerFactory import StemmerFactory

    nltk.download("punkt")
    nltk.download("stopwords")

    st.markdown("""
    <div style="
        background: linear-gradient(90deg,#11998e,#38ef7d);
        padding:30px;
        border-radius:20px;
        text-align:center;
    ">
        <h1 style="color:white;">
            PREPROCESSING DAN PELABELAN DATA
        </h1>
    </div>
    """, unsafe_allow_html=True)

    st.write("")

    uploaded_file = st.file_uploader(
        "Upload Dataset CSV",
        type=["csv"]
    )

    if uploaded_file is not None:

        review = pd.read_csv(uploaded_file)

        st.subheader("📄 Dataset Awal")

        st.dataframe(
            review.head(),
            use_container_width=True
        )

        # =====================================
        # VALIDASI
        # =====================================

        if "content" not in review.columns:

            st.error(
                "Kolom 'content' tidak ditemukan."
            )

        else:

            with st.spinner(
                "Sedang melakukan preprocessing..."
            ):

                # =====================================
                # HAPUS DATA KOSONG
                # =====================================

                review = review.dropna(
                    subset=["content"]
                )

                review = review[
                    review["content"]
                    .astype(str)
                    .str.strip() != ""
                ]

                review = review.reset_index(
                    drop=True
                )

                # =====================================
                # 1. CASE FOLDING
                # =====================================

                def case_folding(text):

                    if pd.isna(text):
                        return ""

                    text = str(text)
                    text = text.lower()
                    text = text.strip()

                    text = " ".join(
                        text.split()
                    )

                    return text

                review["CaseFolding"] = review[
                    "content"
                ].apply(case_folding)

                # =====================================
                # 2. CLEANING
                # =====================================

                def clean_text(text):

                    text = str(text)

                    text = re.sub(
                        r'[\U00010000-\U0010ffff]',
                        '',
                        text
                    )

                    text = re.sub(
                        r'http\S+|www\S+',
                        '',
                        text
                    )

                    text = re.sub(
                        r'@\w+',
                        '',
                        text
                    )

                    text = re.sub(
                        r'#\w+',
                        '',
                        text
                    )

                    text = re.sub(
                        r'\d+',
                        '',
                        text
                    )

                    text = re.sub(
                        r'[^\w\s]',
                        '',
                        text
                    )

                    text = re.sub(
                        r'\s+',
                        ' ',
                        text
                    ).strip()

                    return text

                review["Cleaning"] = review[
                    "CaseFolding"
                ].apply(clean_text)

                # =====================================
                # HAPUS DUPLIKAT
                # =====================================

                review = review.drop_duplicates(
                    subset=["Cleaning"]
                )

                review = review.reset_index(
                    drop=True
                )

                # =====================================
                # 3. TOKENIZING
                # =====================================

                def tokenizing_text(text):

                    return word_tokenize(
                        str(text)
                    )

                review["Tokenizing"] = review[
                    "Cleaning"
                ].apply(tokenizing_text)

                # =====================================
                # 4. STOPWORD REMOVAL
                # =====================================

                daftar_stopword = (
                    stopwords.words(
                        "indonesian"
                    )
                )

                daftar_stopword.extend([
                    "yg",
                    "dg",
                    "rt",
                    "aja",
                    "nih",
                    "sih",
                    "nya"
                ])

                daftar_stopword = set(
                    daftar_stopword
                )

                def stopword_removal(words):

                    return [
                        word
                        for word in words
                        if word not in daftar_stopword
                    ]

                review["WithoutStopwords"] = (
                    review["Tokenizing"]
                    .apply(stopword_removal)
                )

                # =====================================
                # 5. NORMALIZATION
                # =====================================

                normalization_dict = {

                    "gk": "tidak",
                    "ga": "tidak",
                    "nggak": "tidak",
                    "tdk": "tidak",

                    "bgt": "banget",
                    "bgtt": "banget",

                    "apk": "aplikasi",
                    "app": "aplikasi",

                    "mantul": "mantap",

                    "sm": "sama",

                    "krn": "karena",

                    "utk": "untuk",

                    "dr": "dari",

                    "yg": "yang",

                    "udh": "sudah",

                    "blm": "belum"
                }

                def normalize_text(text):

                    if isinstance(
                        text,
                        list
                    ):
                        words = text

                    else:
                        words = str(
                            text
                        ).split()

                    hasil = []

                    for word in words:

                        if word in normalization_dict:

                            hasil.append(
                                normalization_dict[
                                    word
                                ]
                            )

                        else:

                            hasil.append(
                                word
                            )

                    return " ".join(
                        hasil
                    )

                review["Normalized"] = (
                    review["WithoutStopwords"]
                    .apply(normalize_text)
                )

                # =====================================
                # 6. STEMMING
                # =====================================

                factory = (
                    StemmerFactory()
                )

                stemmer = (
                    factory
                    .create_stemmer()
                )

                def stemming_text(text):

                    if pd.isna(text):
                        return ""

                    return stemmer.stem(
                        str(text)
                    )

                review["Stemming"] = (
                    review["Normalized"]
                    .apply(stemming_text)
                )

                # =====================================
                # 7. KAMUS SENTIMEN
                # =====================================

                positif = [

                    "bagus",
                    "baik",
                    "cepat",
                    "mudah",
                    "mantap",
                    "bantu",
                    "lengkap",
                    "praktis",
                    "guna",
                    "puas",
                    "nyaman",
                    "stabil",
                    "aman",
                    "suka",
                    "hasil",
                    "puas"
                ]

                negatif = [

                    "buruk",
                    "error",
                    "gagal",
                    "lambat",
                    "lemot",
                    "bug",
                    "susah",
                    "ribet",
                    "kecewa",
                    "ganggu",
                    "masalah",
                    "kendala",
                    "rusak",
                    "jelek",
                    "parah",
                    "payah"
                ]

                # =====================================
                # 8. LABELING
                # =====================================

                def sentiment_analysis(text):

                    if pd.isna(text):
                        return 0, "netral"

                    words = str(
                        text
                    ).split()

                    score = 0

                    for word in words:

                        if word in positif:
                            score += 1

                        elif word in negatif:
                            score -= 1

                    if score > 0:
                        label = "positif"

                    elif score < 0:
                        label = "negatif"

                    else:
                        label = "netral"

                    return score, label

                hasil = review[
                    "Stemming"
                ].apply(
                    sentiment_analysis
                )

                review["score"] = (
                    hasil.apply(
                        lambda x: x[0]
                    )
                )

                review["sentimen"] = (
                    hasil.apply(
                        lambda x: x[1]
                    )
                )

            st.success(
                "✅ Preprocessing dan pelabelan berhasil dilakukan"
            )

            st.write("")

            # =====================================
            # HASIL
            # =====================================

            st.subheader(
                "📊 Hasil Preprocessing"
            )

            st.dataframe(
                review[
                    [
                        "content",
                        "CaseFolding",
                        "Cleaning",
                        "Normalized",
                        "Stemming",
                        "score",
                        "sentimen"
                    ]
                ],
                use_container_width=True
            )

            st.write("")

            # =====================================
            # DISTRIBUSI LABEL
            # =====================================

            st.subheader(
                "📈 Distribusi Sentimen"
            )

            st.bar_chart(
                review[
                    "sentimen"
                ].value_counts()
            )

            st.write("")

            # =====================================
            # # DOWNLOAD CSV
            # # =====================================
            output_file = "hasil_preprocessing_label.csv"
            review.to_csv(
                output_file,
                index=False
                )
            with open(
                output_file,
                "rb"
                ) as file:
                st.download_button(
        label="📥 Download Hasil Preprocessing",
        data=file,
        file_name=output_file,
        mime="text/csv",
        use_container_width=True
    )
elif page == "Pemodelan Unsupervised":
    import matplotlib.pyplot as plt
    from wordcloud import WordCloud
    import pandas as pd

    st.markdown("""
    <div style="
        background: linear-gradient(90deg,#11998e,#38ef7d);
        padding:25px;
        border-radius:20px;
        text-align:center;
        color:white;
    ">
        <h1>CLUSTER PERSONA ANALISIS</h1>
    </div>
    """, unsafe_allow_html=True)

    st.write("")

    # =====================================
    # CHECK DATA
    # =====================================
    if "data_upload" not in st.session_state:

        st.warning("Silakan upload dataset terlebih dahulu")

    else:

        data = st.session_state["data_upload"]

        # =====================================
        # HITUNG CLUSTER
        # =====================================
        jumlah_cluster = data["Cluster"].value_counts().sort_index()

        total_data = len(data)

        cluster_terbesar = jumlah_cluster.idxmax()

        jumlah_terbesar = jumlah_cluster.max()

        # =====================================
        # CARD METRIC (MODERN)
        # =====================================

        col1, col2, col3 = st.columns(3)

        col1.markdown(f"""
        <div style="
            background: linear-gradient(135deg,#00c853,#64dd17);
            padding:20px;
            border-radius:15px;
            text-align:center;
            color:white;
            box-shadow:0px 4px 10px rgba(0,0,0,0.2);
        ">
            <h4>Jumlah Cluster</h4>
            <h2>{len(jumlah_cluster)}</h2>
        </div>
        """, unsafe_allow_html=True)

        col2.markdown(f"""
        <div style="
            background: linear-gradient(135deg,#2962ff,#448aff);
            padding:20px;
            border-radius:15px;
            text-align:center;
            color:white;
            box-shadow:0px 4px 10px rgba(0,0,0,0.2);
        ">
            <h4>Total Data</h4>
            <h2>{total_data}</h2>
        </div>
        """, unsafe_allow_html=True)

        col3.markdown(f"""
        <div style="
            background: linear-gradient(135deg,#ff6d00,#ff9100);
            padding:20px;
            border-radius:15px;
            text-align:center;
            color:white;
            box-shadow:0px 4px 10px rgba(0,0,0,0.2);
        ">
            <h4>Cluster Terbesar</h4>
            <h2>Cluster {cluster_terbesar}</h2>
        </div>
        """, unsafe_allow_html=True)

        st.write("")

        # =====================================
        # CLUSTER DOMINAN INFO
        # =====================================
        st.markdown(f"""
        <div style="
            background-color:#e8f5e9;
            padding:20px;
            border-radius:15px;
            border-left:8px solid #4caf50;
        ">
            <h3 style="color:#2e7d32;">
                CLUSTER PERSONA DOMINAN
            </h3>
            <p style="font-size:18px;">
                Cluster <b>{cluster_terbesar}</b>
                memiliki jumlah data terbanyak yaitu
                <b>{jumlah_terbesar}</b> data pengguna.
            </p>
        </div>
        """, unsafe_allow_html=True)

        st.write("")

        # =====================================
        # DISTRIBUSI CLUSTER
        # =====================================
        st.subheader("DISTRIBUSI CLUSTER PERSONA")

        fig4, ax4 = plt.subplots(figsize=(8,4))  # 🔥 diperkecil

        bars = ax4.bar(
            jumlah_cluster.index.astype(str),
            jumlah_cluster.values,
            color="#11998e"
        )

        ax4.set_title("Distribusi Cluster Persona", fontsize=14)
        ax4.set_xlabel("Cluster")
        ax4.set_ylabel("Jumlah Data")

        for bar in bars:
            yval = bar.get_height()
            ax4.text(bar.get_x() + bar.get_width()/2,
                     yval + 1,
                     int(yval),
                     ha='center',
                     fontsize=10)

        st.pyplot(fig4)

        st.write("")

        # =====================================
        # PERSENTASE CLUSTER
        # =====================================
        st.subheader("PERSENTASE CLUSTER")

        for i in jumlah_cluster.index:

            persen = (jumlah_cluster[i] / total_data) * 100

            st.write(f"Cluster {i} ({persen:.1f}%)")
            st.progress(float(persen / 100))

        st.write("")

        # =====================================
        # PILIH CLUSTER
        # =====================================
        st.subheader("DETAIL PERSONA CLUSTER")

        selected_cluster = st.selectbox(
            "Pilih Cluster Persona",
            sorted(data["Cluster"].unique())
        )

        cluster_data = data[data["Cluster"] == selected_cluster]

        st.write(
            f"Jumlah Data pada Cluster {selected_cluster}:",
            len(cluster_data)
        )

        # =====================================
        # SENTIMEN DOMINAN
        # =====================================
        sentimen_dominan = cluster_data["Prediksi_Sentimen"].value_counts()
        dominant_sentiment = sentimen_dominan.idxmax()

        st.info(f"""
        Persona Cluster {selected_cluster}
        didominasi oleh sentimen '{dominant_sentiment}'
        """)

        # =====================================
        # TABEL
        # =====================================
        st.subheader("CONTOH KOMENTAR")

        st.dataframe(
            cluster_data[["content", "Prediksi_Sentimen"]].head(10),
            use_container_width=True
        )

        # =====================================
        # WORDCLOUD
        # =====================================
        st.subheader("WORD CLOUD CLUSTER")

        text_cluster = " ".join(cluster_data["content"].astype(str))

        wordcloud = WordCloud(
            width=800,
            height=300,
            background_color='white',
            colormap='viridis'
        ).generate(text_cluster)

        fig5, ax5 = plt.subplots(figsize=(8,3))  # 🔥 diperkecil

        ax5.imshow(wordcloud, interpolation='bilinear')
        ax5.axis("off")

        st.pyplot(fig5)

        # =====================================
        # INSIGHT
        # =====================================
        st.subheader("INSIGHT PERSONA")

        if dominant_sentiment == "positif":

            st.success(f"""
            Cluster {selected_cluster} menunjukkan pengguna yang puas dan memberikan ulasan positif.
            """)

        elif dominant_sentiment == "negatif":

            st.error(f"""
            Cluster {selected_cluster} menunjukkan adanya ketidakpuasan pengguna terhadap aplikasi.
            """)

        else:

            st.warning(f"""
            Cluster {selected_cluster} menunjukkan pengguna dengan opini netral.
            """)

elif page == "Tentang":


    st.markdown("""
    <div style="
        background: linear-gradient(90deg,#11998e,#38ef7d);
        padding:30px;
        border-radius:20px;
        text-align:center;
    ">
        <h1 style="color:white;">
            TENTANG APLIKASI
        </h1>
    </div>
    """, unsafe_allow_html=True)

    st.write("")

    st.markdown("""
    <div style="
        background:white;
        padding:20px;
        border-radius:15px;
        box-shadow:0px 4px 10px rgba(0,0,0,0.1);
    ">
        <h3>TENTANG APLIKASI</h3>
        <p>
            Aplikasi ini merupakan sistem analisis sentimen yang dibangun
        untuk menganalisis ulasan pengguna aplikasi M-Pajak.
        Sistem menggunakan metode Machine Learning yaitu
        Support Vector Machine (SVM) untuk klasifikasi sentimen
        serta K-Means Clustering untuk segmentasi atau pengelompokan
        persona pengguna berdasarkan pola ulasan yang diberikan.
        </p>
    </div>
    """, unsafe_allow_html=True)

    st.write("")


    st.markdown("""
    <div style="
        text-align:center;
        color:#555;
    "><br><br><br><br><br><br>
        © 2026 Sistem Analisis Sentimen M-Pajak
    </div>
    """, unsafe_allow_html=True)

    
