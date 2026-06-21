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

    if st.button("Pemodelan Supervised", use_container_width=True):
        st.session_state.page = "Pemodelan Supervised"

    if st.button("Pemodelan Unsupervised", use_container_width=True):
        st.session_state.page = "Pemodelan Unsupervised"
    if st.button("Perbandingan", use_container_width=True):
        st.session_state.page = "Perbandingan"


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
                    "HASIL SUPERVISED"
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
                    "HASIL UNSUPERVISED"
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
    import numpy as np
    import re
    import nltk
    import streamlit as st

    from nltk.tokenize import word_tokenize
    from nltk.corpus import stopwords
    from Sastrawi.Stemmer.StemmerFactory import StemmerFactory

    # ================================
# NLTK RESOURCE
# ================================

nltk.download("punkt")
nltk.download("punkt_tab")
nltk.download("stopwords")

    st.markdown("""
    <div style="
        background: linear-gradient(90deg,#11998e,#38ef7d);
        padding:25px;
        border-radius:20px;
        text-align:center;
        color:white;
    ">
        <h1>INPUT MANUAL + PREPROCESSING + SENTIMEN + SVM + CLUSTER</h1>
    </div>
    """, unsafe_allow_html=True)

    st.write("")

    # ================================
    # INPUT
    # ================================

    df_template = pd.DataFrame({
        "userName": [""],
        "content": [""]
    })

    data = st.data_editor(
        df_template,
        num_rows="dynamic",
        use_container_width=True
    )

    if data is not None:

        try:

            data = data.dropna(subset=["content"])
            data = data[data["content"].astype(str).str.strip() != ""]
            data = data.reset_index(drop=True)

            if len(data) == 0:
                st.warning("Data masih kosong")
                st.stop()

            st.subheader("📄 DATA INPUT")
            st.dataframe(data, use_container_width=True)

            # ================================
            # CASE FOLDING
            # ================================

            data["CaseFolding"] = (
                data["content"]
                .astype(str)
                .str.lower()
                .str.strip()
            )

            # ================================
            # CLEANING
            # ================================

            def clean_text(text):

                text = str(text)

                text = re.sub(r"http\S+|www\S+", "", text)
                text = re.sub(r"@\w+", "", text)
                text = re.sub(r"#\w+", "", text)
                text = re.sub(r"\d+", "", text)
                text = re.sub(r"[^\w\s]", "", text)
                text = re.sub(r"\s+", " ", text)

                return text.strip()

            data["Cleaning"] = data["CaseFolding"].apply(clean_text)

            # ================================
# TOKENIZING
# ================================

def tokenize_text(text):

    try:
        return word_tokenize(
            str(text),
            language="english"
        )

    except:

        return str(text).split()



data["Tokenizing"] = (
    data["Cleaning"]
    .apply(tokenize_text)
)

            # ================================
            # STOPWORD
            # ================================

            stop_words = set(
                stopwords.words("indonesian")
            )

            stop_words.update([
                "yg",
                "dg",
                "aja",
                "nih",
                "sih",
                "nya"
            ])

            data["WithoutStopwords"] = (
                data["Tokenizing"]
                .apply(
                    lambda x: [
                        w for w in x
                        if w not in stop_words
                    ]
                )
            )

            # ================================
            # NORMALIZATION
            # ================================

            norm_dict = {
                "gk": "tidak",
                "ga": "tidak",
                "nggak": "tidak",
                "tdk": "tidak",
                "bgt": "banget",
                "apk": "aplikasi",
                "app": "aplikasi",
                "krn": "karena",
                "utk": "untuk",
                "dr": "dari",
                "udh": "sudah",
                "blm": "belum"
            }

            data["Normalized"] = (
                data["WithoutStopwords"]
                .apply(
                    lambda x: " ".join(
                        [
                            norm_dict.get(w, w)
                            for w in x
                        ]
                    )
                )
            )

            # ================================
            # STEMMING
            # ================================

            factory = StemmerFactory()
            stemmer = factory.create_stemmer()

            data["Stemming"] = (
                data["Normalized"]
                .apply(
                    lambda x: stemmer.stem(str(x))
                )
            )

            # ================================
            # SENTIMEN LEXICON
            # ================================

            positif = [
                "bagus",
                "baik",
                "cepat",
                "mudah",
                "mantap",
                "bantu",
                "lengkap",
                "praktis",
                "puas",
                "nyaman",
                "suka",
                "aman"
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
                "parah",
                "rusak"
            ]

            def sentiment(text):

                score = 0

                for word in str(text).split():

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

            result = data["Stemming"].apply(sentiment)

            data["score"] = result.apply(lambda x: x[0])
            data["sentimen"] = result.apply(lambda x: x[1])

            # ================================
            # TF-IDF
            # ================================

            vector = tfidf.transform(
                data["content"]
            )

            # ================================
            # SVM
            # ================================

            data["Prediksi_SVM"] = (
                svm_model.predict(vector)
            )

            # ================================
            # CONFIDENCE SVM
            # ================================

            if hasattr(svm_model, "predict_proba"):

                probability = (
                    svm_model.predict_proba(vector)
                )

                confidence_svm = (
                    np.max(probability, axis=1)
                    * 100
                )

            else:

                decision = (
                    svm_model.decision_function(vector)
                )

                confidence_svm = (
                    abs(decision)
                    /
                    (np.max(abs(decision)) + 1e-10)
                ) * 100

            data["Confidence_SVM"] = (
                confidence_svm
                .round(2)
                .astype(str)
                + "%"
            )

            # ================================
            # # KMEANS CLUSTER
            # ================================
            cluster_pred = (
                kmeans_model.predict(vector)
                )
            data["Cluster"] = cluster_pred
            # ================================
            # # SENTIMEN CLUSTER
            # # ================================
            data["Sentimen_Cluster"] = (
                data["Prediksi_SVM"]
                )

            # ================================
            # CONFIDENCE CLUSTER
            # ================================

            distance = (
                kmeans_model.transform(vector)
            )

            nearest_distance = np.min(
                distance,
                axis=1
            )

            furthest_distance = np.max(
                distance,
                axis=1
            )

            confidence_cluster = (
                1 -
                (
                    nearest_distance
                    /
                    (furthest_distance + 1e-10)
                )
            ) * 100

            data["Confidence_Cluster"] = (
                confidence_cluster
                .round(2)
                .astype(str)
                + "%"
            )

            # ================================
            # OUTPUT
            # ================================

            st.success("✅ Proses selesai")

            st.subheader("📊 HASIL AKHIR")

            st.dataframe(
                data[
                    [
                        "userName",
                        "content",
                        "CaseFolding",
                        "Cleaning",
                        "Tokenizing",
                        "WithoutStopwords",
                        "Normalized",
                        "Stemming",
                        "score",
                        "sentimen",
                        "Prediksi_SVM",
                        "Confidence_SVM",
                        "Cluster",
                        "Sentimen_Cluster",
                        "Confidence_Cluster"
                    ]
                ],
                use_container_width=True
            )

            st.subheader("📈 Distribusi Sentimen")

            st.bar_chart(
                data["sentimen"]
                .value_counts()
            )

            st.session_state["data_upload"] = data

        except Exception as e:

            st.error("Terjadi kesalahan")
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
    import plotly.express as px
    import pandas as pd

    from wordcloud import WordCloud

    from sklearn.metrics import (
        confusion_matrix,
        accuracy_score,
        classification_report
    )


    # ===============================
    # STYLE
    # ===============================

    st.markdown("""
    <style>

    .header-card{
        background:linear-gradient(135deg,#11998e,#38ef7d);
        padding:30px;
        border-radius:25px;
        color:white;
        text-align:center;
        box-shadow:0 8px 25px rgba(0,0,0,.2);
    }


    .card{
        padding:25px;
        border-radius:22px;
        box-shadow:0 6px 20px rgba(0,0,0,.15);
        margin-bottom:20px;
    }


    .metric{
        color:white;
        text-align:center;
        padding:20px;
        border-radius:20px;
    }


    .metric h1{
        font-size:35px;
    }


    </style>
    """,
    unsafe_allow_html=True)



    # ===============================
    # HEADER
    # ===============================


    st.markdown("""
    <div class="header-card">

    <h1>VISUALISASI SENTIMEN</h1>

    </div>
    """,
    unsafe_allow_html=True)



    # ===============================
    # DATA CHECK
    # ===============================

    if "data_upload" not in st.session_state:

        st.warning(
            "⚠️ Upload dataset terlebih dahulu"
        )

        st.stop()



    data = st.session_state["data_upload"]




    # ===============================
    # PREDIKSI
    # ===============================


    vector = tfidf.transform(
        data["content"].astype(str)
    )


    y_true = data["sentimen"]


    y_pred = svm_model.predict(vector)


    data["Prediksi"] = y_pred


    st.session_state["data_upload"]=data




    # ===============================
    # SENTIMEN CARD
    # ===============================


    st.subheader(
        "DISTRIBUSI SENTIMEN"
    )


    positif = (
        data["sentimen"]=="positif"
    ).sum()


    negatif = (
        data["sentimen"]=="negatif"
    ).sum()


    netral = (
        data["sentimen"]=="netral"
    ).sum()



    c1,c2,c3 = st.columns(3)


    for col,title,value,color in [

        (c1,"😊 Positif",positif,"#00c853"),

        (c2,"😡 Negatif",negatif,"#d50000"),

        (c3,"😐 Netral",netral,"#2962ff")

    ]:


        with col:

            st.markdown(f"""

            <div class="metric"
            style="
            background:linear-gradient(135deg,{color},#222);
            ">


            <h3>{title}</h3>

            <h1>{value}</h1>


            </div>

            """,
            unsafe_allow_html=True)



    st.write("")



    # ===============================
    # WORDCLOUD
    # ===============================


    st.subheader(
        "WORDCLOUD SENTIMEN"
    )



    wc_col = st.columns(3)



    sentiment_list=[

        ("positif","😊 Positif","Greens"),

        ("negatif","😡 Negatif","Reds"),

        ("netral","😐 Netral","Blues")

    ]



    for col,(label,title,color) in zip(
        wc_col,
        sentiment_list
    ):


        with col:


            st.markdown(
            f"""
            <div class="card">

            <h3>{title}</h3>

            </div>
            """,
            unsafe_allow_html=True
            )


            text=" ".join(
                data[
                    data["sentimen"]==label
                ]["content"]
                .astype(str)
            )


            if text.strip():


                wc=WordCloud(

                    width=400,

                    height=300,

                    background_color="white",

                    colormap=color

                ).generate(text)



                fig,ax=plt.subplots(
                    figsize=(4,3)
                )


                ax.imshow(
                    wc,
                    interpolation="bilinear"
                )

                ax.axis("off")


                st.pyplot(fig)




    # ===============================
    # EVALUASI MODEL
    # ===============================


    st.subheader(
        "EVALUASI MODEL SVM"
    )



    accuracy=accuracy_score(
        y_true,
        y_pred
    )


    report=classification_report(
        y_true,
        y_pred,
        output_dict=True,
        zero_division=0
    )



    metrics=[

        ("Accuracy",report["accuracy"],"#00c853"),

        ("Precision",
        report["weighted avg"]["precision"],
        "#2979ff"),

        ("Recall",
        report["weighted avg"]["recall"],
        "#8e24aa"),

        ("F1 Score",
        report["weighted avg"]["f1-score"],
        "#ff6d00")

    ]



    cols=st.columns(4)



    for col,(title,value,color) in zip(
        cols,
        metrics
    ):

        with col:

            st.markdown(f"""

            <div class="metric"
            style="
            background:linear-gradient(135deg,{color},#111);
            ">


            <h3>{title}</h3>


            <h1>
            {value:.2%}
            </h1>


            </div>


            """,
            unsafe_allow_html=True)




    # ===============================
    # CONFUSION MATRIX
    # ===============================


    st.subheader(
        "CONFUSION MATRIX"
    )


    labels=sorted(
        set(y_true)
        .union(set(y_pred))
    )


    cm=confusion_matrix(
        y_true,
        y_pred,
        labels=labels
    )



    cm_df=pd.DataFrame(
        cm,
        index=labels,
        columns=labels
    )



    fig=px.imshow(

        cm_df,

        text_auto=True,

        color_continuous_scale="Teal",

        labels=dict(
            x="Prediksi",
            y="Aktual"
        )

    )


    fig.update_layout(
        height=450
    )


    st.plotly_chart(
        fig,
        use_container_width=True
    )



    # ===============================
    # REPORT
    # ===============================


    st.subheader(
        "CLASSIFICATION REPORT"
    )


    st.dataframe(
        pd.DataFrame(report)
        .transpose(),
        use_container_width=True
    )



    # ===============================
    # HASIL
    # ===============================


    st.subheader(
        "HASIL PREDIKSI SENTIMEN"
    )


    st.dataframe(

        data[
            [
            "userName",
            "content",
            "sentimen",
            "Prediksi"
            ]
        ],

        use_container_width=True,

        height=400
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
        # # DISTRIBUSI CLUSTER PERSONA MODERN
        # # =====================================
        
        cluster_df = pd.DataFrame({
            "Cluster":
                jumlah_cluster.index.astype(str),
                "Jumlah":
                    jumlah_cluster.values
                    })
        fig4 = px.bar(
            cluster_df,
            x="Cluster",
            y="Jumlah",
            text="Jumlah",
            color="Cluster",
            title="Distribusi Pengguna Berdasarkan Cluster",
            )
        fig4.update_traces(
            textposition="outside",
            marker_line_width=1
            )
        fig4.update_layout(
            height=450,
            showlegend=False,
            plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)",
            title={
                "text":
                    "Distribusi Cluster Persona",
                    "x":0.5,
                    "xanchor":"center"
                    },
            font=dict(
                size=14
                ),
            margin=dict(
                t=70,
                b=40,
                l=40,
                r=40
                )
            )
        st.plotly_chart(
            fig4,
            use_container_width=True
            )
        cluster_max = jumlah_cluster.idxmax()
        jumlah_max = jumlah_cluster.max()
        st.markdown(f"""

""",
unsafe_allow_html=True)
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
        sentimen_dominan = cluster_data["sentimen"].value_counts()
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
            cluster_data[["content", "sentimen"]].head(10),
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
elif page == "Perbandingan":

    import pandas as pd
    import plotly.express as px

    from sklearn.metrics import (
        accuracy_score,
        precision_score,
        recall_score,
        f1_score,
        silhouette_score
    )

    # =====================================
    # STYLE
    # =====================================

    st.markdown("""
    <style>

    .main-card{
        padding:25px;
        border-radius:25px;
        box-shadow:0 8px 25px rgba(0,0,0,0.15);
        margin-bottom:20px;
    }

    </style>
    """,
    unsafe_allow_html=True)

    # =====================================
    # HEADER
    # =====================================

    st.markdown("""
    <div class="main-card"
    style="
    background:linear-gradient(135deg,#11998e,#38ef7d);
    color:white;
    text-align:center;
    ">

    <h1>PERBANDINGAN MODEL</h1>

    <h3>
    Supervised Learning VS Unsupervised Learning
    </h3>

    </div>
    """,
    unsafe_allow_html=True)

    # =====================================
    # CHECK DATA
    # =====================================

    if "data_upload" not in st.session_state:

        st.warning(
            "⚠️ Silakan input dataset terlebih dahulu"
        )

        st.stop()

    data = st.session_state["data_upload"]

    # =====================================
    # AUTO PREDIKSI SVM
    # =====================================

    if "Prediksi_SVM" not in data.columns:

        vector = tfidf.transform(
            data["content"].astype(str)
        )

        data["Prediksi_SVM"] = (
            svm_model.predict(vector)
        )

    # =====================================
    # AUTO CLUSTER
    # =====================================

    if "Cluster" not in data.columns:

        vector = tfidf.transform(
            data["content"].astype(str)
        )

        data["Cluster"] = (
            kmeans_model.predict(vector)
        )

    st.session_state["data_upload"] = data

    # =====================================
    # METRIK SUPERVISED
    # =====================================

    svm_accuracy = 0
    svm_precision = 0
    svm_recall = 0
    svm_f1 = 0

    if (
        "sentimen" in data.columns
        and
        "Prediksi_SVM" in data.columns
    ):

        y_true = (
            data["sentimen"]
            .astype(str)
            .str.lower()
        )

        y_pred = (
            data["Prediksi_SVM"]
            .astype(str)
            .str.lower()
        )

        svm_accuracy = (
            accuracy_score(
                y_true,
                y_pred
            ) * 100
        )

        svm_precision = (
            precision_score(
                y_true,
                y_pred,
                average="weighted",
                zero_division=0
            ) * 100
        )

        svm_recall = (
            recall_score(
                y_true,
                y_pred,
                average="weighted",
                zero_division=0
            ) * 100
        )

        svm_f1 = (
            f1_score(
                y_true,
                y_pred,
                average="weighted",
                zero_division=0
            ) * 100
        )

    # =====================================
    # METRIK UNSUPERVISED
    # =====================================

    cluster_score = 0

    try:

        vector_cluster = tfidf.transform(
            data["content"].astype(str)
        )

        cluster_score = (
            silhouette_score(
                vector_cluster,
                data["Cluster"]
            ) * 100
        )

    except:
        cluster_score = 0

    # =====================================
    # CARD MODEL
    # =====================================

    col1, col2 = st.columns(2)

    with col1:

        st.markdown("""
        <div class="main-card"
        style="
        background:linear-gradient(135deg,#2196f3,#21cbf3);
        color:white;
        ">

        <h2>SUPERVISED</h2>

        <h3>SVM Classification</h3>

        <p>
        Melakukan prediksi sentimen pengguna.
        </p>

        </div>
        """,
        unsafe_allow_html=True)

    with col2:

        st.markdown("""
        <div class="main-card"
        style="
        background:linear-gradient(135deg,#ff9800,#ff5722);
        color:white;
        ">

        <h2>UNSUPERVISED</h2>

        <h3>K-Means Clustering</h3>

        <p>
        Mengelompokkan pola ulasan pengguna.
        </p>

        </div>
        """,
        unsafe_allow_html=True)


    # =====================================
    # TABEL PERBANDINGAN
    # =====================================

    st.subheader(
        "RINGKASAN PERBANDINGAN"
    )

    comparison = pd.DataFrame({

        "Model": [
            "SVM",
            "K-Means"
        ],

        "Jenis": [
            "Supervised",
            "Unsupervised"
        ],

        "Output": [
            "Prediksi Sentimen",
            "Cluster Persona"
        ],

        "Jumlah Kelas": [
            len(data["Prediksi_SVM"].unique()),
            len(data["Cluster"].unique())
        ],

        "Evaluasi": [
            f"Akurasi {svm_accuracy:.2f}%",
            f"Akurasi {cluster_score:.2f}%"
        ]

    })

    st.dataframe(
        comparison,
        hide_index=True,
        use_container_width=True
    )

    # =====================================
    # INSIGHT
    # =====================================

    dominant_sentiment = (
        data["Prediksi_SVM"]
        .value_counts()
        .idxmax()
    )

    dominant_cluster = (
        data["Cluster"]
        .value_counts()
        .idxmax()
    )

    st.markdown(f"""

    <div style="
    background:linear-gradient(135deg,#141e30,#243b55);
    color:white;
    padding:25px;
    border-radius:20px;
    ">

    <h1 style="text-align:center;">
    SMART ANALYTICS INSIGHT
    </h1>

    <h2>
    Sentimen Dominan :
    {str(dominant_sentiment).upper()}
    </h2>

    <h2>
    Cluster Dominan :
    {dominant_cluster}
    </h2>

    <hr>

    <h3>
    Accuracy Supervised : {svm_accuracy:.2f}%<br>
    Accuracy Unsupervised : {cluster_score:.2f}%
    </h3>

    </div>

    """,
    unsafe_allow_html=True)

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

    
