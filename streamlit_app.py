import streamlit as st
import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing import image
from PIL import Image
import tempfile

# =====================================
# KONFIGURASI HALAMAN
# =====================================
st.set_page_config(
    page_title="Sistem Klasifikasi CNN",
    page_icon="🧠",
    layout="wide"
)

# =====================================
# CSS CUSTOM
# =====================================
st.markdown("""
<style>

/* Background */
.stApp{
    background-color:#2b2b2b;
}

/* Navbar */
.navbar{
    background:#000;
    padding:25px;
    border-radius:15px;
    text-align:center;
    margin-bottom:30px;
}

.navbar h1{
    color:white;
    margin:0;
    font-size:42px;
    font-weight:bold;
}

/* Card Upload */
.upload-card{
    background:white;
    padding:20px;
    border-radius:15px;
    box-shadow:0px 4px 15px rgba(0,0,0,0.2);
    min-height:220px;
}

.upload-title{
    text-align:center;
    font-size:26px;
    font-weight:bold;
    color:black;
    margin-bottom:20px;
}

/* Footer */
.footer{
    text-align:center;
    color:white;
    margin-top:30px;
}

</style>
""", unsafe_allow_html=True)
# =====================================
# NAVBAR
# =====================================
st.markdown("""
<div class="navbar">
    <h1>🧠 Sistem Klasifikasi CNN</h1>
</div>
""", unsafe_allow_html=True)

# =====================================
# DESKRIPSI
# =====================================
st.markdown("""
<div class="card">
    <h3 style='text-align:center;'>
        Upload model CNN (.keras) dan gambar untuk melakukan klasifikasi
    </h3>
</div>
""", unsafe_allow_html=True)

st.write("")

# =====================================
# KOLOM UPLOAD
# =====================================
col1, col2 = st.columns(2)

with col1:
    st.markdown(
        "<div class='upload-title'>📦 Upload Model</div>",
        unsafe_allow_html=True
    )

    model_file = st.file_uploader(
        "",
        type=["keras"]
    )

with col2:
    st.markdown(
        "<div class='upload-title'>🖼️ Upload Gambar</div>",
        unsafe_allow_html=True
    )

    image_file = st.file_uploader(
        "",
        type=["jpg", "jpeg", "png"]
    )

# =====================================
# NAMA KELAS
# =====================================
class_names = [
    "ikan koi",
    "kucing"
]

# =====================================
# LOAD MODEL
# =====================================
if model_file is not None:

    with tempfile.NamedTemporaryFile(
        delete=False,
        suffix=".keras"
    ) as tmp_model:

        tmp_model.write(model_file.read())
        model_path = tmp_model.name

    try:
        model = tf.keras.models.load_model(model_path)

        st.success("✅ Model berhasil dimuat")

        # =====================================
        # PROSES GAMBAR
        # =====================================
        if image_file is not None:

            img = Image.open(image_file)

            st.markdown("---")

            col_img, col_result = st.columns([1, 1])

            # ==========================
            # TAMPILKAN GAMBAR
            # ==========================
            with col_img:
                st.subheader("📷 Gambar Input")

                st.image(
                    img,
                    use_container_width=True
                )

            # ==========================
            # PREPROCESSING
            # ==========================
            img_resized = img.resize((227, 227))

            img_array = image.img_to_array(
                img_resized
            )

            img_array = np.expand_dims(
                img_array,
                axis=0
            )

            # ==========================
            # PREDIKSI
            # ==========================
            hasil = model.predict(img_array)

            prediksi_idx = np.argmax(
                hasil
            )

            prediksi_label = class_names[
                prediksi_idx
            ]

            confidence = float(
                np.max(hasil) * 100
            )

            # ==========================
            # HASIL
            # ==========================
            with col_result:

                st.subheader(
                    "🎯 Hasil Klasifikasi"
                )

                st.success(
                    f"Prediksi: {prediksi_label}"
                )

                st.info(
                    f"Tingkat Keyakinan: {confidence:.2f}%"
                )

                st.subheader(
                    "📊 Probabilitas Tiap Kelas"
                )

                for i, kelas in enumerate(
                    class_names
                ):

                    nilai = float(
                        hasil[0][i] * 100
                    )

                    st.progress(
                        min(int(nilai), 100)
                    )

                    st.write(
                        f"{kelas}: {nilai:.2f}%"
                    )

    except Exception as e:

        st.error(
            f"❌ Gagal memuat model: {e}"
        )

# =====================================
# FOOTER
# =====================================
st.markdown("""
<div class="footer">
    Sistem Klasifikasi Citra Menggunakan CNN
</div>
""", unsafe_allow_html=True)
