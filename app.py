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
    page_title="Sistem Klasifikasi AYAM & GAJAH",
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

/* Sembunyikan menu bawaan */
#MainMenu{
    visibility:hidden;
}

footer{
    visibility:hidden;
}

header{
    visibility:hidden;
}

/* Navbar */
.navbar{
    background-color:#000000;
    padding:20px;
    border-radius:15px;
    text-align:center;
    margin-bottom:25px;
}

.navbar h1{
    color:white;
    margin:0;
    font-size:42px;
    font-weight:bold;
}

/* Deskripsi */
.description{
    text-align:center;
    color:white;
    font-size:15px;
    margin-bottom:25px;
}

/* Card Upload */
.upload-card{
    background:white;
    padding:20px;
    border-radius:15px;
    border:3px solid #000000;
    box-shadow:0px 4px 15px rgba(0,0,0,0.15);
    margin-bottom:15px;
}

/* Judul Upload */
.upload-title{
    text-align:center;
    font-size:24px;
    font-weight:bold;
    color:black;
}

/* Footer */
.footer{
    text-align:center;
    color:white;
    margin-top:30px;
    font-size:14px;
}

</style>
""", unsafe_allow_html=True)

# =====================================
# NAVBAR
# =====================================
st.markdown("""
<div class="navbar">
    <h1>🧠 Sistem Klasifikasi Hewan</h1>
</div>
""", unsafe_allow_html=True)

# =====================================
# DESKRIPSI
# =====================================
st.markdown("""
<div class="description">
    Upload model CNN (.keras) dan gambar untuk melakukan klasifikasi
</div>
""", unsafe_allow_html=True)

# =====================================
# UPLOAD SECTION
# =====================================
col1, col2 = st.columns(2)

with col1:

    st.markdown("""
    <div class="upload-card">
        <div class="upload-title">
            📦 Upload Model
        </div>
    </div>
    """, unsafe_allow_html=True)

    model_file = st.file_uploader(
        "",
        type=["keras"],
        key="model"
    )

with col2:

    st.markdown("""
    <div class="upload-card">
        <div class="upload-title">
            🖼️ Upload Gambar
        </div>
    </div>
    """, unsafe_allow_html=True)

    image_file = st.file_uploader(
        "",
        type=["jpg", "jpeg", "png"],
        key="gambar"
    )

# =====================================
# NAMA KELAS
# =====================================
class_names = [
    "Ayam",
    "Gajah"
]

# =====================================
# LOAD MODEL DAN PREDIKSI
# =====================================
if model_file is not None:

    with tempfile.NamedTemporaryFile(
        delete=False,
        suffix=".keras"
    ) as tmp_model:

        tmp_model.write(model_file.read())
        model_path = tmp_model.name

    try:

        model = tf.keras.models.load_model(
            model_path
        )

        st.success(
            "✅ Model berhasil dimuat"
        )

        if image_file is not None:

            img = Image.open(
                image_file
            )

            st.markdown("---")

            col_img, col_result = st.columns(
                [1, 1]
            )

            with col_img:

                st.subheader(
                    "📷 Gambar Input"
                )

                st.image(
                    img,
                    use_container_width=True
                )

            # Preprocessing
            img_resized = img.resize(
                (227, 227)
            )

            img_array = image.img_to_array(
                img_resized
            )

            img_array = np.expand_dims(
                img_array,
                axis=0
            )

            # Prediksi
            hasil = model.predict(
                img_array
            )

            prediksi_idx = np.argmax(
                hasil
            )

            prediksi_label = class_names[
                prediksi_idx
            ]

            confidence = float(
                np.max(hasil) * 100
            )

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

                    st.write(
                        f"{kelas}: {nilai:.2f}%"
                    )

                    st.progress(
                        min(
                            int(nilai),
                            100
                        )
                    )

    except Exception as e:

        st.error(
            f"Gagal memuat model: {e}"
        )

# =====================================
# FOOTER
# =====================================
st.markdown("""
<div class="footer">
    Sistem Klasifikasi Citra Menggunakan CNN
</div>
""", unsafe_allow_html=True)
