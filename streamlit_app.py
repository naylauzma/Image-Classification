import streamlit as st
import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing import image
from PIL import Image
import tempfile

st.set_page_config(page_title="Klasifikasi Ayam dan Gajah")

st.title("Klasifikasi Gambar Hewan")
st.write("Upload model dan gambar untuk diprediksi")

# Upload model
model_file = st.file_uploader(
    "Upload Model (.keras)",
    type=["keras"]
)

# Upload gambar
image_file = st.file_uploader(
    "Upload Gambar",
    type=["jpg", "jpeg", "png"]
)

# Nama kelas
class_names = ["ayam", "gajah"]

if model_file is not None:

    with tempfile.NamedTemporaryFile(delete=False, suffix=".keras") as tmp:
        tmp.write(model_file.read())
        model_path = tmp.name

    try:
        model = tf.keras.models.load_model(model_path)

        st.success("Model berhasil dimuat")

        if image_file is not None:

            img = Image.open(image_file).convert("RGB")

            st.image(
                img,
                caption="Gambar yang diupload",
                use_container_width=True
            )

            img_resize = img.resize((227, 227))

            img_array = image.img_to_array(img_resize)
            img_array = np.expand_dims(img_array, axis=0)

            hasil = model.predict(img_array)

            prediksi = np.argmax(hasil)
            label = class_names[prediksi]
            confidence = np.max(hasil) * 100

            st.subheader("Hasil Prediksi")
            st.success(f"{label}")

            st.subheader("Tingkat Keyakinan")
            st.info(f"{confidence:.2f}%")

            st.subheader("Probabilitas Tiap Kelas")

            for i, kelas in enumerate(class_names):
                st.write(
                    f"{kelas}: {hasil[0][i] * 100:.2f}%"
                )

    except Exception as e:
        st.error(f"Terjadi kesalahan: {e}")
