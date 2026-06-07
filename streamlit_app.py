import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
from tensorflow.keras.preprocessing import image
import tempfile

st.set_page_config(
    page_title="Klasifikasi Gambar",
    page_icon="🖼️",
    layout="centered"
)

st.title("Klasifikasi Gambar CNN")

# Sesuaikan dengan model Anda
class_names = ["koi", "kucing"]

model_file = st.file_uploader(
    "Upload Model (.keras)",
    type=["keras"]
)

image_file = st.file_uploader(
    "Upload Gambar",
    type=["jpg", "jpeg", "png"]
)

@st.cache_resource
def load_model(uploaded_model):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".keras") as tmp:
        tmp.write(uploaded_model.read())
        model_path = tmp.name

    return tf.keras.models.load_model(model_path)

if model_file is not None:

    model = load_model(model_file)

    if image_file is not None:

        img = Image.open(image_file).convert("RGB")

        st.image(
            img,
            caption="Gambar yang diupload",
            use_container_width=True
        )

        if st.button("Prediksi"):

            img_resize = img.resize((224, 224))
            img_array = image.img_to_array(img_resize)
            img_array = np.expand_dims(img_array, axis=0)
            img_array = img_array / 255.0

            hasil = model.predict(img_array, verbose=0)

            prediksi = np.argmax(hasil)
            probabilitas = np.max(hasil)

            st.success(
                f"Hasil Prediksi: {class_names[prediksi]}"
            )

            st.write(
                f"Tingkat Keyakinan: {probabilitas:.2%}"
            )

            st.subheader("Probabilitas Tiap Kelas")

            for i, nama in enumerate(class_names):
                st.write(
                    f"{nama}: {hasil[0][i]:.4f}"
                )
