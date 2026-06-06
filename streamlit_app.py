import streamlit as st
import tensorflow as tf
from PIL import Image
import tempfile

st.title("Klasifikasi CNN")

model_file = st.file_uploader(
    "Upload Model CNN (.keras)",
    type=["keras"]
)

image_file = st.file_uploader(
    "Upload Gambar",
    type=["jpg", "jpeg", "png"]
)

if model_file is not None:

    with tempfile.NamedTemporaryFile(delete=False, suffix=".keras") as tmp:
        tmp.write(model_file.read())
        model_path = tmp.name

    model = tf.keras.models.load_model(model_path)

    st.success("Model berhasil dimuat")

    if image_file is not None:
        img = Image.open(image_file)

        st.image(img, use_container_width=True)

        st.write("Siap untuk prediksi")
