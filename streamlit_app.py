import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
from tensorflow.keras.preprocessing import image

st.set_page_config(
    page_title="Klasifikasi Gambar",
    page_icon="🐱",
    layout="centered"
)

@st.cache_resource
def load_model(model_file):
    return tf.keras.models.load_model(model_file)

st.title("Klasifikasi Gambar")

jenis = st.selectbox(
    "Pilih Jenis Klasifikasi",
    ["Ayam", "Gajah"]
)

model_file = st.file_uploader(
    "Upload Model (.keras)",
    type=["keras"]
)

image_file = st.file_uploader(
    "Upload Gambar",
    type=["jpg", "jpeg", "png"]
)

if model_file is not None and image_file is not None:

    model = load_model(model_file)

    if jenis == "Koi":
        class_names = ["koi", "bukan_koi"]
    else:
        class_names = ["kucing", "bukan_kucing"]

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

        st.subheader("Probabilitas Kelas")

        for i, nama in enumerate(class_names):
            st.write(
                f"{nama}: {hasil[0][i]:.4f}"
            )
