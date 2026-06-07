import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
from tensorflow.keras.preprocessing import image

st.set_page_config(
    page_title="Klasifikasi Gambar CNN",
    page_icon="🖼️",
    layout="centered"
)

st.title("Klasifikasi Gambar CNN")

# Upload model
model_file = st.file_uploader(
    "Upload Model (.keras)",
    type=["keras"]
)

# Input nama kelas
st.subheader("Nama Kelas")

kelas1 = st.text_input(
    "Nama Kelas 1",
    placeholder="Contoh: Koi"
)

kelas2 = st.text_input(
    "Nama Kelas 2",
    placeholder="Contoh: Kucing"
)

# Upload gambar
image_file = st.file_uploader(
    "Upload Gambar",
    type=["jpg", "jpeg", "png"]
)

if model_file is not None:

    @st.cache_resource
    def load_model(file):
        return tf.keras.models.load_model(file)

    model = load_model(model_file)

    if image_file is not None:

        img = Image.open(image_file).convert("RGB")

        st.image(
            img,
            caption="Gambar yang Diupload",
            use_container_width=True
        )

        if st.button("Prediksi"):

            if kelas1 == "" or kelas2 == "":
                st.warning("Masukkan nama kedua kelas terlebih dahulu")

            else:

                class_names = [kelas1, kelas2]

                # Preprocessing gambar
                img_resize = img.resize((224, 224))
                img_array = image.img_to_array(img_resize)
                img_array = np.expand_dims(img_array, axis=0)
                img_array = img_array / 255.0

                # Prediksi
                hasil = model.predict(img_array, verbose=0)

                prediksi = np.argmax(hasil)
                probabilitas = np.max(hasil)

                st.success(
                    f"Hasil Prediksi: {class_names[prediksi]}"
                )

                st.write(
                    f"Tingkat Keyakinan: {probabilitas:.2%}"
                )

                st.subheader("Probabilitas Setiap Kelas")

                for i, nama in enumerate(class_names):
                    st.write(
                        f"{nama}: {hasil[0][i]:.4f}"
                    )
