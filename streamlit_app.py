import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np
import tempfile

# Judul aplikasi
st.title("Klasifikasi Hewan: Kelinci atau Lumba-Lumba")

# Upload model
model_file = st.file_uploader(
    "Upload Model CNN (.keras)",
    type=["keras"]
)

# Upload gambar
image_file = st.file_uploader(
    "Upload Gambar Hewan",
    type=["jpg", "jpeg", "png"]
)

# Nama kelas sesuai urutan saat training model
class_names = [
    "Kelinci",
    "Lumba-Lumba"
]

if model_file is not None:

    try:
        # Simpan model sementara
        with tempfile.NamedTemporaryFile(delete=False, suffix=".keras") as tmp:
            tmp.write(model_file.read())
            model_path = tmp.name

        # Load model
        model = tf.keras.models.load_model(model_path)

        st.success("Model berhasil dimuat")

        if image_file is not None:

            # Buka gambar
            img = Image.open(image_file).convert("RGB")

            # Tampilkan gambar asli
            st.image(
                img,
                caption="Gambar yang Diunggah",
                use_container_width=True
            )

            # Ambil ukuran input model
            input_shape = model.input_shape

            # Biasanya (None, 224, 224, 3)
            img_height = input_shape[1]
            img_width = input_shape[2]

            # Resize sesuai ukuran model
            img_resize = img.resize((img_width, img_height))

            # Konversi ke array
            img_array = np.array(img_resize)

            # Normalisasi
            img_array = img_array.astype("float32") / 255.0

            # Tambah dimensi batch
            img_array = np.expand_dims(img_array, axis=0)

            # Prediksi
            prediction = model.predict(img_array)

            # Ambil kelas dengan probabilitas tertinggi
            predicted_index = np.argmax(prediction)

            predicted_class = class_names[predicted_index]

            confidence = float(np.max(prediction)) * 100

            # Tampilkan hasil
            st.subheader("Hasil Klasifikasi")

            st.success(
                f"Hewan Terdeteksi: {predicted_class}"
            )

            st.write(
                f"Tingkat Keyakinan: {confidence:.2f}%"
            )

            # Tampilkan seluruh probabilitas
            st.subheader("Probabilitas Setiap Kelas")

            for i, class_name in enumerate(class_names):
                prob = float(prediction[0][i]) * 100

                st.write(
                    f"{class_name}: {prob:.2f}%"
                )

                st.progress(
                    min(int(prob), 100)
                )

    except Exception as e:
        st.error(f"Terjadi kesalahan: {e}")

else:
    st.info("Silakan upload model .keras terlebih dahulu")
