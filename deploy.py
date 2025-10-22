import streamlit as st
import tensorflow as tf
import numpy as np
import cv2
import os
import requests

# ========== 1️⃣ Tải model từ Google Drive ==========
@st.cache_resource
def load_model():
    model_path = "flatfoot_model_VietNam.keras"
    if not os.path.exists(model_path):
        st.info("⏳ Đang tải model từ Google Drive...")
        file_id = "1NuiJ7hbT3wOAgKUuvj2D49y_PSB1R5vk"                     # 👈 thay bằng ID thật
        url = f"https://drive.google.com/uc?id={file_id}"
        r = requests.get(url)
        open(model_path, "wb").write(r.content)
        st.success("✅ Đã tải xong model!")
    model = tf.keras.models.load_model(model_path)
    return model

model = load_model()

# ========== 2️⃣ Tiền xử lý ảnh ==========
def preprocess_image(uploaded_file):
    file_bytes = np.frombuffer(uploaded_file.read(), np.uint8)
    img = cv2.imdecode(file_bytes, cv2.IMREAD_GRAYSCALE)
    img = cv2.equalizeHist(img)
    kernel = np.array([[0, -1, 0],
                       [-1, 5, -1],
                       [0, -1, 0]])
    img = cv2.filter2D(img, -1, kernel)
    img = cv2.resize(img, (224, 224))
    img = img / 255.0
    return img.reshape(1, 224, 224, 1)

# ========== 3️⃣ Giao diện ==========
st.set_page_config(page_title="AI Chẩn đoán bàn chân bẹt", page_icon="🦶")

st.title("🦶 HỆ THỐNG AI CHẨN ĐOÁN HỘI CHỨNG BÀN CHÂN BẸT")
st.write("Tải lên ảnh X-quang bàn chân để hệ thống AI chẩn đoán:")

uploaded_file = st.file_uploader("📁 Chọn ảnh X-quang", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    st.image(uploaded_file, caption="Ảnh X-quang đã tải lên", use_column_width=True)
    img = preprocess_image(uploaded_file)
    preds = model.predict(img)
    pred_class = np.argmax(preds)
    confidence = np.max(preds)

    labels = ["Bình thường", "Bẹt nhẹ", "Bẹt vừa", "Bẹt nặng", "Không xác định"]
    st.subheader(f"📊 Kết quả: {labels[pred_class]}")
    st.write(f"Độ tin cậy: **{confidence:.2%}**")
    st.bar_chart(preds[0])

st.markdown("---")
st.caption("👨‍⚕️ Đại học Thủy Lợi – Dự án AI Chẩn đoán bàn chân bẹt (2025)")
