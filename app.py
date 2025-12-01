import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

# 1. Cấu hình trang
st.set_page_config(
    page_title="Phân Loại Rác Thải AI",
    page_icon="♻️",
    layout="centered"
)

# 2. Load Model (Dùng Cache để không phải load lại mỗi khi bấm nút)
@st.cache_resource
def load_model():
    # Đường dẫn đến file model của bạn
    model = tf.keras.models.load_model('best_model_scratch.keras')
    return model

try:
    model = load_model()
    st.success("✅ Đã tải mô hình thành công!")
except Exception as e:
    st.error(f"Không thể tải mô hình: {e}")

# 3. Định nghĩa nhãn (theo đúng thứ tự train)
# Kiểm tra lại thứ tự trong code train của bạn (thường là alpha-beta)
CLASS_NAMES = ['Cardboard (Bìa)', 'Glass (Thủy tinh)', 'Metal (Kim loại)', 
               'Paper (Giấy)', 'Plastic (Nhựa)', 'Trash (Rác khác)']

# 4. Hàm xử lý ảnh
def preprocess_image(image):
    # Resize về 224x224 giống lúc train
    image = image.resize((224, 224))
    # Chuyển thành mảng numpy
    img_array = np.array(image)
    # Chuẩn hóa (chia 255)
    img_array = img_array / 255.0
    # Thêm chiều batch (1, 224, 224, 3)
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

# 5. Giao diện người dùng
st.title("♻️ Hệ Thống Phân Loại Rác Thải")
st.write("Tải lên hình ảnh rác thải để AI nhận diện.")

# Widget tải ảnh
uploaded_file = st.file_uploader("Chọn ảnh...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Hiển thị ảnh
    image = Image.open(uploaded_file)
    st.image(image, caption='Ảnh đã tải lên', use_column_width=True)
    
    # Nút dự đoán
    if st.button('🔍 Phân loại ngay'):
        with st.spinner('Đang phân tích...'):
            # Xử lý và dự đoán
            processed_img = preprocess_image(image)
            predictions = model.predict(processed_img)
            
            # Lấy kết quả cao nhất
            score = tf.nn.softmax(predictions[0])
            class_index = np.argmax(predictions[0])
            confidence = np.max(predictions[0]) * 100
            
            # Hiển thị kết quả
            st.markdown(f"### Kết quả: **{CLASS_NAMES[class_index]}**")
            st.progress(int(confidence))
            st.info(f"Độ tin cậy: {confidence:.2f}%")
            
            # Hiển thị chi tiết xác suất các lớp khác
            with st.expander("Xem chi tiết xác suất"):
                for i, name in enumerate(CLASS_NAMES):
                    st.write(f"{name}: {predictions[0][i]*100:.2f}%")

# Footer
st.markdown("---")
st.caption("Đồ án môn học: Học máy và Ứng dụng - Nhóm 20")