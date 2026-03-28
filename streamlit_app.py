import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import svm
import matplotlib.pyplot as plt

# -----------------------------
# Chuẩn bị dữ liệu
# -----------------------------
data = pd.read_csv("https://raw.githubusercontent.com/quanqanh720-dev/diabetes-prediction/refs/heads/master/Data/Diabetes-dataset_OK.csv")
X = data.drop("Outcome", axis=1)
y = data["Outcome"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# -----------------------------
# Huấn luyện mô hình SVM
# -----------------------------
svm_model = svm.SVC(kernel='linear', gamma=1, C=0.1, probability=True)
svm_model.fit(X_train, y_train)

# -----------------------------
# Giao diện Streamlit
# -----------------------------
st.set_page_config(page_title="Ứng dụng dự đoán tiểu đường", layout="wide")
st.markdown("<h1 style='text-align: center; color: #2E86C1;'>Dự đoán nguy cơ tiểu đường</h1>", unsafe_allow_html=True)

col1, col2 = st.columns([2, 1])

with col1:
    st.subheader("📋 Thông tin bệnh nhân")
    glucose = st.number_input("Mức glucose", min_value=0, max_value=200, value=100)
    blood_pressure = st.number_input("Huyết áp", min_value=0, max_value=150, value=70)
    skin_thickness = st.number_input("Độ dày da", min_value=0, max_value=100, value=20)
    insulin = st.number_input("Insulin", min_value=0, max_value=900, value=80)
    bmi = st.number_input("BMI", min_value=0.0, max_value=70.0, value=25.0)
    dpf = st.number_input("Diabetes Pedigree Function", min_value=0.0, max_value=3.0, value=0.5)
    age = st.number_input("Tuổi", min_value=1, max_value=120, value=30)

    if st.button("🚀 Dự đoán"):
        input_data = np.array([[glucose, blood_pressure, skin_thickness,
                                insulin, bmi, dpf, age]])
        prediction = svm_model.predict(input_data)
        probability = svm_model.predict_proba(input_data)[0][1] * 100

        with col2:
            st.subheader("📊 Kết quả")
            if prediction[0] == 1:
                st.error(f"⚠️ Bệnh nhân có **nguy cơ TIỂU ĐƯỜNG**\n\nXác suất: {probability:.2f}%")
            else:
                st.success(f"✅ Bệnh nhân **KHÔNG có nguy cơ tiểu đường**\n\nXác suất: {100-probability:.2f}%")

            # -----------------------------
            # Biểu đồ chỉ số
            # -----------------------------
            st.subheader("📈 Biểu đồ các chỉ số")
            features = ["Glucose", "Huyết áp", "Độ dày da", "Insulin", "BMI", "DPF", "Tuổi"]
            values = [glucose, blood_pressure, skin_thickness, insulin, bmi, dpf, age]

            fig, ax = plt.subplots()
            ax.bar(features, values, color="#2E86C1")
            ax.set_ylabel("Giá trị")
            ax.set_title("Các chỉ số bệnh nhân")
            plt.xticks(rotation=45)

            st.pyplot(fig)
