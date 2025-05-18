import streamlit as st
import pandas as pd
from data_preprocessing import (
    data_preprocessing,
    encoder_Application_mode,
    encoder_Course,
    encoder_Debtor,
    encoder_Scholarship_holder,
    encoder_Gender,
    encoder_Tuition_fees_up_to_date
)
from prediction import prediction


st.set_page_config(page_title="Student Performance Predictor", layout="wide")

st.markdown("<h1 style='text-align: center;'>🎓 Student Performance Predictor 📊</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center; font-size: 18px;'>Masukkan data mahasiswa untuk memprediksi performa akademiknya berdasarkan histori akademik dan data personal.</p>", unsafe_allow_html=True)
st.markdown("---")


# Input Form
data = pd.DataFrame()

st.subheader("📝 Data Input Mahasiswa")

with st.expander("📌 Masukkan Informasi Akademik & Personal", expanded=True):

    # Categorical Inputs
    st.markdown("### 📋 Data Kategorikal")
    col1, col2, col3 = st.columns(3)

    with col1:
        Application_mode = st.selectbox("Application Mode", options=encoder_Application_mode.classes_)
        data["Application_mode"] = [Application_mode]

    with col2:
        Course = st.selectbox("Course", options=encoder_Course.classes_)
        data["Course"] = [Course]

    with col3:
        Debtor = st.selectbox("Debtor", options=encoder_Debtor.classes_)
        data["Debtor"] = [Debtor]

    col1, col2, col3 = st.columns(3)
    with col1:
        Scholarship_holder = st.selectbox("Scholarship Holder", options=encoder_Scholarship_holder.classes_)
        data["Scholarship_holder"] = [Scholarship_holder]

    with col2:
        Gender = st.selectbox("Gender", options=encoder_Gender.classes_)
        data["Gender"] = [Gender]

    with col3:
        Tuition_fees_up_to_date = st.selectbox("Tuition Fees Up To Date", options=encoder_Tuition_fees_up_to_date.classes_)
        data["Tuition_fees_up_to_date"] = [Tuition_fees_up_to_date]

    st.markdown("### 📈 Data Numerik")

    # Numerical Inputs
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        data["Application_order"] = [st.number_input("Application Order", min_value=1, value=1)]
    with col2:
        data["Previous_qualification_grade"] = [st.number_input("Previous Qualification Grade", value=150.0)]
    with col3:
        data["Curricular_units_1st_sem_credited"] = [st.number_input("1st Sem Credited Units", value=6.0)]
    with col4:
        data["Curricular_units_1st_sem_enrolled"] = [st.number_input("1st Sem Enrolled Units", value=6.0)]

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        data["Curricular_units_1st_sem_approved"] = [st.number_input("1st Sem Approved Units", value=6.0)]
    with col2:
        data["Curricular_units_1st_sem_grade"] = [st.number_input("1st Sem Grade", value=14.0)]
    with col3:
        data["Curricular_units_2nd_sem_credited"] = [st.number_input("2nd Sem Credited Units", value=6.0)]
    with col4:
        data["Curricular_units_2nd_sem_enrolled"] = [st.number_input("2nd Sem Enrolled Units", value=6.0)]

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        data["Curricular_units_2nd_sem_approved"] = [st.number_input("2nd Sem Approved Units", value=6.0)]
    with col2:
        data["Curricular_units_2nd_sem_grade"] = [st.number_input("2nd Sem Grade", value=14.0)]
    with col3:
        data["Age_at_enrollment"] = [st.number_input("Age at Enrollment", value=18.0)]
    with col4:
        data["Admission_grade"] = [st.number_input("Admission Grade", value=160.0)]

    col1, col2 = st.columns(2)
    with col1:
        data["Unemployment_rate"] = [st.number_input("Unemployment Rate (%)", value=6.5)]
    with col2:
        data["GDP"] = [st.number_input("GDP (%)", value=2.1)]


with st.expander("📄 Lihat Data Mentah"):
    st.dataframe(data, use_container_width=True)


st.markdown("---")
st.subheader("🔍 Hasil Prediksi")

if st.button("🎯 Prediksi Sekarang!"):
    new_data = data_preprocessing(data)
    with st.expander("🧪 Lihat Data Setelah Preprocessing"):
        st.dataframe(new_data, use_container_width=True)

    hasil_prediksi = prediction(new_data)
    st.success(f"Hasil Prediksi: **{hasil_prediksi}** 🏆")
