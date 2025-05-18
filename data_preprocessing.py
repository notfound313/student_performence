import joblib
import numpy as np
import pandas as pd

# Load all the data preprocessing objects
encoder_Application_mode = joblib.load("model/encoder_Application_mode.joblib")
encoder_Course = joblib.load("model/encoder_Course.joblib")
encoder_Debtor = joblib.load("model/encoder_Debtor.joblib")
encoder_Gender = joblib.load("model/encoder_Gender.joblib")
encoder_Scholarship_holder = joblib.load("model/encoder_Scholarship_holder.joblib")
encoder_target = joblib.load("model/encoder_target.joblib")
encoder_Tuition_fees_up_to_date = joblib.load("model/encoder_Tuition_fees_up_to_date.joblib")
pca_1 = joblib.load("model/pca_1.joblib")
pca_2 = joblib.load("model/pca_2.joblib")
scaler_Admission_grade = joblib.load("model/scaler_Admission_grade.joblib")
scaler_Age_at_enrollment = joblib.load("model/scaler_Age_at_enrollment.joblib")
scaler_Application_order = joblib.load("model/scaler_Application_order.joblib")
scaler_Curricular_units_1st_sem_approved = joblib.load("model/scaler_Curricular_units_1st_sem_approved.joblib")
scaler_Curricular_units_1st_sem_credited = joblib.load("model/scaler_Curricular_units_1st_sem_credited.joblib")
scaler_Curricular_units_1st_sem_grade = joblib.load("model/scaler_Curricular_units_1st_sem_grade.joblib")
scaler_Curricular_units_2nd_sem_approved = joblib.load("model/scaler_Curricular_units_2nd_sem_approved.joblib")
scaler_Curricular_units_2nd_sem_credited = joblib.load("model/scaler_Curricular_units_2nd_sem_credited.joblib")
scaler_Curricular_units_2nd_sem_grade = joblib.load("model/scaler_Curricular_units_2nd_sem_grade.joblib")
scaler_GDP = joblib.load("model/scaler_GDP.joblib")
scaler_Previous_qualification_grade =joblib.load("model/scaler_Previous_qualification_grade.joblib")
scaler_Unemployment_rate = joblib.load("model/scaler_Unemployment_rate.joblib")


pca_numerical_columns_1 = [
    'Application_order',
    'Previous_qualification_grade',
    'Curricular_units_1st_sem_credited',
    'Curricular_units_1st_sem_enrolled',
    'Curricular_units_1st_sem_approved',
    'Curricular_units_1st_sem_grade',
    'Curricular_units_2nd_sem_credited',
    'Curricular_units_2nd_sem_enrolled',
    'Curricular_units_2nd_sem_approved',
    'Curricular_units_2nd_sem_grade',
]

pca_numerical_columns_2 = [
    'Age_at_enrollment',
    'Unemployment_rate',
    'GDP',
    'Admission_grade',
]

def data_preprocessing(data):
    data = data.copy()
    df = pd.DataFrame()

    # Encode categorical variables
    df["Application_mode"] = encoder_Application_mode.transform(data["Application_mode"])
    df["Course"] = encoder_Course.transform(data["Course"])
    df["Debtor"] = encoder_Debtor.transform(data["Debtor"])
    df["Gender"] = encoder_Gender.transform(data["Gender"])
    df["Scholarship_holder"] = encoder_Scholarship_holder.transform(data["Scholarship_holder"])
    df["Tuition_fees_up_to_date"] = encoder_Tuition_fees_up_to_date.transform(data["Tuition_fees_up_to_date"])

    # Scale numerical variables
    df["Admission_grade"] = scaler_Admission_grade.transform(np.asarray(data["Admission_grade"]).reshape(-1, 1))[0]
    df["Age_at_enrollment"] = scaler_Age_at_enrollment.transform(np.asarray(data["Age_at_enrollment"]).reshape(-1, 1))[0]
    df["Application_order"] = scaler_Application_order.transform(np.asarray(data["Application_order"]).reshape(-1, 1))[0]
    df["Curricular_units_1st_sem_approved"] = scaler_Curricular_units_1st_sem_approved.transform(np.asarray(data["Curricular_units_1st_sem_approved"]).reshape(-1, 1))[0]
    df["Curricular_units_1st_sem_credited"] = scaler_Curricular_units_1st_sem_credited.transform(np.asarray(data["Curricular_units_1st_sem_credited"]).reshape(-1, 1))[0]
    df["Curricular_units_1st_sem_grade"] = scaler_Curricular_units_1st_sem_grade.transform(np.asarray(data["Curricular_units_1st_sem_grade"]).reshape(-1, 1))[0]
    df["Curricular_units_2nd_sem_approved"] = scaler_Curricular_units_2st_sem_approved.transform(np.asarray(data["Curricular_units_2nd_sem_approved"]).reshape(-1, 1))[0]
    df["Curricular_units_2nd_sem_credited"] = scaler_Curricular_units_2st_sem_credited.transform(np.asarray(data["Curricular_units_2nd_sem_credited"]).reshape(-1, 1))[0]
    df["Curricular_units_2nd_sem_grade"] = scaler_Curricular_units_2st_sem_grade.transform(np.asarray(data["Curricular_units_2nd_sem_grade"]).reshape(-1, 1))[0]
    df["GDP"] = scaler_GDP.transform(np.asarray(data["GDP"]).reshape(-1, 1))[0]
    df["Previous_qualification_grade"] = scaler_Previous_qualification_grade.transform(np.asarray(data["Previous_qualification_grade"]).reshape(-1, 1))[0]
    df["Unemployment_rate"] = scaler_Unemployment_rate.transform(np.asarray(data["Unemployment_rate"]).reshape(-1, 1))[0]

    # Apply PCA transformations
    pca_1_input = pd.DataFrame([{
        key: df[key] for key in pca_numerical_columns_1
    }])
    pca_2_input = pd.DataFrame([{
        key: df[key] for key in pca_numerical_columns_2
    }])
    
    pca1_result = pca_1.transform(pca_1_input)
    pca2_result = pca_2.transform(pca_2_input)

    df_pca = pd.DataFrame(pca1_result, columns=[f"pca1_{i+1}" for i in range(pca1_result.shape[1])])
    df_pca2 = pd.DataFrame(pca2_result, columns=[f"pca2_{i+1}" for i in range(pca2_result.shape[1])])

    df = pd.concat([df, df_pca, df_pca2], axis=1)

    return df
