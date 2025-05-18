import joblib

model = joblib.load("model/gboost_model.joblib")
result_target = joblib.load("model/encoder_target.joblib")

def prediction(data):    
    result = model.predict(data)
    final_result = result_target.inverse_transform(result)[0]
    return final_result