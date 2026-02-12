import numpy as np
import joblib

def predict_disease(age, gender, symptoms):
    # Load models and encoders
    model = joblib.load("media/best_model.pkl")
    scaler = joblib.load("media/scaler.pkl")
    label_encoders = joblib.load("media/label_encoders.pkl")
    disease_encoder = joblib.load("media/disease_encoder.pkl")

    encoded = []
    encoded.append(age)  # Age stays numeric
    encoded.append(label_encoders['Gender'].transform([gender])[0])  # Gender

    for i, sym in enumerate(symptoms):
        key = f'Symptom{i + 1}'
        if sym in label_encoders[key].classes_:
            encoded.append(label_encoders[key].transform([sym])[0])
        else:
            encoded.append(0)  # unknown symptom

    # Predict
    input_data = np.array(encoded).reshape(1, -1)
    input_data = scaler.transform(input_data)
    prediction = model.predict(input_data)
    return disease_encoder.inverse_transform(prediction)[0]
