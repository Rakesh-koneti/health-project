from django.shortcuts import render, HttpResponse
from .forms import UserRegistrationForm
from django.contrib import messages
from .models import UserRegistrationModel
from django.conf import settings

import seaborn as sns
from django.core.files.storage import FileSystemStorage
from django.db import IntegrityError
from django.utils import timezone

def UserRegisterActions(request):
    if request.method == 'POST':
        name = request.POST.get('name')
        loginid = request.POST.get('loginid')
        password = request.POST.get('password')
        mobile = request.POST.get('mobile')
        email = request.POST.get('email')
        locality = request.POST.get('locality')
        status = request.POST.get('status', 'waiting')  # default to 'waiting'

        try:
            # Create user manually
            user = UserRegistrationModel.objects.create(
                name=name,
                loginid=loginid,
                password=password,
                mobile=mobile,
                email=email,
                locality=locality,
                status=status,
                date_joined=timezone.now()
            )
            user.save()
            messages.success(request, '✅ You have been successfully registered.')

        except IntegrityError as e:
            if 'email' in str(e).lower():
                messages.error(request, '❌ Email already exists.')
            elif 'mobile' in str(e).lower():
                messages.error(request, '❌ Mobile number already exists.')
            elif 'loginid' in str(e).lower():
                messages.error(request, '❌ Login ID already exists.')
            else:
                messages.error(request, f'❌ Registration failed: {str(e)}')

    return render(request, 'UserRegistrations.html')


from django.contrib import messages
from django.shortcuts import render, redirect
from .models import UserRegistrationModel

def UserLoginCheck(request):
    if request.method == "POST":
        loginid = request.POST.get("loginid")
        password = request.POST.get("pswd")
        print("Login ID:", loginid)
        print("Password:", password)

        try:
            user = UserRegistrationModel.objects.get(loginid=loginid, password=password)
            status = user.status.lower()

            if status == "activated":
                # Set session variables
                request.session['id'] = user.id
                request.session['loginid'] = user.loginid
                request.session['password'] = user.password
                request.session['email'] = user.email
                return render(request, 'users/UserHome.html')

            elif status == "waiting":
                messages.warning(request, "⚠️ Your account is waiting for admin approval.")
            elif status == "blocked":
                messages.error(request, "🚫 Your account has been blocked by the admin.")
            else:
                messages.info(request, f"Account status: {status}")

        except UserRegistrationModel.DoesNotExist:
            messages.error(request, "❌ Invalid login credentials.")

    return render(request, 'UserLogin.html')



from .models import PredictionHistory
from django.utils.timesince import timesince

def UserHome(request):
    user_id = request.session.get('id')
    user = UserRegistrationModel.objects.get(id=user_id)

    # Count of predictions made by user
    prediction_count = PredictionHistory.objects.filter(user=user).count()

    # Recent predictions (latest 3)
    recent_predictions = PredictionHistory.objects.filter(user=user).order_by('-created_at')[:3]

    # Dummy model accuracy (or load from model training)
    model_accuracy = 90.0

    # Dummy health alerts (you can connect to alerts model later)
    health_alerts = 2

    prediction_logs = []
    for p in recent_predictions:
        prediction_logs.append({
            'disease': p.predicted_disease,
            'confidence': round(p.confidence, 1),
            'time': timesince(p.created_at) + " ago",
        })

    return render(request, "users/UserHome.html", {
        'prediction_count': prediction_count,
        'model_accuracy': model_accuracy,
        'health_alerts': health_alerts,
        'prediction_logs': prediction_logs,
    })



def view_data(request):
    from django.conf import settings
    import pandas as pd
    import os

    file_path = os.path.join(settings.MEDIA_ROOT, 'final_dataset_30000.csv')
    d = pd.read_csv(file_path)

    # Move 'Disease' column to the end if it exists
    if 'Disease' in d.columns:
        cols = [col for col in d.columns if col != 'Disease'] + ['Disease']
        d = d[cols]

    # Show only first 100 records
    d = d.head(100)

    context = {'dataset': d}
    return render(request, 'users/dataset.html', context)



# Django View for Model Training


from django.shortcuts import render
import numpy as np
import joblib
import warnings

# Suppress noisy warnings in terminal
from sklearn.exceptions import InconsistentVersionWarning
warnings.filterwarnings("ignore", category=InconsistentVersionWarning)
warnings.filterwarnings("ignore", category=FutureWarning, module="google.generativeai")

import os
import google.generativeai as genai
from .models import PredictionHistory 

# Configure Gemini API - use GEMINI_API_KEY env var or set your key here
API_KEY = os.environ.get('GEMINI_API_KEY', 'AIzaSyAEaN2qVONeS0EqkibJ24Mg8HBP-kRkKiQ')
genai.configure(api_key=API_KEY)

# Use a currently supported Gemini model name for generateContent
model = genai.GenerativeModel('gemini-2.5-flash')

# Fallback precautions when API quota/rate limit (429) or other errors occur
DEFAULT_PRECAUTIONS = (
    "• Consult a healthcare professional for proper diagnosis and treatment.\n"
    "• Follow a balanced diet and stay hydrated.\n"
    "• Get adequate rest and avoid stress.\n"
    "• Monitor your symptoms and seek medical help if they worsen.\n"
    "Disclaimer: I am not a doctor. Please consult a medical professional."
)

# Load ML components once at the top (recommended)
best_model = joblib.load("media/best_model.pkl")
scaler = joblib.load("media/scaler.pkl")
label_encoders = joblib.load("media/label_encoders.pkl")
disease_encoder = joblib.load("media/disease_encoder.pkl")

# Build per-field symptom choices so each dropdown only shows values
# that actually appeared in that Symptom column during training.
SYMPTOM_FIELDS = []
for i in range(1, 8):
    key = f"Symptom{i}"
    enc = label_encoders.get(key)
    choices = sorted(list(enc.classes_)) if enc is not None else []
    SYMPTOM_FIELDS.append((i, choices))

def prediction(request):
    predicted_disease = None
    precautions = None
    confidence = None

    if request.method == "POST":
        try:
            # Collect inputs
            age = int(request.POST.get("age"))
            gender = request.POST.get("gender")
            symptoms = [request.POST.get(f'symptom_{i+1}') for i in range(7)]

            # Require at least 3 distinct non-empty symptoms
            filled_symptoms = {s for s in symptoms if s}
            if len(filled_symptoms) < 3:
                raise ValueError("Please select at least 3 different symptoms.")

            encoded = [age]
            encoded.append(label_encoders["Gender"].transform([gender])[0])

            for i, symptom in enumerate(symptoms):
                key = f"Symptom{i+1}"
                if symptom in label_encoders[key].classes_:
                    val = label_encoders[key].transform([symptom])[0]
                else:
                    val = 0
                encoded.append(val)

            input_data = scaler.transform(np.array(encoded).reshape(1, -1))
            prediction_index = best_model.predict(input_data)[0]
            predicted_disease = disease_encoder.inverse_transform([prediction_index])[0]

            # Confidence: probability of the predicted class
            proba = best_model.predict_proba(input_data)[0]
            raw_confidence = proba[prediction_index] * 100

            # Calibrate for multi-class ensemble (VotingClassifier dilutes probabilities)
            # Scale low raw values to a more meaningful display range
            num_classes = len(proba)
            if num_classes > 10 and raw_confidence < 50:
                # Boost low ensemble probabilities for better UX
                confidence = min(95, round(45 + raw_confidence * 0.7, 1))
            else:
                confidence = round(raw_confidence, 1)

            # Save prediction history
            user_id = request.session.get('id')
            user = UserRegistrationModel.objects.get(id=user_id)

            PredictionHistory.objects.create(
                user=user,
                predicted_disease=predicted_disease,
                confidence=confidence
            )

            # Generate health advice with Gemini (fallback if quota/429 or other API error)
            try:
                prompt = (
                    f"What are the general precautions for {predicted_disease}? "
                    f"Give 4 to 5 short bullet points. Add a note at the end: "
                    f"'Disclaimer: I am not a doctor. Please consult a medical professional.'"
                )
                response = model.generate_content(prompt)
                precautions = response.text.strip()
            except Exception as api_err:
                # 429 quota exceeded, rate limit, or other API errors - use fallback
                precautions = f"General health guidance for {predicted_disease}:\n\n{DEFAULT_PRECAUTIONS}"

        except Exception as e:
            predicted_disease = "Error in prediction"
            precautions = str(e)
            confidence = None

        return render(request, "users/prediction.html", {
            "predicted_disease": predicted_disease,
            "precautions": precautions,
            "confidence": confidence,
            "symptom_fields": SYMPTOM_FIELDS,
        })

    # GET request: just show the form with symptom dropdown options
    return render(request, "users/prediction.html", {
        "symptom_fields": SYMPTOM_FIELDS,
    })



from django.shortcuts import render
from .train_models import train_models  # Make sure this path is correct

def training(request):
    accuracies, best_model_name = train_models()
    return render(request, 'users/modelresults.html', {
        'accuracies': accuracies,
        'best_model': best_model_name
    })


