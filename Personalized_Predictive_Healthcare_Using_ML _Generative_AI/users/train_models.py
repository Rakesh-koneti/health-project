import pandas as pd
import numpy as np
import joblib
import os

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score

from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from xgboost import XGBClassifier
from django.conf import settings


def train_models():
    # Use plain ASCII in logs to avoid Windows console encoding issues
    print("Starting training pipeline...")

    # === Step 1: Load Dataset ===
    file_path = os.path.join(settings.MEDIA_ROOT, 'final_dataset_30000.csv')
    print(f"Loading dataset from: {file_path}")

    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Dataset not found: {file_path}")

    df = pd.read_csv(file_path, encoding='ISO-8859-1')
    print(f"Dataset loaded with shape: {df.shape}")

    df.dropna(inplace=True)
    print(f"Dropped nulls. New shape: {df.shape}")

    # === Step 2: Ensure Media Folder ===
    os.makedirs("media", exist_ok=True)
    print("'media' folder ready.")

    # === Step 3: Encode Categorical Columns ===
    label_encoders = {}
    for col in ['Gender'] + [f'Symptom{i}' for i in range(1, 8)]:
        print(f"Encoding column: {col}")
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
        label_encoders[col] = le

    disease_encoder = LabelEncoder()
    df['Disease'] = disease_encoder.fit_transform(df['Disease'])
    print("Target column 'Disease' encoded.")

    # === Step 4: Save Encoders ===
    joblib.dump(label_encoders, "media/label_encoders.pkl")
    joblib.dump(disease_encoder, "media/disease_encoder.pkl")
    print("Saved encoders to media/.")

    # === Step 5: Prepare Features and Target ===
    X = df.drop(columns=["Disease"])
    y = df["Disease"]
    print(f"Features shape: {X.shape}, Labels shape: {y.shape}")

    # === Step 6: Scale Features ===
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    joblib.dump(scaler, "media/scaler.pkl")
    print("Feature scaling completed and saved.")

    # === Step 7: Split Dataset ===
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, stratify=y, random_state=42
    )
    print(f"Data split: Train={X_train.shape[0]}, Test={X_test.shape[0]}")

    # === Step 8: Define Models ===
    base_models = {
        "DecisionTree": DecisionTreeClassifier(),
        "RandomForest": RandomForestClassifier(),
        "NaiveBayes": GaussianNB(),
        "LogisticRegression": LogisticRegression(max_iter=1000),
        "SVM": SVC(probability=True),
        "KNN": KNeighborsClassifier(),
        "XGBoost": XGBClassifier(use_label_encoder=False, eval_metric='mlogloss')
    }

    model_instances = []
    accuracies = {}
    best_model = None
    best_model_name = ""
    best_accuracy = 0.0

    # === Step 9: Train Base Models ===
    print("Training base models...")
    for name, model in base_models.items():
        print(f"Training {name}...")
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        acc = round(accuracy_score(y_test, y_pred), 4)
        accuracies[name] = acc
        print(f"{name} accuracy: {acc}")

        joblib.dump(model, f"media/{name}_model.pkl")
        print(f"Saved {name} model.")

        model_instances.append((name, model))

        if acc > best_accuracy:
            best_model = model
            best_model_name = name
            best_accuracy = acc

    # === Step 10: Train Soft Voting Ensemble ===
    print("Training VotingEnsemble (soft voting)...")
    voting_ensemble = VotingClassifier(estimators=model_instances, voting='soft')
    voting_ensemble.fit(X_train, y_train)
    y_vote = voting_ensemble.predict(X_test)
    vote_acc = round(accuracy_score(y_test, y_vote), 4)
    accuracies["VotingEnsemble"] = vote_acc
    print(f"VotingEnsemble accuracy: {vote_acc}")

    joblib.dump(voting_ensemble, "media/VotingEnsemble_model.pkl")
    print("Saved VotingEnsemble model.")

    # === Step 11: Save Best Model ===
    if vote_acc > best_accuracy:
        best_model = voting_ensemble
        best_model_name = "VotingEnsemble"
        best_accuracy = vote_acc
        print("VotingEnsemble selected as the best model.")

    joblib.dump(best_model, "media/best_model.pkl")
    print(f"Best model ({best_model_name}) saved as best_model.pkl")

    print("Training pipeline complete.\n")
    return accuracies, best_model_name
