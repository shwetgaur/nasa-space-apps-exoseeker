# train_model.py
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import joblib # For saving the model

def train():
    """Trains and saves the exoplanet classification model."""

    # Load the cleaned data
    df = pd.read_csv('data/cleaned_koi_data.csv')

    # --- Feature and Target Preparation ---
    # 'is_exoplanet' is our target. All other columns are features.
    X = df.drop('is_exoplanet', axis=1)
    y = df['is_exoplanet']

    # Split data into training and testing sets (80% train, 20% test)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y # Stratify helps with imbalanced classes
    )

    print("Data split complete.")
    print("Training data shape:", X_train.shape)

    # --- Model Training ---
    # We're using a RandomForestClassifier. It's robust and great for this kind of tabular data.
    model = RandomForestClassifier(n_estimators=120, random_state=42, n_jobs=-1)

    print("Training the model...")
    model.fit(X_train, y_train)
    print("Model training complete.")

    # --- Model Evaluation ---
    predictions = model.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)

    print("\n--- Model Performance ---")
    print(f"Accuracy: {accuracy:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, predictions))

    # --- Save the Model ---
    # The trained model is saved so our app can use it without retraining.
    model_filename = 'exoplanet_model.joblib'
    joblib.dump(model, model_filename)
    print(f"\nModel saved as {model_filename}")

if __name__ == '__main__':
    train()