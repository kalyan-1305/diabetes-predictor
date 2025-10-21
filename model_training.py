import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import pickle
import os

# Create the models directory if it doesn't exist
if not os.path.exists('models'):
    os.makedirs('models')

# Load the dataset
try:
    df = pd.read_csv('data/diabetes_data.csv')
except FileNotFoundError:
    print("Error: data/diabetes_data.csv not found.")
    print("Please run 'python generate_data.py' first.")
    exit()

print("Data loaded successfully.")

# Define features (X) and target (y)
X = df.drop('Outcome', axis=1)
y = df['Outcome']

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# --- Crucial Step: Scale the data ---
# We use StandardScaler to normalize the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train the model
model = LogisticRegression(max_iter=1000)
model.fit(X_train_scaled, y_train)

# Evaluate the model
y_pred = model.predict(X_test_scaled)
accuracy = accuracy_score(y_test, y_pred)
print(f"Model trained with accuracy: {accuracy * 100:.2f}%")

# --- Crucial Step: Save the model AND the scaler ---
# We need the scaler to transform live user input later
model_path = 'models/model.pkl'
scaler_path = 'models/scaler.pkl'

with open(model_path, 'wb') as f:
    pickle.dump(model, f)

with open(scaler_path, 'wb') as f:
    pickle.dump(scaler, f)

print(f"Model saved to {model_path}")
print(f"Scaler saved to {scaler_path}")