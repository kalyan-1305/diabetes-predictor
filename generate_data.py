import pandas as pd
import numpy as np
import os

# Create the data directory if it doesn't exist
if not os.path.exists('data'):
    os.makedirs('data')

# Set the number of samples
n_samples = 2000

# Generate synthetic data
data = {
    'Pregnancies': np.random.randint(0, 17, n_samples),
    'Glucose': np.random.randint(70, 200, n_samples),
    'BloodPressure': np.random.randint(40, 122, n_samples),
    'SkinThickness': np.random.randint(7, 99, n_samples),
    'Insulin': np.random.randint(14, 846, n_samples),
    'BMI': np.round(np.random.uniform(18.0, 67.0, n_samples), 1),
    'DiabetesPedigreeFunction': np.round(np.random.uniform(0.078, 2.42, n_samples), 3),
    'Age': np.random.randint(21, 81, n_samples),
    'Outcome': np.random.randint(0, 2, n_samples)  # 0 = No Diabetes, 1 = Diabetes
}

# Create a DataFrame
df = pd.DataFrame(data)

# Save to CSV
file_path = 'data/diabetes_data.csv'
df.to_csv(file_path, index=False)

print(f"Successfully generated {n_samples} rows of data at {file_path}")