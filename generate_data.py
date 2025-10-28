import pandas as pd
import numpy as np
import os

# Create the data directory if it doesn't exist
if not os.path.exists('data'):
    os.makedirs('data')

n_samples = 20000
print(f"Generating {n_samples} rows of *logical* data...")

# 1. Generate the features
data = {
    'Blood_Sugar_Level': np.random.randint(70, 200, n_samples),
    'Body_Mass_Index': np.round(np.random.uniform(18.0, 67.0, n_samples), 1),
    'Age': np.random.randint(21, 81, n_samples),
    'Avg_Blood_Sugar_3Mo': np.round(np.random.uniform(4.0, 9.0, n_samples), 1),
    'Total_Cholesterol': np.random.randint(150, 300, n_samples),
    'Has_High_Blood_Pressure': np.random.randint(0, 2, n_samples), # 0 = No, 1 = Yes
    'Has_Heart_Disease': np.random.randint(0, 2, n_samples), # 0 = No, 1 = Yes
    'Smoking_Status': np.random.choice([0, 1, 2], n_samples, p=[0.6, 0.2, 0.2]), 
}
df = pd.DataFrame(data)

# 2. Create a "risk score" based on the features
# This creates a logical pattern for the model to find.
risk_score = (
    (df['Blood_Sugar_Level'] > 125).astype(int) * 3 +
    (df['Avg_Blood_Sugar_3Mo'] > 6.5).astype(int) * 4 +
    (df['Body_Mass_Index'] > 30).astype(int) * 2 +
    (df['Age'] > 45).astype(int) * 1 +
    df['Has_High_Blood_Pressure'] * 1 +
    df['Has_Heart_Disease'] * 1 +
    (df['Total_Cholesterol'] > 240).astype(int) * 1
)

# 3. Add some randomness (noise) to make it more realistic
risk_score = risk_score + np.random.normal(0, 1, n_samples)

# 4. Assign outcome: People with the top 30% of risk scores have diabetes
threshold = np.percentile(risk_score, 70) 
df['Has_Diabetes'] = (risk_score > threshold).astype(int)

# 5. Save the new, logical dataset
file_path = 'data/diabetes_data.csv'
df.to_csv(file_path, index=False)

print(f"Successfully generated {n_samples} rows of logical data at {file_path}")
print(f"Total diabetic samples: {df['Has_Diabetes'].sum()}")