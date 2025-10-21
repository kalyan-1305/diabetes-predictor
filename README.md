# 🩺 Diabetes Prediction Web App

This is a machine learning project that predicts the likelihood of a patient having diabetes based on their medical data. The model is deployed as an interactive web application using Streamlit.

**🔗 Live App:** [**You can find the live demo here!**](https://diabetes-predictor-8gvkg4slz2wekxhxfknjvm.streamlit.app/)

---

## 📸 App Preview

![Diabetes Prediction App Screenshot](<img width="650" height="412" alt="image" src="https://github.com/user-attachments/assets/d91cb43c-89da-4c10-9442-23e0dce5f87c" />
)


## 🎯 Project Overview

This project uses a Logistic Regression model trained on a synthetic dataset of 2,000 patient records. The web app provides a user-friendly interface to input patient data, which is then processed by the trained model to generate a prediction (Diabetes or No Diabetes).

---

## 🛠️ Technologies Used

* **Python:** The core programming language.
* **Pandas:** For data manipulation and creating the dataset.
* **Scikit-learn (sklearn):** For training the machine learning model (Logistic Regression) and preprocessing (StandardScaler).
* **Streamlit:** For building and deploying the interactive web front-end.
* **Pickle:** For saving and loading the trained model and scaler.

---

## 📁 Project Structure
diabetes_predictor/

│

├── data/

│   └── diabetes_data.csv   (Synthetic dataset)

│

├── models/

│   ├── model.pkl           (The trained logistic regression model)

│   └── scaler.pkl          (The saved StandardScaler object)

│

├── .gitignore

├── app.py                  (The main Streamlit application file)

├── generate_data.py        (Script to create the synthetic data)

├── model_training.py       (Script to train and save the model)

├── requirements.txt        (List of Python libraries required)

└── README.md               (You are here!) 
