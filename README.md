# 🩺 Diabetes Prediction Web App

This is a machine learning project that predicts the likelihood of a patient having diabetes based on their medical data. The model is deployed as an interactive web application using Streamlit.

**🔗 Live App:** [**You can find the live demo here!**](https://diabetes-predictor-8gvkg4slz2wekxhxfknjvm.streamlit.app/)

---

## 📸 App Preview

![Diabetes Prediction App Screenshot](<img width="632" height="414" alt="image" src="https://github.com/user-attachments/assets/df58c6a6-7f99-4136-9b38-2ef174997890" />

)


## 🎯 Project Overview

This project is an interactive web application designed to predict the likelihood of a patient having diabetes with high accuracy.

It uses a high-performance Random Forest Classifier model, which was trained on a custom-generated, logical dataset of 20,000 patient records. By analyzing 8 critical health features (including 3-Month Avg. Blood Sugar, Total Cholesterol, and Blood Pressure), the model achieves 91.12% accuracy in its predictions. The user-friendly interface is built with Streamlit to provide instant, real-time results.
---

## 🛠️ Technologies Used

* **Python:** The core programming language.
* **Pandas:** For data manipulation and creating the 20,000-row synthetic dataset.
* **Numpy:** For numerical operations and generating the logical data patterns.
* **Scikit-learn (sklearn):** For training the machine learning model (**RandomForestClassifier**) and preprocessing (**StandardScaler**).
* **Streamlit:** For building and deploying the interactive web front-end.
* **Pickle:** For saving and loading the trained model and scaler.

---
(**Note:** To add a preview, take a screenshot of your app, upload it to a site like [imgur.com](https://imgur.com/), and paste the direct image link here.)*

---

## ✨ Key Features

* **High Accuracy:** Achieves **91.12%** accuracy on the test dataset.
* **Robust Model:** Uses a **Random Forest Classifier** for reliable and precise predictions.
* **Interactive UI:** A simple and clear web interface built with Streamlit for easy user input.
* **Rich Dataset:** Trained on a custom-generated, logical dataset of 20,000 patient records.
* **Key Health Indicators:** Predicts based on 8 critical features, including:
    * Blood Sugar Level
    * 3-Month Avg. Blood Sugar (HbA1c)
    * Body Mass Index (BMI)
    * Total Cholesterol
    * High Blood Pressure
    * Heart Disease
    * Age
    * Smoking Status

---

## 📁 Project Structure
diabetes_predictor/

│

├── data/

│   └── diabetes_data.csv   (Synthetic 20,000-row dataset)

│

├── models/

│   ├── model.pkl           (The trained Random Forest model)

│   └── scaler.pkl          (The saved StandardScaler object)

│

├── .gitignore

├── app.py                  (The main Streamlit application file)

├── generate_data.py        (Script to create the logical data)

├── model_training.py       (Script to train and save the model)

├── requirements.txt        (List of Python libraries required)

└── README.md               (You are here!)
