# Acceleration-Based Features for Continuous User Authentication Using Neural Networks

This project implements a continuous authentication system in MATLAB using acceleration data from smart devices. It explores the viability of using behavioral biometrics (gait/walking patterns) to continuously and passively verify a user's identity.

The system processes raw 3-axis accelerometer data from 10 users, extracts a set of time-domain and frequency-domain features, and trains a Support Vector Machine (SVM) classifier to build a biometric model for each user. The model's performance is rigorously evaluated using standard classification and biometric authentication metrics.

## 🚀 Key Features

* **Feature Extraction:** Extracts 21 statistical features (time-domain and frequency-domain) from raw accelerometer data using a 2-second sliding window.
* **Classifier:** Implements a multi-class Support Vector Machine (SVM) for robust user classification.
* **Data Visualization:** Includes t-SNE (cluster) plots and feature scatter plots to visualize user separability.
* **Performance Evaluation:** Automatically calculates and plots key metrics:
    * **Accuracy:** Overall classification accuracy.
    * **Confusion Matrix:** A detailed plot of classification errors.
    * **FAR/FRR/EER:** Generates the False Acceptance Rate (FAR), False Rejection Rate (FRR), and Equal Error Rate (EER) curves, which are the standard for authentication systems.

## Dataset

[cite_start]The dataset consists of acceleration data from **10 users**. [cite: 18]
* [cite_start]**Data:** 3-axis (x, y, z) acceleration data. [cite: 18]
* [cite_start]**Sessions:** Each user performed two 6-minute walking sessions on separate days (Session 1 (FD) for Training, Session 2 (MD) for Testing). [cite: 19]
* [cite_start]**Sampling:** Data was recorded at ~30-32 samples per second. [cite: 18]

## 📋 How to Run

1.  **Ensure you have the required toolboxes** (see Dependencies).
2.  Place all 20 dataset files (e.g., `U1NW_FD.csv`, `U1NW_MD.csv`... `U10NW_MD.csv`) in the same directory as the `.m` files.
3.  Open MATLAB.
4.  Open the `main_authentication.m` script.
5.  Click the **"Run"** button.

The script will process all the data, train the SVM, and automatically generate four figures:
1.  t-SNE Cluster Plot
2.  Feature Scatter Plot
3.  SVM Confusion Matrix
4.  FAR/FRR/EER Curve

All results (Accuracy, EER, etc.) will be printed to the MATLAB Command Window.

## 📂 Code Structure

This project is organized into three `.m` files:

* **`main_authentication.m`**
    This is the main executable script. It handles data loading, defines parameters, calls the helper functions, trains the classifier, and generates all plots and results.

* **`extractFeatures.m`**
    A helper function that takes raw sensor data as input. It segments the data using a sliding window and calculates the 21-feature vector for each window.

* **`calculateEER.m`**
    A helper function that takes the trained model's scores as input. It calculates the FAR, FRR, and EER by iterating over genuine and imposter scores.

## 🛠️ Dependencies

This code was written for **MATLAB** and requires the following toolboxes:
* **Statistics and Machine Learning Toolbox:** (For `fitcecoc`, `tsne`, `confusionchart`)
* **Signal Processing Toolbox:** (For `fft`, `rms` - *recommended*)
