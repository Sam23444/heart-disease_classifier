# heart-disease_classifier

# Heart Disease Prediction Project

## Overview
The Heart Disease Prediction Project is a machine learning-based system designed to predict the likelihood of heart disease in individuals based on medical data. The project utilizes a Random Forest Classifier to analyze features such as age, cholesterol levels, blood pressure, and more. This project aims to provide preliminary insights into heart health, assisting healthcare providers in early detection and prevention strategies.

---

## Features
- Predicts the likelihood of heart disease based on user input.
- Implements feature scaling for better model performance.
- Utilizes cross-validation to enhance accuracy and reduce overfitting.
- Provides model evaluation metrics such as confusion matrix, accuracy, and classification report.

---

## Dataset
The dataset includes medical features such as:
- Age
- Gender
- Chest pain type
- Resting blood pressure
- Cholesterol level
- Maximum heart rate achieved
- Exercise-induced angina
- ST depression
- And more...

**Target variable:** Presence (1) or absence (0) of heart disease.

---

## Prerequisites
Ensure you have the following installed:
- Python 3.7 or higher
- Jupyter Notebook or any Python IDE
- Libraries: `numpy`, `pandas`, `scikit-learn`, `matplotlib`

---

## Installation
1. Clone the repository:
   ```bash
   git clone <repository-url>
   ```
2. Navigate to the project directory:
   ```bash
   cd heart_disease_prediction
   ```
3. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

---

## Usage
1. Open the Jupyter Notebook:
   ```bash
   jupyter notebook Heart_disease_prediction.ipynb
   ```
2. Follow the cells in the notebook to:
   - Load and preprocess the dataset.
   - Train the machine learning model.
   - Evaluate the model.
   - Test with new input data.

3. To run the standalone Python script:
   ```bash
   python heart_disease_prediction.py
   ```

---

## Model Details
- **Algorithm:** Random Forest Classifier
- **Hyperparameters:**
  - `n_estimators`: 50
  - `max_depth`: 10
  - `min_samples_split`: 5
  - `min_samples_leaf`: 2
- **Cross-validation:** 5-fold cross-validation used to evaluate model performance.

---

## Results
- **Accuracy:** Achieved using validation and test sets.
- **Evaluation Metrics:**
  - Confusion Matrix
  - Accuracy Score
  - Classification Report

---

## Contributions
Contributions are welcome! Feel free to fork the repository and submit a pull request with your improvements.

---

## License
This project is licensed under the MIT License. See the LICENSE file for more information.

---

## Acknowledgments
- The dataset used in this project is publicly available.
- Inspiration from healthcare AI applications to aid in early disease detection.

