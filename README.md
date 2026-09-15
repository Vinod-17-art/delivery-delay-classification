#  Delivery Delay Classification using Scikit-learn

A machine learning classification project that predicts whether a delivery will be **On Time** or **Delayed** based on delivery distance, weather, traffic conditions, vehicle type, order type, preparation time, driver experience, and peak-hour status.

The project is implemented using **Python and Scikit-learn** and includes data preprocessing, model training, evaluation, model serialization, and prediction on new delivery data.

---

##  Project Overview

Delivery delays can be influenced by several factors such as traffic, weather, delivery distance, preparation time, and peak-hour conditions.

This project builds a supervised machine learning model to classify delivery status into two categories:

*  **On Time**
*  **Delayed**

The model uses a **Random Forest Classifier** combined with a Scikit-learn preprocessing pipeline to handle both categorical and numerical features.

---

##  Objectives

* Analyze factors that may influence delivery delays.
* Preprocess categorical and numerical delivery features.
* Train a classification model using Random Forest.
* Evaluate model performance using accuracy and classification metrics.
* Generate a confusion matrix for visual evaluation.
* Save the trained model for future predictions.
* Predict delivery status for new delivery records.

---

##  Dataset

The project uses a synthetic dataset containing **1,500 delivery records**.

### Features

| Feature                   | Description                                              |
| ------------------------- | -------------------------------------------------------- |
| `distance_km`             | Delivery distance in kilometers                          |
| `weather`                 | Weather condition during delivery                        |
| `traffic_level`           | Traffic level: Low, Medium, or High                      |
| `vehicle_type`            | Delivery vehicle such as Bike, Car, Van, or Truck        |
| `order_type`              | Type of order such as Food, Grocery, Medicine, or Parcel |
| `preparation_time_min`    | Order preparation time in minutes                        |
| `driver_experience_years` | Driver's experience in years                             |
| `peak_hour`               | Indicates whether the delivery occurs during peak hours  |
| `delivery_status`         | Target variable: On Time or Delayed                      |

### Target Distribution

* **On Time:** 921 records
* **Delayed:** 579 records

The dataset contains no missing values.

> **Note:** The dataset is synthetic and is intended for educational and academic purposes.

---

##  Machine Learning Approach

The project uses the following workflow:

```text
Raw Dataset
     ↓
Separate Features & Target
     ↓
Train/Test Split
     ↓
Categorical Feature Encoding
     ↓
Random Forest Classifier
     ↓
Model Evaluation
     ↓
Save Trained Model
     ↓
Make Predictions
```

### Model

**Random Forest Classifier**

Configuration:

* `n_estimators = 250`
* `random_state = 42`
* `class_weight = "balanced"`

### Preprocessing

Categorical features are transformed using:

```python
OneHotEncoder(handle_unknown="ignore")
```

The preprocessing and classifier are combined into a Scikit-learn `Pipeline`, making the same transformations automatically available during prediction.

---

##  Model Performance

Using an **80/20 train-test split** with stratification, the model achieved:

**Accuracy: 95.67%**

### Classification Report

| Class                | Precision | Recall | F1-Score |
| -------------------- | --------: | -----: | -------: |
| Delayed              |      0.96 |   0.92 |     0.94 |
| On Time              |      0.95 |   0.98 |     0.97 |
| **Overall Accuracy** |           |        | **0.96** |

Test set size: **300 records**

A confusion matrix is also generated during model training and saved as:

```text
confusion_matrix.png
```

---

##  Project Structure

```text
Delivery_Delay_Classification_Sklearn/
│
├── data/
│   └── delivery_delay_dataset.csv
│
├── delivery_delay_classifier.pkl
├── train_model.py
├── predict.py
├── confusion_matrix.png
├── requirements.txt
└── README.md
```

### File Description

**`data/delivery_delay_dataset.csv`**
Synthetic dataset used to train and evaluate the model.

**`train_model.py`**
Loads the dataset, preprocesses the features, trains the Random Forest model, evaluates performance, generates the confusion matrix, and saves the trained model.

**`predict.py`**
Loads the saved model and predicts the delivery status for a new delivery example.

**`delivery_delay_classifier.pkl`**
Serialized trained Scikit-learn pipeline used for making predictions.

**`confusion_matrix.png`**
Visualization of model classification performance.

**`requirements.txt`**
Python dependencies required to run the project.

---

##  Installation

Clone the repository:

```bash
git clone <YOUR_GITHUB_REPOSITORY_URL>
cd Delivery_Delay_Classification_Sklearn
```

Install the required dependencies:

```bash
pip install -r requirements.txt
```

---

##  Train the Model

Run:

```bash
python train_model.py
```

The script will:

1. Load the delivery dataset.
2. Separate input features and target.
3. Encode categorical variables.
4. Split the dataset into training and testing sets.
5. Train the Random Forest classifier.
6. Evaluate the model.
7. Generate a classification report.
8. Generate and save the confusion matrix.
9. Save the trained model as `delivery_delay_classifier.pkl`.

---

##  Make a Prediction

Run:

```bash
python predict.py
```

The prediction script uses a sample delivery such as:

```text
Distance: 28.5 km
Weather: Rain
Traffic: High
Vehicle: Bike
Order Type: Food
Preparation Time: 42 minutes
Driver Experience: 2 years
Peak Hour: Yes
```

The model returns:

```text
Predicted Delivery Status: Delayed
```

along with the probability for each class.

---

##  Technologies Used

* **Python**
* **Pandas** – Data loading and manipulation
* **NumPy** – Numerical operations
* **Scikit-learn** – Machine learning and preprocessing
* **Random Forest** – Classification algorithm
* **Joblib** – Model serialization
* **Matplotlib** – Confusion matrix visualization

---

##  Key Machine Learning Concepts Demonstrated

This project demonstrates practical usage of:

* Binary classification
* Train/test splitting
* Stratified sampling
* Categorical feature encoding
* One-hot encoding
* Scikit-learn pipelines
* ColumnTransformer
* Random Forest classification
* Class weighting
* Model evaluation
* Precision, recall, and F1-score
* Confusion matrix
* Model serialization with Joblib
* Making predictions with a saved model

---

## Future Improvements

Possible improvements include:

* Hyperparameter tuning using `GridSearchCV` or `RandomizedSearchCV`
* Cross-validation
* Feature importance analysis
* ROC-AUC evaluation
* Precision-Recall curves
* Exploratory Data Analysis (EDA)
* More realistic real-world delivery data
* Handling class imbalance with additional techniques
* Building a Streamlit web application
* Deploying the model as a REST API
* Adding automated model retraining

---

##  Disclaimer

This project uses a **synthetic dataset** and should not be used for real-world delivery decisions without validation using representative production data.

---

##  Author

**Karanth K**

Machine Learning / Data Science Project

---

 If you find this project useful, consider giving the repository a star!
# delivery-delay-classification
Machine learning project using Scikit-learn and Random Forest to predict whether deliveries will be On Time or Delayed based on distance, weather, traffic, vehicle type, preparation time, driver experience, and peak-hour conditions.
