# Binary Prediction with a Rainfall Dataset - README

## Overview
This project is a solution for the Kaggle competition "Playground Series S5E3" - Binary Prediction with a Rainfall Dataset. The goal is to predict whether rainfall will occur (a binary outcome: yes or no) based on a set of features provided in the dataset. The code builds, trains, and evaluates a neural network model using Python libraries like TensorFlow, Pandas, and Scikit-learn, then generates predictions for a test dataset to submit to Kaggle.


---

## What the Code Does
Here’s a breakdown of what happens in the code:

### 1. Setting Up the Environment
- **Imports Kaggle Data**: The code connects to Kaggle using the `kagglehub` library and downloads the competition dataset (`playground-series-s5e3`), which includes `train.csv` and `test.csv` files.
- **Loads Libraries**: It uses libraries like:
  - `pandas` for handling data tables.
  - `numpy` for math operations.
  - `sklearn` for splitting data and scaling features.
  - `tensorflow` for building and training the neural network.
- **Fixes Randomness**: Sets a random seed (42) to make results reproducible.

### 2. Loading and Preparing the Data
- **Loads Datasets**: Reads `train.csv` (training data with features and rainfall labels) and `test.csv` (test data without labels).
- **Separates Features and Target**:
  - From `train.csv`: Drops the `id` and `rainfall` columns to get features (`X`), and keeps `rainfall` as the target (`y`).
  - From `test.csv`: Drops `id` to get test features (`X_test`) and saves `id` for the submission file.
- **Handles Missing Values**: Fills any missing data in the features with the average (mean) value of each column.
- **Scales Features**: Uses `StandardScaler` to normalize the data (important for neural networks to work well).

### 3. Splitting the Training Data
- Splits the training data into two parts:
  - 80% for training the model (`X_train`, `y_train`).
  - 20% for validating the model (`X_val`, `y_val`).
- Keeps the split balanced (`stratify=y`) so the yes/no rainfall ratio stays the same.

### 4. Building the Neural Network Model
- Creates a neural network using TensorFlow’s `Sequential` model with these layers:
  - **Input Layer**: 128 neurons with `relu` activation (takes the scaled features).
  - **Batch Normalization**: Adjusts data between layers for better training.
  - **Dropout (0.3)**: Randomly ignores 30% of neurons to prevent overfitting.
  - **Hidden Layer 1**: 64 neurons with `relu`.
  - **Batch Normalization + Dropout (0.3)**.
  - **Hidden Layer 2**: 32 neurons with `relu`.
  - **Batch Normalization + Dropout (0.2)**.
  - **Output Layer**: 1 neuron with `sigmoid` activation (outputs a probability between 0 and 1 for rainfall).
- This structure helps the model learn patterns while avoiding over-reliance on the training data.

### 5. Compiling and Training the Model
- **Compiles the Model**: Uses:
  - Optimizer: `adam` (a method to adjust the model efficiently).
  - Loss: `binary_crossentropy` (measures error for binary yes/no predictions).
  - Metric: `accuracy` (tracks how often predictions are correct).
- **Early Stopping**: Stops training if the validation loss doesn’t improve for 10 epochs (rounds), keeping the best version of the model.
- **Trains the Model**: Runs for up to 100 epochs with a batch size of 32, using the training data and checking performance on the validation data.

### 6. Evaluating the Model
- Predicts probabilities for the validation set (`y_val_pred`).
- Calculates the **ROC AUC score** (a measure of how well the model separates yes/no rainfall cases) and prints it. Higher is better (max is 1.0).

### 7. Making Predictions and Submission
- Predicts rainfall probabilities for the test set (`test_pred`).
- Creates a submission file (`submission.csv`) with two columns: `id` (from `test.csv`) and `rainfall` (predicted probabilities).

---
