

# this is the ann code that passes all test cases and it passes chatgpt hidden testcases.

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.exceptions import ConvergenceWarning
import warnings

# Ignore convergence warnings, but capture them to handle model retraining later
warnings.filterwarnings('ignore', category=ConvergenceWarning)

# Load and preprocess the data
# input: file_path: str (path to the dataset)
# output: tuple of X (features) and y (target)
def load_and_preprocess_data(file_path, fill_missing_with='mean'):
    # Load the dataset and drop unnecessary columns
    data = pd.read_csv(file_path)
    
    # Drop 'GarbageValues' column if it exists
    if 'garbage_column' in data.columns:
        data = data.drop(columns=['garbage_column'])
    
    # Handle missing data
    if fill_missing_with == 'mean':
        data.fillna(data.mean(), inplace=True)  # Fill missing values with column means
    elif fill_missing_with == 'median':
        data.fillna(data.median(), inplace=True)  # Fill missing values with column medians
    else:
        data = data.dropna()  # Default: drop rows with missing values
    
    # Separate features and target
    X = data.drop(columns=['quality'])  # 'quality' is assumed to be the outcome
    y = data['quality']
    
    return X, y
    pass

# Split the data into training and testing sets and standardize the features
# input: 1) X: list/ndarray (features)
#        2) y: list/ndarray (target)
# output: split: tuple of X_train, X_test, y_train, y_test
def split_and_standardize(X, y, test_size=0.2, random_state=42):
    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
    
    # Standardize the features
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    
    return X_train, X_test, y_train, y_test
    pass

# Create and train 2 MLP classifiers with different parameters
# input:  1) X_train: list/ndarray
#         2) y_train: list/ndarray
# output: 1) models: model1, model2 - tuple
def create_model(X_train, y_train, max_iter1=300, max_iter2=200):
    # Define parameters for model 1 (High Accuracy)
    model1 = MLPClassifier(hidden_layer_sizes=(100, 100, 100), max_iter=max_iter1, random_state=42)
    
    # Train model 1
    try:
        model1.fit(X_train, y_train)
    except ConvergenceWarning:
        print("Model 1 failed to converge. Increasing max_iter.")
        model1.set_params(max_iter=max_iter1 + 200)
        model1.fit(X_train, y_train)
    
    # Define parameters for model 2 (Moderate Accuracy)
    model2 = MLPClassifier(hidden_layer_sizes=(50, 50, 50), max_iter=max_iter2, random_state=42)
    
    # Train model 2
    try:
        model2.fit(X_train, y_train)
    except ConvergenceWarning:
        print("Model 2 failed to converge. Increasing max_iter.")
        model2.set_params(max_iter=max_iter2 + 200)
        model2.fit(X_train, y_train)
    
    return model1, model2
    pass

# Predict and evaluate the model's performance
# input  : 1) model: MLPClassifier after training
#          2) X_test: list/ndarray
#          3) y_test: list/ndarray
# output : 1) metrics: tuple - accuracy, precision, recall, fscore, confusion matrix
def predict_and_evaluate(model, X_test, y_test):
    # Predict on the test data
    y_pred = model.predict(X_test)
    
    # Calculate evaluation metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='weighted')  # Handles multiclass
    recall = recall_score(y_test, y_pred, average='weighted')        # Handles multiclass
    fscore = f1_score(y_test, y_pred, average='weighted')            # Handles multiclass
    conf_matrix = confusion_matrix(y_test, y_pred)
    
    return accuracy, precision, recall, fscore, conf_matrix
    pass