import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

# Load and preprocess the data
# input: file_path: str (path to the dataset)
# output: tuple of X (features) and y (target)
def load_and_preprocess_data(file_path):
    # Load the dataset and drop unnecessary columns
    # Separate features and target
    data = pd.read_csv(file_path)
    if 'GarbageValues' in data.columns:
        data = data.drop(columns=['GarbageValues'])
    
    data = data.dropna()   #drops missing val rows
    X = data.drop(columns=['Outcome'])
    y = data['Outcome']
    
    return X, y
    pass

# Split the data into training and testing sets and standardize the features
# input: 1) X: list/ndarray (features)
#        2) y: list/ndarray (target)
# output: split: tuple of X_train, X_test, y_train, y_test
def split_and_standardize(X, y):
    # Split the data into training and testing sets
    # Standardize the features
    
    # x--features and y--targets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    
    return X_train, X_test, y_train, y_test
    pass

# Create and train 2 MLP classifiers with different parameters
# input:  1) X_train: list/ndarray
#         2) y_train: list/ndarray
# output: 1) models: model1, model2 - tuple
def create_model(X_train, y_train):
    # Define parameters for model 1
    # Train model 1
    model1 = MLPClassifier(hidden_layer_sizes=(100, 100, 100), max_iter=300, random_state=42)  # 75--.58 200--.5__
    model1.fit(X_train, y_train)
    # model2
    model2 = MLPClassifier(hidden_layer_sizes=(50, 50, 50), max_iter=200, random_state=42)
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
    # Calculate evaluation metrics
    y_pred = model.predict(X_test)
    
    accuracy = accuracy_score(y_test, y_pred)
    # print("accuracy:",accuracy)
    precision = precision_score(y_test, y_pred, average='weighted')  
    # print("precision:",precision)
    recall = recall_score(y_test, y_pred, average='weighted')    
    # print("recall:",recall)    
    fscore = f1_score(y_test, y_pred, average='weighted')     
    # print("fscore:",fscore)       
    conf_matrix = confusion_matrix(y_test, y_pred)
    
    return accuracy, precision, recall, fscore, conf_matrix
    pass
