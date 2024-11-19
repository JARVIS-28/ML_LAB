import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix


def load_and_preprocess_data(file_path):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2 random_state=42)
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    return X_train, X_test, y_train, y_test
    pass
   
def split_and_standardize(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2 random_state=42)
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    return X_train, X_test, y_train, y_test
    pass



def create_model(X_train, y_train):
    model1 = MLPClassifier(hidden_layer_sizes=(100, 50, 25), 
                           activation='relu',
                           solver='adam',
                           learning_rate_init=0.001,
                           max_iter=500,
                           random_state=42)
    model1.fit(X_train, y_train)
    # Model 2: Lower accuracy model (between 50% to 95%)
    model2 = MLPClassifier(hidden_layer_sizes=(50, 30, 10),  
                           activation='tanh',  # Change activation function
                           solver='adam',
                           learning_rate_init=0.01,  # Higher learning rate
                           max_iter=300,
                           random_state=42)
    
    model2.fit(X_train, y_train)
    return model1, model2
    pass

def predict_and_evaluate(model, X_test, y_test):
    y_pred_test = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred_test)
    precision = precision_score(y_test, y_pred_test, average='weighted')  # Use 'weighted' for multiclass
    recall = recall_score(y_test, y_pred_test, average='weighted')
    fscore = f1_score(y_test, y_pred_test, average='weighted')
    conf_matrix = confusion_matrix(y_test, y_pred_test)
    return accuracy, precision, recall, fscore, conf_matrix
    pass
