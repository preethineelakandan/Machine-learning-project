import pandas as pd 
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import os
from joblib import dump, load
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


def evaluation_pipeline(x_test_path, y_test_path, model_path):
    x_data = pd.read_csv(x_test_path)
    y_true = pd.read_csv(y_test_path)

    loaded_scaler = load('models/scaler/min_max_scaler.pkl')
    x_data_normalized = loaded_scaler.transform(x_data)
    
    loaded_model = load('models/logistic_regression.pkl')
    y_pred = loaded_model.predict(x_data_normalized)
    y_pred_df = pd.DataFrame(y_pred, columns=['Predicted_Label'])
    accuracy = accuracy_score(y_true, y_pred) * 100
    
    return y_pred_df, accuracy
