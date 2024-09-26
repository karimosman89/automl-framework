import pandas as pd
from sklearn.model_selection import train_test_split

def preprocess_data(data):
    X = data.drop('target', axis=1)
    y = data['target']
    return train_test_split(X, y, test_size=0.2, random_state=42)

