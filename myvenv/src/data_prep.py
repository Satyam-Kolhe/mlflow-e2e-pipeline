from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np

def load_and_split_data(test_size=0.2, random_state=42):
    df=load_breast_cancer()
    X=df.data
    y=df.target

    X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=test_size, random_state=random_state)
    return X_train, X_test, y_train, y_test

if __name__ == "__main__":
    X_train, X_test, y_train, y_test = load_and_split_data()
    print("Training data shape:", X_train.shape)
    print("Test data shape:", X_test.shape)
    print("Training labels shape:", y_train.shape)
    print("Test labels shape:", y_test.shape)
