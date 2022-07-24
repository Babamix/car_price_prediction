import pandas as pd
import pickle

from helper.data_check_preparation import read_and_check_data
from helper.feature_engineering import feature_engineering
from helper.constant import TRAIN_COLUMN, PATH

from sklearn.model_selection import train_test_split

def train_model():
    # pembacaan dan pengecekan data
    df = read_and_check_data(PATH, TRAIN_COLUMN)
    
    # siapkan fitur data dan target data
    X = df
    y = df["price"]

    # data splitting
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=123)
    
    # feature engineering
    X_train_transformed = feature_engineering(df = X_train, state = "train")
    print("Start Saving Result Feature Engineering in Train Data !")
    X_train_transformed.to_csv("artifacts/X_train_transformed.csv")
    
    # feature engineering
    X_test_transformed = feature_engineering(df = X_test, state = "test")
    print("Start Saving Result Feature Engineering in Test Data!")
    X_test_transformed.to_csv("artifacts/X_test_transformed.csv")
    
if __name__ == "__main__":
    print("START RUNNING PIPELINE!")
    train_model()
    print("FINISH RUNNING PIPELINE!")