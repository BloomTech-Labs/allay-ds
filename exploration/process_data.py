import pandas as pd
from sklearn.model_selection import train_test_split


def reset_data_with_val():
    """
    Loads data, returns training and validation data"""
    tweets = pd.read_csv("exploration/data/combined_deduped.csv")
    train, test = train_test_split(tweets, test_size = .2, random_state=42)
    train, val = train_test_split(train, test_size = .15, random_state = 42)
    x_train, y_train, x_val, y_val = train["tweet"], train["inappropriate"], val["tweet"], val["inappropriate"]

    return x_train, y_train, x_val, y_val
