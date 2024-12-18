from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
import os
import glob
from tqdm import tqdm
def find_values(file_path):
   
    try:
        data = pd.read_csv(file_path)
    except Exception as e:
            print(f'Error with filepath {file_path}', repr(e))
    return data
filepaths = [file for file in glob.glob(f'D:/bruh/trading_deep_learning/stock_data/*.csv')]
for file_path in tqdm(filepaths):
    file_name_without_extension = os.path.splitext(os.path.basename(file_path))[0]

    X = find_values(file_path)
    df_train, df_test = train_test_split(X,test_size=0.2, random_state=1)
    df_train.drop('Unnamed: 0', axis=1, inplace=True)
    df_test.drop('Unnamed: 0', axis=1, inplace=True)
    df_train.drop('time', axis=1, inplace=True)
    df_test.drop('time', axis=1, inplace=True)
    df_train.drop('spread', axis=1, inplace=True)
    df_test.drop('spread', axis=1, inplace=True)
    df_test.to_csv(f"D:/bruh/trading_deep_learning/test_data/{file_name_without_extension}_test.csv")
    df_train.to_csv(f"D:/bruh/trading_deep_learning/train_data/{file_name_without_extension}_train.csv")
