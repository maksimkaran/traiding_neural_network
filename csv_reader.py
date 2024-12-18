import numpy as np
import pandas as pd
import math
import glob
def find_values(file_path):
   
    try:
        data = pd.read_csv(file_path)
    except Exception as e:
            print(f'Error with filepath {file_path}', repr(e))
    Y = data['buy']
    data.drop(columns = 'buy',axis = 1, inplace=True)
    data.drop(columns = 'datetime',axis = 1, inplace=True)
    X = data
    X = X.to_numpy()
    X_norm = []
    for i in range(1,data.shape[1]):
        
        column = X[:,i]/abs(X[:,i]).max()
        X_norm.append(column)
    X_norm = np.array(X_norm)
    Y = Y.to_numpy()
    print(X_norm)
    return X_norm,Y

file_path = "D:/bruh/trading_deep_learning/A.csv"
x,y = find_values(file_path)