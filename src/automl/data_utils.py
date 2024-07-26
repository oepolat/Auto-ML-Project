import src.automl.data as data
from pathlib import Path
import pandas as pd
import numpy as np
from itertools import chain

def iqr_outlier_det(df, col_name, scale):
    sorted_data = list(df[col_name])
    sorted_data.sort()

    #TODO: add replace with median

    step = round(len(sorted_data)/4)

    q1 = round(len(sorted_data)/2)-step
    q3 = round(len(sorted_data)/2)+step
    
    iqr = q3-q1 #Interquartile range

    lower_fence  = q1-scale*iqr
    higher_fence = q3+scale*iqr
    
    df_out = []
    df_out = [float(each) if float(each) >= lower_fence and float(each) <= higher_fence else np.nan for each in df[col_name]] 
    return df_out

def drop_outliers(X, y, scale = 1.5):
    for column in X.columns:
        X[column] = iqr_outlier_det(X, column, scale)

    y_temp = pd.DataFrame()

    y_temp["target"] = y

    df = pd.concat([X, y_temp],axis = 1)

    df.dropna(inplace=True)
    X_dropped = df.iloc[:,:-1]
    y_dropped = df.iloc[:,-1]
    return  X_dropped, y_dropped

def normalize_values(X):
    for col in X.columns:
        max_val = np.max(X[col])
        min_val = np.min(X[col])
        X[col] = [(each-min_val)/(max_val-min_val) for each in X[col].values]
    return X