#!/usr/bin/env python3

import pandas as pd
import numpy as np
import math
from typing import Dict
from typing import Tuple


def load_dataset(filename: str) -> pd.DataFrame:
    return pd.read_csv(filename, header=0)

def create_class_mapping(df: pd.DataFrame) -> Tuple[np.ndarray, Dict[str,int]]:
    classes = df["class"].unique()
    class_idx_mapping = {class_name:idx for idx, class_name in enumerate(classes)}
    return classes, class_idx_mapping

def prepare_dataset(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    df = df.sample(frac=1).reset_index(drop=True)    
    training_dataset_size = math.floor(df.shape[0] * 0.65)
    training_df = df.iloc[:training_dataset_size,:]

    test_df = df.iloc[training_dataset_size:,:]
    
    return training_df, test_df

def train_model(df :pd.DataFrame, classes: pd.Series, class_mapping: Dict[str, int]) -> Tuple[np.ndarray,np.ndarray,np.ndarray]:
    num_feature_cols = len(df.columns) - 1
    prior_prob = np.zeros(classes.size)
    mean = np.zeros((classes.size,num_feature_cols))
    cov_matrix = np.zeros((classes.size, num_feature_cols, num_feature_cols))

    for class_name in classes:
        
        df_subset = df[df["class"]==class_name].iloc[:,:-1].to_numpy()
        
        class_idx = class_mapping[class_name]
        prior_prob[class_idx] = 1.0 * df_subset.shape[0] / df.shape[0]
        mean[class_idx] = df_subset.sum(axis=0) * 1.0 / df_subset.shape[0]
        
        temp = df_subset - mean[class_idx]
        cov_matrix[class_idx] = np.matmul(temp.T, temp) * 1.0 / df_subset.shape[0]

    return mean, cov_matrix, prior_prob

def predict(records: pd.DataFrame, mean: np.ndarray, cov_matrix: np.ndarray, prior: np.ndarray) -> np.ndarray:
    cov_matrix = cov_matrix[np.newaxis,:,:,:] # (1, |classes| x |features| x |features|)
    mean = mean[np.newaxis,:,:] # (1 x |classes| x |features|)
    
    records = records[:,np.newaxis,:] #(|records|, 1, |features|)
    
    centralized_records = records - mean
    temp = np.exp(- .5 * np.matmul(np.matmul(centralized_records[:,:,np.newaxis,:], np.linalg.inv(cov_matrix)),centralized_records[:,:,:,np.newaxis])).squeeze()
    pxy = 1.0 / np.sqrt((2*np.pi) ** 2 * np.linalg.det(cov_matrix)) * temp
    
    pyx = pxy * prior
    return pyx.argmax(axis=1).flatten()

#TODO implement K fold cross falidation
def k_fold_cross_validation(df, k=10):
    pass
        

df = load_dataset("iris_csv.csv")
classes, class_idx_mapping = create_class_mapping(df)
training_df, test_df = prepare_dataset(df)

mean, cov_matrix, prior = train_model(df, classes, class_idx_mapping)
predicted_class_idx = predict(test_df.iloc[:,:-1].to_numpy(),mean,cov_matrix,prior)


guessed = 0

test_df = test_df.reset_index()
for record_idx, test_record in test_df.iterrows():
    if class_idx_mapping[test_record[-1]] == predicted_class_idx[record_idx]:
        guessed +=1

print("Accuracy :{}".format(guessed * 1.0 / test_df.shape[0]))

