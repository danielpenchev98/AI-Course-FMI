#!/usr/bin/env python3
from os import truncate

from math import log
from typing import Dict
import numpy as np
import pandas as pd

ENTITY_WEIGHT_COLUMN = 'Sample Weight'
CLASS_COLUMN = 'Class'

class TreeStump:
    def __init__(self, weight: np.ndarray, feature_name: str, classification: Dict):
        self.weight = weight
        self.featureName = feature_name
        self.classification = classification

    def train(self,dataset:pd.DataFrame)->None:
        featureValues = dataset[self.featureName].unique()
        for featureVal in featureValues:
            self.classification[featureVal] = dataset.loc[dataset.loc[:,self.featureName] == featureVal,CLASS_COLUMN].value_counts().idxmax()

    def classify(self, entity: pd.Series):
        return self.classification(entity[self.featureName])

class AdaboostClassifier:
    def __init__(self):
        self.stumps = []

    def train(self, dataset: pd.DataFrame)->None:
        dataset[ENTITY_WEIGHT_COLUMN] =  np.ones(len(dataset))/len(dataset)

        default_stump_weight = 1/2
        while True:
            feature = pick_next_feature(dataset)
            stump = TreeStump(default_stump_weight,feature,{})
            stump.train(dataset)

            misclassified_indices = get_missclassified(dataset,stump)
            if len(misclassified_indices) == 0:
                break

            error = self._calculate_error(dataset, misclassified_indices)
            stump.weight = self._calculate_stump_weight(error)

            update_sample_weights(dataset, misclassified_indices,error)
            self.stumps.append(stump)

            dataset = sample_dataset(dataset)
    
    def _calculate_stump_weight(self,error: float) -> float:
        return  1/2 * log((1-error)/error)

    def _calculate_error(self,dataset: pd.DataFrame, misclassified_indices: np.ndarray)->int:
        return np.sum(dataset.loc[misclassified_indices,ENTITY_WEIGHT_COLUMN])

    def classify(self,entity: pd.Series):
        return sum(stump.weight * stump.classify(entity) for stump in self.stumps)        

def calculate_gini(dataset: pd.DataFrame, featureName: str)->float:
    feature_values = np.sort(dataset.loc[:,featureName].unique())
    feature_value_count = dataset.groupby(featureName).size()
    distributions = []
    for featureValue in feature_values:
        distribution = dataset[dataset[featureName] == featureValue].groupby(CLASS_COLUMN).size()/feature_value_count[feature_values]
        distributions.append(np.array(distribution,dtype=np.float32))

    distributions = np.array(distributions, dtype=object)
    feature_value_distribution = feature_value_count.to_numpy() /len(dataset)
    return np.dot(feature_value_distribution,np.array([1 - np.sum(arr) for arr in distributions ** 2]))

def pick_next_feature(dataset: pd.DataFrame)->str:
    features = dataset.columns[:-2]
    winnerIdx = np.array([calculate_gini(dataset,feature) for feature in features]).argmin()
    return features[winnerIdx]

def update_sample_weights(dataset: pd.DataFrame, misclassified_indices: np.ndarray, error: float) -> None:
    classified_indices = np.delete(dataset.index.to_numpy(),misclassified_indices)

    # update the weights of the successfully classified entities
    dataset.loc[classified_indices,ENTITY_WEIGHT_COLUMN] /=  error
    # normalize the weights of the successfully classified entities
    dataset.loc[classified_indices,ENTITY_WEIGHT_COLUMN] /=  (np.sum(dataset.loc[classified_indices,ENTITY_WEIGHT_COLUMN]) * 2)

    # update the weights of the misclassified entities
    dataset.loc[misclassified_indices, ENTITY_WEIGHT_COLUMN]  /= (1-error)
    # normalize the weights of the misclassified classified entities
    dataset.loc[misclassified_indices,ENTITY_WEIGHT_COLUMN] /= (np.sum(dataset.loc[misclassified_indices,ENTITY_WEIGHT_COLUMN]) * 2)


def get_missclassified(dataset: pd.DataFrame, stump: TreeStump)-> np.ndarray:
    missclassified = np.array([],dtype=np.int32)
    for idx in dataset.index:
        row = dataset.loc[idx,:]
        prediction = stump.classification[row[stump.featureName]]
        if row[CLASS_COLUMN] != prediction:
            missclassified = np.insert(missclassified,0,idx)
    return missclassified

def sample_dataset(dataset: pd.DataFrame):
    dataset_size = len(dataset)
    dataset = dataset.sample(n=dataset_size,weights=ENTITY_WEIGHT_COLUMN, replace=True, ignore_index=True)
    #reset the row indices
    dataset.reset_index(drop=True, inplace=True)
    #normalize entity weights
    dataset[ENTITY_WEIGHT_COLUMN] /= np.sum(dataset[ENTITY_WEIGHT_COLUMN])
    return dataset

if __name__ == '__main__':
    dataset = pd.DataFrame(
        data={
            'Chest Pain':[True,False,True,True,False,False, True, True], 
            'Blocked Arteries':[True,True,False,True,True,True, False, True],
            'Patient Weight':[True,True,True,False,False,False,False,False], # is weight > 176
            'Hearth Disease':[True,True,True,True,False,False,False,False]
        }
    )

    dataset.rename(columns={dataset.columns[-1]:CLASS_COLUMN}, inplace=True)

    classifier = AdaboostClassifier()
    classifier.train(dataset)
    




    

