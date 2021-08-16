#!/usr/bin/env python3
from os import truncate

from pandas.io import feather_format
import numpy as np
import pandas as pd
from typing import Dict

class TreeStump:
    def __init__(self, weight: np.ndarray, featureName: str, classification: Dict):
        self.weight = weight
        self.featureName = featureName
        self.classification = classification

def calculate_gini(dataset: pd.DataFrame, featureName: str)->float:
    featureValues = np.sort(dataset.loc[:,featureName].unique())
    sizePerValue = dataset.groupby(featureName).size()
    distributions = []
    for featureValue in featureValues:
        distribution = dataset[dataset.loc[:,featureName] == featureValue].groupby('Y').size()/sizePerValue[featureValue]
        distributions.append(np.array(distribution))
    
    distributions = np.array(distributions)
    temp = (sizePerValue/len(dataset)).to_numpy()
    return np.dot(temp,(1 - np.array([np.sum(arr) for arr in (distributions ** 2)])))

def pick_next_feature(dataset: pd.DataFrame)->str:
    features = dataset.columns()[:-2]
    best_feature = features[0]
    min_gini = calculate_gini(dataset,best_feature)
    for feature in features[1:]:
        gini = calculate_gini(dataset,feature)
        if gini < min_gini:
            min_gini = gini
            best_feature = feature

    return best_feature



def get_missclassified(dataset: pd.DataFrame, stump: TreeStump, classColumn='Y')-> np.ndarray:
    missclassified = np.array([])
    for idx in dataset.index:
        row = dataset.loc[idx,:]
        prediction = stump.classification[row[stump.featureName]]
        if row[classColumn] != prediction:
            missclassified = np.insert(missclassified,0,idx)
    return missclassified


def train_stump(dataset:pd.DataFrame, stump: TreeStump, classColumn='Y'):
    for featureVal in stump.classification.keys():
        stump.classification[featureVal] = dataset.loc[dataset.loc[:,stump.featureName] == featureVal,classColumn].value_counts().idxmax()


def update_sample_weights(dataset: pd.DataFrame, missclassifiedSamples: np.ndarray, error: float)->None:
    allIndices = np.array([i for i in range(len(dataset))])
    classifiedSamples = np.delete(allIndices,missclassifiedSamples)

    dataset.loc[classifiedSamples,'Sample Weight'] /= error
    dataset.loc[classifiedSamples,'Sample Weight'] *= (2 / np.sum(dataset.loc[classifiedSamples,'Sample Weight']))

    dataset.loc[missclassifiedSamples, 'Sample Weigth'] /= (1-error) 
    dataset.loc[missclassifiedSamples,'Sample Weight'] *= (2 / np.sum(dataset.loc[missclassifiedSamples,'Sample Weight']))
        classifiedSamples

if __name__ == '__main__':
    dataset = pd.DataFrame(
        data={
            'Chest Pain':[True,False,True,True,False,False, True, True], 
            'Blocked Arteries':[True,True,False,True,True,True, False, True],
            'Patient Weight':[205,180,210,167,156,125,168,172],
            'Hearth Disease':[True,True,True,True,False,False,False,False]
        }
    )

    dataset['Sample Weight'] =  np.ones(len(dataset))/len(dataset)
    stumps = []


    while True:
        feature = pick_next_feature(dataset,dataset.columns())
        stump = TreeStump(1/2,feature,{})
        featureValues = dataset.loc[:,feature].unique()
        train_stump(dataset, stump, featureValues)

        missclassifiedExamples = get_missclassified(dataset,stump)
        if len(missclassifiedExamples) == 0:
            break

        error = np.sum(dataset.loc[missclassifiedExamples,'Sample Weight'])
        stump.weight = (1-error)/(2 * error)

        update_sample_weights(dataset, missclassifiedExamples)
        futureSampleIndices = np.random.choice(np.array([i for i in range(len(dataset))]),len(dataset),p=dataset.loc[:,'Sample Weight'])
        dataset = dataset.loc[futureSampleIndices,:]
        stumps.append(stump)

    

