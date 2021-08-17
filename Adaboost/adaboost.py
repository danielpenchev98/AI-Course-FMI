#!/usr/bin/env python3
from os import truncate

import numpy as np
import pandas as pd
from typing import Dict
from math import log

class TreeStump:
    def __init__(self, weight: np.ndarray, featureName: str, classification: Dict):
        self.weight = weight
        self.featureName = featureName
        self.classification = classification

    def train(self,dataset:pd.DataFrame)->None:
        classColumnName = dataset.columns[-2]
        featureValues = dataset[self.featureName].unique()
        for featureVal in featureValues:
            self.classification[featureVal] = dataset.loc[dataset.loc[:,self.featureName] == featureVal,classColumnName].value_counts().idxmax()

class AdaboostClassifier:
    def __init__(self):
        self.stumps = []
        self.stumpWeightsColumn = 'Sample Weight'

    def train(self, dataset: pd.DataFrame)->None:
        dataset[self.stumpWeightsColumn] =  np.ones(len(dataset))/len(dataset)

        while True:
            feature = pick_next_feature(dataset)
            stump = TreeStump(1/2,feature,{})
            stump.train(dataset)

            missclassifiedExamples = get_missclassified(dataset,stump)
            if len(missclassifiedExamples) == 0:
                break

            error = np.sum(dataset.loc[missclassifiedExamples,self.stumpWeightsColumn])
            stump.weight = 1/2 * log((1-error)/error)

            update_sample_weights(dataset, missclassifiedExamples,error, self.stumpWeightsColumn)
            futureSampleIndices = np.random.choice(np.array([i for i in range(len(dataset))]),len(dataset),p=dataset.loc[:,self.stumpWeightsColumn])
        

            dataset = dataset.loc[futureSampleIndices,:]
            dataset.reset_index(drop=True, inplace=True)
        
            self.stumps.append(stump)


def calculate_gini(dataset: pd.DataFrame, featureName: str)->float:
    classFeatureName = dataset.columns[-2]
    featureValues = np.sort(dataset.loc[:,featureName].unique())
    sizePerValue = dataset.groupby(featureName).size()
    distributions = []
    for featureValue in featureValues:
        distribution = dataset[dataset.loc[:,featureName] == featureValue].groupby(classFeatureName).size()/sizePerValue[featureValue]
        distributions.append(np.array(distribution,dtype=np.float32))
    
    distributions = np.array(distributions, dtype=object)
    temp = (sizePerValue/len(dataset)).to_numpy()
    return np.dot(temp,(1 - np.array([np.sum(arr) for arr in (distributions ** 2)])))

def pick_next_feature(dataset: pd.DataFrame)->str:
    features = dataset.columns[:-2]
    best_feature = features[0]
    min_gini = calculate_gini(dataset,best_feature)
    for feature in features[1:]:
        gini = calculate_gini(dataset,feature)
        if gini < min_gini:
            min_gini = gini
            best_feature = feature

    return best_feature

def update_sample_weights(dataset: pd.DataFrame, missclassifiedSamples: np.ndarray, error: float, weightColumnName: str)->None:
    allIndices = dataset.index
    classifiedSamples = np.delete(allIndices.to_numpy(),missclassifiedSamples)

    dataset.loc[classifiedSamples,weightColumnName] /=  error
    dataset.loc[classifiedSamples,weightColumnName] /=  (2 * np.sum(dataset.loc[classifiedSamples,weightColumnName]))

    dataset.loc[missclassifiedSamples, weightColumnName]  /= (1-error)
    dataset.loc[missclassifiedSamples,weightColumnName] /= (np.sum(dataset.loc[missclassifiedSamples,weightColumnName] * 2))


def get_missclassified(dataset: pd.DataFrame, stump: TreeStump)-> np.ndarray:
    classColumnName=dataset.columns[-2]
    missclassified = np.array([],dtype=np.int32)
    for idx in dataset.index:
        row = dataset.loc[idx,:]
        prediction = stump.classification[row[stump.featureName]]
        if row[classColumnName] != prediction:
            missclassified = np.insert(missclassified,0,idx)
    return missclassified


if __name__ == '__main__':
    dataset = pd.DataFrame(
        data={
            'Chest Pain':[True,False,True,True,False,False, True, True], 
            'Blocked Arteries':[True,True,False,True,True,True, False, True],
            'Patient Weight':[True,True,True,False,False,False,False,False], # is weight > 176
            'Hearth Disease':[True,True,True,True,False,False,False,False]
        }
    )

    classifier = AdaboostClassifier()
    classifier.train(dataset)
    




    

