import pandas as pd
import numpy as np
import math
import random
from sklearn import metrics

class Node:
    def __init__(self, left_child, right_child, records_cnt, split_attribute = None, split_value = None):
        self.left_child = left_child
        self.right_child = right_child
        self.records_cnt = records_cnt
        self.split_attribute = split_attribute
        self.split_value = split_value
        
    def is_external(self):
        return self.left_child is None and self.right_child is None

# add height
class Tree:
    def __init__(self, df, height_limit):
        self.height_limit = height_limit
        self.attribute_list = df.columns
        self.root = self._build(df, 0)
    
    def _build(self, df, curr_height):
        if curr_height >= self.height_limit or df.shape[0] < 2 or df.drop_duplicates().shape[0] == 1: # if the branch has reached its max length or there is only one record in the dataset or all of the records have the same stats
            return Node(None, None, df.shape[0])
        
        split_attribute = random.choice(self.attribute_list) # pick a random attribute
        max_val, min_val = df[split_attribute].max(), df[split_attribute].min()
        split_value = random.uniform(min_val, max_val) # pick random number from [min,max]
        
        left_child_df = df[df[split_attribute] < split_value] 
        right_child_df = df[df[split_attribute] >= split_value]
        
        if left_child_df.shape[0] == 0 or right_child_df.shape[0] == 0: # if  the split_value is on the border on the interval, very unlikely event
            return self._build(df,curr_height)
        
        left_child = self._build(left_child_df, curr_height + 1)
        right_child = self._build(right_child_df, curr_height + 1)
                
        return Node(left_child, right_child, df.shape[0], split_attribute, split_value)

    def path_length(self, record):
        return self._path_length(record, self.root, 0)
    
    def adjustment(records_cnt): # when the subtree is prematurely stopped from further construction
        if records_cnt == 1:
            return 0
        
        EULER_CONSTANT = 0.5772156649
        return 2 * (math.log(records_cnt-1) + EULER_CONSTANT) - (2.0 * (records_cnt - 1)) / records_cnt
        
    
    def _path_length(self, record, curr_node, curr_height):
        if curr_node.is_external():
            return curr_height + Tree.adjustment(curr_node.records_cnt)
        
        if record[curr_node.split_attribute] < curr_node.split_value:
            return self._path_length(record, curr_node.left_child, curr_height + 1)
        else:
            return self._path_length(record, curr_node.right_child, curr_height + 1)

class IForest:
    def __init__(self, tree_cnt, sampling_size, features_sample_cnt=20):
        self.tree_cnt = tree_cnt # number of trees in the assemble of tree classifiers
        self.trees = None # assemble of tree classifiers
        self.sampling_size = sampling_size # number of records to be used per tree classifier for training
        self.features_sample_cnt = features_sample_cnt
        
    def train(self, df):
        self.trees = []
        height_limit =  math.ceil(math.log2(self.sampling_size)) # max tree height
        for i in range(self.tree_cnt):
            sample_df = df.sample(n=self.sampling_size, replace=False) # sample dataset records for the creation of a tree
            
            kurtosis = sample_df.kurtosis(axis=0, numeric_only=True)
            picked_features = list(map(lambda x: x[0], sorted(enumerate(kurtosis),key=lambda x: x[1])[-self.features_sample_cnt:]))
            print(picked_features)
            print(sample_df.shape)
            sample_df = sample_df.iloc[:,picked_features]
            
            tree = Tree(sample_df, height_limit)
            self.trees.append(tree)        
    
    def predict_anomalies(self, test_records, anomaly_score_threshold = 0.5):
        anomality_scores = []
        for _,record in test_records.iterrows():
            path_lengths = []
            for tree in self.trees:
                path_lengths.append(tree.path_length(record)) # calculate the search path for each tree to the target node
            
            avg_path_length = sum(path_lengths) * 1.0 / len(path_lengths) # calculate the avg path from all trees
            anomality_score = 2.0 ** -(avg_path_length / Tree.adjustment(self.sampling_size)) # anomality score calculation
            anomality_scores.append(anomality_score)
        
        return [anomality_score > anomaly_score_threshold for  anomality_score in anomality_scores]

def f1_score(expected_classes, predicted_classes, eps=0.001):
    table = [[0,0],[0,0]]

    for pair in zip(expected_classes,predicted_classes):
        table[pair[0]][pair[1]] = table[pair[0]][pair[1]] + 1

    recall = (table[True][True] * 1.0 + eps) / (table[True][True] + table[True][False] + eps)
    precision = (table[True][True] * 1.0 + eps) / (table[True][True] + table[False][True] + eps)

    F1_score = 2.0 * recall * precision / (recall + precision)
    
    print("Consufusion matrix is : {}".format(table))
    
    return F1_score 

def accuracy(expected_classes, predicted_classes):
   return sum([1 for pair in zip(expected_classes, predicted_classes) if pair[0] == pair[1]]) * 1.0 / len(expected_classes)

seed = 1337


from sklearn.model_selection import train_test_split
def get_data(df, clean_train=True):
    """
        clean_train=True returns a train sample that only contains clean samples.
        Otherwise, it will return a subset of each class in train and test (10% outlier)
    """
    
    clean = df[df.Class == 0].copy().reset_index(drop=True)
    fraud = df[df.Class == 1].copy().reset_index(drop=True)
    print(f'Clean Samples: {len(clean)}, Fraud Samples: {len(fraud)}')

    if clean_train:
        train, test_clean = train_test_split(clean, test_size=len(fraud), random_state=seed)
        print(f'Train Samples: {len(train)}')

        test = pd.concat([test_clean, fraud]).reset_index(drop=True)

        print(f'Test Samples: {len(test)}')

        # shuffle the test data
        test.sample(frac=1, random_state=seed).reset_index(drop=True)
        
        train_X, train_y = train.loc[:, ~train.columns.isin(['Class'])], train.loc[:, train.columns.isin(['Class'])]
        test_X, test_y = test.loc[:, ~test.columns.isin(['Class'])], test.loc[:, test.columns.isin(['Class'])]
    else:
        clean_train, clean_test = train_test_split(clean, test_size=int(len(fraud)+(len(fraud)*0.9)), random_state=seed)
        fraud_train, fraud_test = train_test_split(fraud, test_size=int(len(fraud)*0.1), random_state=seed)
        
        train_samples = pd.concat([clean_train, fraud_train]).reset_index(drop=True)
        test_samples = pd.concat([clean_test, fraud_test]).reset_index(drop=True)
        
        # shuffle
        train_samples.sample(frac=1, random_state=seed).reset_index(drop=True)
        
        print(f'Train Samples: {len(train_samples)}')
        test_samples.sample(frac=1, random_state=seed).reset_index(drop=True)
        
        print(f'Test Samples: {len(test_samples)}')
        train_X, train_y = train_samples.loc[:, ~train_samples.columns.isin(['Class'])], train_samples.loc[:, train_samples.columns.isin(['Class'])]
        test_X, test_y = test_samples.loc[:, ~test_samples.columns.isin(['Class'])], test_samples.loc[:, test_samples.columns.isin(['Class'])]
    
    return train_X, train_y, test_X, test_y

if __name__ == '__main__':  
    TREE_COUNT = 100
    SAMPLING_SIZE = 1024
    FEATURE_SAMPLE_SIZE = 15
         
    df = pd.read_csv("creditcard.csv", header=0, sep = ',')
    
    def flatten(t):
        return [item for sublist in t for item in sublist]
    
    train_X, train_y, test_X, test_y = get_data(df)
    test_y = [item for sublist in test_y.values.tolist() for item in sublist]
    
    isolation_forest = IForest(TREE_COUNT,SAMPLING_SIZE,FEATURE_SAMPLE_SIZE)
    isolation_forest.train(train_X)

    posLenLst=[]
    negLenLst=[]
    
    print(test_y[:10])
    for i in [0.2,0.3,0.4,0.5,0.53, 0.55,0.58,0.6, 0.62, 0.64, 0.67,0.7,0.8]:
        predictions = isolation_forest.predict_anomalies(test_X,i)
        print(predictions[:10])
        print("AUC score with anomality_threshold {}: {}".format(i,metrics.roc_auc_score(test_y, predictions)))

        print("F1 Score :{}".format(f1_score(test_y, predictions)))

        print("Accuracy :{}".format(accuracy(test_y, predictions)))
    
