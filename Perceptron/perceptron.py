#!/usr/bin/env python3

import numpy as np
import pandas as pd
import random

def hypothesis(element, weight):
    return np.sign(np.dot(element,weight))

def get_misclasified_examples(elements: pd.DataFrame, labels: np.array, weights: np.array):
    predictions = np.apply_along_axis(hypothesis, 1, elements, weights)
    return [i for i in range(elements.shape[0]) if predictions[i] != labels[i]]

def get_random_example(choices, elements, labels):
    winner = random.randint(0,len(choices)-1)
    idx = choices[winner]
    return elements.iloc[idx,:],  labels.iloc[idx]

def create_model(elements, labels):
    elements.insert(loc=0, column='X0', value=1, allow_duplicates=True)
    weights = np.random.rand(elements.shape[1]) 
    misclasified = get_misclasified_examples(elements, labels, weights)

    while len(misclasified) > 0:
        random_example, label = get_random_example(misclasified, elements, labels)
        weights = weights + label * random_example
        misclasified = get_misclasified_examples(elements, labels, weights)

    return weights

def get_training_examples():
    dataset = np.array([[8, 7, 1], [4, 10, 1], [9, 7, 1], [7, 10, 1],
                   [9, 6, 1], [4, 8, 1], [10, 10, 1], [2, 7, -1], [8, 3, -1], [7, 5, -1], [4, 4, -1],
                   [4, 6, -1], [1, 3, -1], [2, 5, -1]])

    return pd.DataFrame(data=dataset,   
                        columns=["X1", "X2", "Y"])

def get_test_examples():
    dataset = np.array([[2, 9, 1], [1, 10, 1], [1, 11, 1], [3, 9, 1], [11, 5, 1],
                   [10, 6, 1], [10, 11, 1], [7, 8, 1], [8, 8, 1], [4, 11, 1],
                   [9, 9, 1], [7, 7, 1], [11, 7, 1], [5, 8, 1], [6, 10, 1], 
                   [11, 2, -1], [11, 3, -1], [1, 7, -1], [5, 5, -1], [6, 4, -1],
                   [9, 4, -1], [2, 6, -1], [9, 3, -1], [7, 4, -1], [7, 2, -1], [4, 5, -1],
                   [3, 6, -1], [1, 6, -1], [2, 3, -1], [1, 1, -1], [4, 2, -1], [4, 3, -1]])

    return pd.DataFrame(data=dataset,   
                        columns=["X1", "X2", "Y"])

def verify_model(elements, labels, model):
    elements.insert(loc=0, column='X0', value=1, allow_duplicates=True)
    correct = 0
    for i in range(elements.shape[0]):
        prediction = hypothesis(elements.iloc[i,:], model)
        if prediction == labels[i]:
            correct+=1
    
    print("Accuracy :{}".format(correct/elements.shape[0]))

if __name__ == "__main__":
    # last column is the label
    training_dataset = get_training_examples()
    elements, labels = training_dataset.iloc[:,:-1], training_dataset.iloc[:,-1]

    print("Starting the creation of the model")
    model = create_model(elements, labels)
    print("Creation of the model succeeded")

    test_dataset = get_test_examples()
    elements, labels = test_dataset.iloc[:,:-1], test_dataset.iloc[:,-1]

    verify_model(elements, labels, model)

    