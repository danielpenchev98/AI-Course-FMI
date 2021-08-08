#!/usr/bin/env python3

import numpy as np
import pandas as pd
import cvxopt
import math
import functools
from typing import Callable

class Model:
    def __init__(self,sv_multipliers, support_vectors, support_vectors_y, b):
        self.sv_multipliers = sv_multipliers
        self.support_vectors = support_vectors
        self.support_vectors_y = support_vectors_y
        self.b = b

def get_training_dataset() -> pd.DataFrame:
    dataset = np.array([[8, 7, 1], [4, 10, 1], [9, 7, 1], [7, 10, 1],
                   [9, 6, 1], [4, 8, 1], [10, 10, 1], [2, 7, -1], [8, 3, -1], [7, 5, -1], [4, 4, -1],
                   [4, 6, -1], [1, 3, -1], [2, 5, -1]])

    return pd.DataFrame(data=dataset,   
                        columns=["X1", "X2", "Y"])

def get_test_dataset() -> pd.DataFrame:
    dataset = np.array([[2, 9, 1], [1, 10, 1], [1, 11, 1], [3, 9, 1], [11, 5, 1],
                   [10, 6, 1], [10, 11, 1], [7, 8, 1], [8, 8, 1], [4, 11, 1],
                   [9, 9, 1], [7, 7, 1], [11, 7, 1], [5, 8, 1], [6, 10, 1], 
                   [11, 2, -1], [11, 3, -1], [1, 7, -1], [5, 5, -1], [6, 4, -1],
                   [9, 4, -1], [2, 6, -1], [9, 3, -1], [7, 4, -1], [7, 2, -1], [4, 5, -1],
                   [3, 6, -1], [1, 6, -1], [2, 3, -1], [1, 1, -1], [4, 2, -1], [4, 3, -1]])

    return pd.DataFrame(data=dataset,   
                        columns=["X1", "X2", "Y"])


def solve_svm_constraint_optimization_problem(X: np.ndarray, y: np.ndarray, kernel: Callable[[np.ndarray, np.ndarray], float]) -> np.ndarray:
    m = X.shape[0]
    GramMatrix = np.array([kernel(X[i],X[j]) * 1.0 for i in range(m) for j in range(m)]).reshape((m,m))

    P = cvxopt.matrix(np.outer(y,y) * GramMatrix) # optimization function
    q = cvxopt.matrix(-1 * np.ones(m))

    # Equality constraints ∑ label(i) . αi = 0
    A = cvxopt.matrix(y*1.0, (1, m)) # vector row
    b = cvxopt.matrix(0.0)
    
    # Inequality constraints -αi ≤ 0
    G = cvxopt.matrix(np.diag(-1 * np.ones(m)))
    h = cvxopt.matrix(np.zeros(m))
    
    # Solve the problem
    solution = cvxopt.solvers.qp(P, q, G, h, A, b)
    
    # Lagrange multipliers
    return np.ravel(solution['x'])

def create_svm_model(dataset: pd.DataFrame, kernel: Callable[[np.ndarray, np.ndarray], float]) -> Model:
    X, y = dataset.iloc[:,:-1].to_numpy(), dataset.iloc[:,-1].to_numpy()

    multipliers = solve_svm_constraint_optimization_problem(X, y, kernel)
    
    # Support vectors have positive multipliers.
    has_positive_multiplier = multipliers > 1e-7

    sv_multipliers = multipliers[has_positive_multiplier]
    support_vectors = X[has_positive_multiplier]
    support_vectors_y = y[has_positive_multiplier]

    w = compute_w(sv_multipliers,support_vectors,support_vectors_y)
    b = compute_b(w,support_vectors,support_vectors_y)
    return Model(sv_multipliers,support_vectors,support_vectors_y,b)


def compute_w(multipliers: np.ndarray, support_vectors: np.ndarray, support_vectors_y: np.ndarray) -> float:
    return sum(multipliers[i] * support_vectors[i] * support_vectors_y[i] for i in range(len(support_vectors_y)))

def compute_b(w: np.ndarray, support_vectors: np.ndarray, support_vectors_y: np.ndarray) -> float:
    return np.mean(support_vectors_y -  np.dot(support_vectors,w))

def hypothesis(multipliers: np.ndarray,support_vectors: np.ndarray,support_vectors_y: np.ndarray,b: float, kernel: Callable[[np.ndarray, np.ndarray], float], element: np.ndarray) -> int:
    partial_result = np.array([multipliers[i] * support_vectors_y[i] * kernel(support_vectors[i],element) for i in range(len(support_vectors))])
    return np.sign(sum(partial_result)+b)

def linear_kernel(x_i: np.ndarray, x_j: np.ndarray) -> float:
    return np.dot(x_i,x_j)

def polynomial_kernel(x_i: np.ndarray, x_j: np.ndarray, d: float, c=0.0) -> float:
    return (np.dot(x_i,x_j) +  c) ** d

def gaussian_kernel(x_i: np.ndarray,x_j: np.ndarray, sigma: float)->float:
    return math.exp(-np.linalg.norm(x_i - x_j) * 1.0/ sigma)

def validate_model(dataset: pd.DataFrame, hypothesis_func: functools.partial) -> None:
    X, y = dataset.iloc[:,:-1].to_numpy(), dataset.iloc[:,-1].to_numpy()
    correct = 0
    for i in range(X.shape[0]):
        prediction = hypothesis_func(X[i])
        if prediction == y[i]:
            correct+=1
    
    print("Model accuracy :{}".format(correct * 1.0/X.shape[0]))

if __name__ == "__main__":
    model = create_svm_model(get_training_dataset(),linear_kernel)
    hypothesis_func = functools.partial(hypothesis,model.sv_multipliers,model.support_vectors,model.support_vectors_y,model.b, linear_kernel)
    validate_model(get_test_dataset(),hypothesis_func)


