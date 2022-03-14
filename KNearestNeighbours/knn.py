#!/usr/bin/env python3

import pandas as pd
import heapq
import math

class Node:
    def __init__(self, record, separator_dim_idx, area, left_child, right_child):
        self.target_value = record[-1]
        self.separator_dim_idx = separator_dim_idx
        self.separator_value = record[separator_dim_idx]
        self.left_child = left_child
        self.right_child = right_child
        self.coords = record[:-1].tolist()
        self.area = area

class KDTree:
    def __init__(self, df):
        self.root = self._build_tree(df)
    
    def _build_tree(self, df):
        area = [[float("-inf"),float("inf")] for _ in range(1, df.shape[1])]
        return self._build_tree_rec(df, 0,area)
    
    def _build_tree_rec(self, df, depth, area, median_candidate_count = 10):
        if df.shape[0] == 0:
            return None
        
        dim_idx = depth % (df.shape[1]-1)
        
        median_candidate_count = min(median_candidate_count, df.shape[0])
        median_candidates = df.sample(n = median_candidate_count).sort_values(by=df.columns[dim_idx]) # could be probably done efficiently with insertion sort or quick select cuz low amount of numbers
        winner = median_candidates.iloc[median_candidate_count // 2,:]
        
        separator_value = winner[dim_idx]
        
        left_area = area.copy()
        left_area[dim_idx] = [left_area[dim_idx][0],separator_value]
        left_child = self._build_tree_rec(df[(df.iloc[:,dim_idx] <= separator_value) & (df.index != winner.name)], depth+1,left_area)
        
        right_area = area.copy()
        right_area[dim_idx] = [separator_value,right_area[dim_idx][1]]
        right_child = self._build_tree_rec(df[df.iloc[:,dim_idx] > separator_value], depth+1,right_area)
        
        return Node(winner, dim_idx,area,left_child, right_child)
    
    def nearest_neighbours(self, target_record, k=6):
        heap = self._create_dummy_max_heap(k)
        self._nearest_neighbours_rec(target_record, self.root, heap)
        
        return [element[1] + [element[2]] for element in heap]
    
    def _nearest_neighbours_rec(self, target_record, curr_node, heap):
        if curr_node == None:
            return
        
        distance_to_current_node = distance_between_records(target_record, curr_node.coords)
        if self._get_max_distance(heap) > distance_to_current_node:
            heapq.heappop(heap)
            self._push_to_heap(heap,(distance_to_current_node, curr_node.coords, curr_node.target_value))
        
        first_child_to_traverse, second_child_to_traverse = curr_node.left_child, curr_node.right_child
        if curr_node.separator_value < target_record[curr_node.separator_dim_idx]: # the order of traversal could lead to branch skips
            first_child_to_traverse, second_child_to_traverse = second_child_to_traverse, first_child_to_traverse
        
        kth_point_distance = self._get_max_distance(heap)
        if first_child_to_traverse != None and kth_point_distance > distance_to_area(target_record, first_child_to_traverse.area):
            self._nearest_neighbours_rec(target_record, first_child_to_traverse, heap)
        
        kth_point_distance = self._get_max_distance(heap)
        if second_child_to_traverse != None and kth_point_distance > distance_to_area(target_record, second_child_to_traverse.area):
            self._nearest_neighbours_rec(target_record, second_child_to_traverse, heap)
    
    def _get_max_distance(self,heap):
        return abs(heap[0][0])  
    
    def _push_to_heap(self, heap, element):
        element = list(element)
        element[0] = -element[0] # make the distance negative
        heapq.heappush(heap,tuple(element))
        
    def _create_dummy_max_heap(self, n):
        return  [(float('-inf'),)] * n # the comma is needed otherwise the tuple will be converted to single value automatically

def insertion_sort(array): # best usage for arrays around 64 items cuz of caching
    for i in range(1,len(array)):
        j = i - 1
        temp = array[i]
        while j >= 0 and array[j] >= temp:
            array[i] = array[j]
            j = j - 1
        array[j+1] = temp     

def distance_to_area(target_record, area):
    distance = 0.0 
    for dim in range(0,len(area)):
        if target_record[dim] < area[dim][0]:
            distance += (target_record[dim] -  area[dim][0]) ** 2
        elif target_record[dim] > area[dim][1]:
            distance += (target_record[dim] -  area[dim][1]) ** 2
    return distance ** (1/2)
    

def distance_between_records(point_a, point_b):
    return sum([(point_a[i] - point_b[i])**2 for i in range(0, len(point_a))]) ** (1/2)

def get_most_frequent_element(array):
    bestElement = None
    bestCnt = 0
    counter = {}
    for element in array:
        if element not in counter:
            counter[element] = 0
        counter[element] += 1
        
        if counter[element] > bestCnt:
            bestElement = element
            bestCnt = counter[element]
    
    return bestElement
    

if __name__ == "__main__":
    df = pd.read_csv("iris_csv.csv",header=0)
    
    print("Dataset size {} with number of features {}".format(df.shape[0],df.shape[1]))
    print("Class distribution:\n{}".format(df["class"].value_counts()))
    
    
    df = df.sample(frac=1).reset_index(drop=True)
    
    training_dataset_size = math.floor(df.shape[0] * 0.7)
    training_df = df.iloc[:training_dataset_size,:]
    
    test_df = df.iloc[training_dataset_size:,:]
    
    
    tree = KDTree(training_df)
    
    predicted_right = 0
    
    for _,test in test_df.iterrows():
        neighbours = tree.nearest_neighbours(test[:-1],k=10)
        prediction = get_most_frequent_element([neighbour[-1] for neighbour in neighbours])
        
        predicted_right = predicted_right + (1 if test[-1] == prediction else 0)
    
    print("Accuracy :{}".format(predicted_right * 1.0 /test_df.shape[0]))
    
    