#!/usr/bin/env python3

from typing import Callable, List
from queue import Queue
import copy

class Constraint:
    def __init__(self,varA :int,varB : int, rule:Callable[[int,int],bool]) -> None:
        self.varA = varA
        self.varB = varB
        self.rule = rule

def Revise(domainA :List[int], domainB :List[int], constraint :Callable[[int,int],bool]) -> bool:
    revised = False
    copyDomain = copy.copy(domainA)

    for valueA in copyDomain:
        result = any([constraint(valueA,valueB) for valueB in domainB])
        if not result:
            domainA.remove(valueA)
            revised = True
    
    return revised


def AC3(binaryConstraints: List[Constraint], domains: List[List[int]]) -> bool:
    graph = {}

    arcs = Queue()
    for constraint in binaryConstraints:
        if not constraint.varA in graph:
            graph[constraint.varA] = []

        graph[constraint.varA].append((constraint.varB,constraint.rule))
        arcs.put(constraint)
    
    while not arcs.empty():
        arc = arcs.get()
        modified = Revise(domains[arc.varA], domains[arc.varB], arc.rule)

        if not modified:
            continue
        if len(domains[arc.varA]) == 0:
            return False
        for constraint in graph[arc.varA]:
            arcs.put(Constraint(constraint[0],arc.varA,constraint[1]))

    return True
            

if __name__ == "__main__":
    print("Arc consistency algorithm")

    rule1 = lambda x,y : x**2 == y
    rule2 = lambda y,x : rule1(x,y)

    constraints = []
    constraints.append(Constraint(0,1,rule1))
    constraints.append(Constraint(1,0,rule2))


    domains = [[0,1,2,3,4,5,6,7,8,9],[0,1,2,3,4,5,6,7,8,9]]
    succeeded = AC3(constraints,domains)
    if not succeeded:
        print("Problem with the algorithm")
        exit(1)

    print("Domain for x is {}".format(domains[0]))
    print("Domain for y is {}".format(domains[1]))