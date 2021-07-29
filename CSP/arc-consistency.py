
from typing import Callable, List
from queue import Queue

class Constraint:
    def __init__(self,varA :int,varB : int, rule:Callable[[int,int],bool]) -> None:
        self.varA = varA
        self.varB = varB
        self.rule = rule

def Revise(domainA :List[int], domainB :List[int], constraint :Callable[[int,int],bool]) -> bool:
    revised = False
    for valueA in domainA:
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
        if not constraint.varB in graph:
            graph[constraint.varB] = []

        graph[constraint.varA].append((constraint.B,constraint.rule))
        graph[constraint.varB].append((constraint.A,constraint.rule))

        arcs.put(constraint)
        arcs.put(Constraint(constraint.varB,constraint.varA,constraint.rule))
    
    while not arcs.empty():
        arc = arcs.get()
        modified = Revise(domains[arc.varA], domains[arc.varB], arc.constraint)
        if not modified:
            continue
        if len(domains[arc.varA]):
            return False
        for constraint in graph[constraint.varA]:
            if constraint.varB == arc.varB:
                continue
            arcs.put(Constraint(constraint.varB,arc.varA,constraint.rule))

    return True
            

if __name__ == "__main__":
    print("Arc consistency algorithm")