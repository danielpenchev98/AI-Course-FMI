#!/usr/bin/env python3

from enum import Enum
from typing import Callable, List
import copy
from typing import Tuple

class Color(Enum):
    RED = 0
    GREEN = 1
    BLUE = 2
    DEFAULT = 3


def notEqual(a: Color, b: Color) -> bool:
    return a != b

# removes a particular value from the domain of the unassigned neightbours of the currently examined node
def forward_chaining(assignedColor: Color,unassignedNeighbours:List[int],domains:List[List[Color]], constraint: Callable[[Color,Color],bool]) -> bool:
    for neighbour in unassignedNeighbours:
        for color in domains[neighbour]:
            if not constraint(assignedColor,color):
                 domains[neighbour].remove(assignedColor)

        if  len(domains[neighbour]) == 0:
            return False

    return True

# picks the next variable according the most constrained variable policy (minimum remaining values)
def pick_variable(candidates:List[int],domains:List[List[Color]]) -> int:
    bestCandidate = candidates[0]
    minDomainLength = len(domains[bestCandidate])
    for candidate in candidates:
        if minDomainLength > len(domains[candidate]):
            bestCandidate = candidate
            minDomainLength = len(domains[candidate])
    
    return bestCandidate

def least_constraining_value(valueOptions:List[Color],neighbours:List[int],domains:List[List[Color]], constraint :Callable[[Color,Color],bool]) -> Color:
    bestValue = valueOptions[0]
    maxCnt = 0;
    for option in valueOptions:
        currCnt = 0
        for neighbour in neighbours:
            currCnt += sum(1 for val in domains[neighbour] if constraint(option,val))
        
        if currCnt > maxCnt:
            bestValue = option
            maxCnt = currCnt
    
    return bestValue

# finds a solution to the map colouring problem using backtracking 
def mapColoringRec(graph:List[List[int]], domains:List[List[Color]], solution: List[Color]) -> bool:
    unassignedVariables = [i for i in range(0,len(solution)) if solution[i] == Color.DEFAULT]
    if len(unassignedVariables) == 0:
        return True

    currNode = pick_variable(unassignedVariables,domains)
    unassignedNeightbours = [i for i in graph[currNode] if solution[i] == Color.DEFAULT]
    
    for color in domains[currNode]:
        copyDomains = copy.deepcopy(domains)

        solution[currNode] = color
        succeded = forward_chaining(color,unassignedNeightbours,copyDomains, notEqual)
        if succeded and mapColoringRec(graph,copyDomains,solution):
            return True
            
        solution[currNode] = Color.DEFAULT
      
    
    return False


def mapColoringSolve(graph:List[List[int]], domains:List[List[Color]]) -> Tuple[List[Color],bool]:
    solution = [Color.DEFAULT for i in range(0,len(graph))]
    found = mapColoringRec(graph,domains,solution)
    return (solution,found)

if __name__ == "__main__":
    vertexNum = 7
    graph = [[] for _ in range(0,vertexNum)]
    
    #    Map of Australia
    #    WA: 0 GREEN
    #    NT: 1 RED
    #    Q: 2 GREEN
    #    NSW: 3 RED
    #    V: 4 GREEN
    #    SA: 5 BLUE
    #    T: 6 GREEN

    #WA
    graph[0].append(1)
    graph[0].append(5)

    #NT
    graph[1].append(0)
    graph[1].append(5)
    graph[1].append(2)

    #Q
    graph[2].append(1)
    graph[2].append(5)
    graph[2].append(3)

    #NSW
    graph[3].append(2)
    graph[3].append(5)
    graph[3].append(4)

    #V
    graph[4].append(5)
    graph[4].append(3)

    #SA
    graph[5].append(0)
    graph[5].append(1)
    graph[5].append(2)
    graph[5].append(3)
    graph[5].append(4)

    domains = [[Color.GREEN,Color.RED,Color.BLUE] for _ in range(0,vertexNum)]

    (solution, found) = mapColoringSolve(graph,domains)
    if not found:
        print("No possible way to color the map of Australia")
        exit(0)

    print("The solution is the following :")
    for i in range(0,len(graph)):
        print("For node {} the color is {}".format(i,solution[i].name))
       
