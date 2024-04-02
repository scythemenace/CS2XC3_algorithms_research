
import csv
from math import radians, cos, sin, sqrt, atan2
import time
import matplotlib.pyplot as plt


class PriorityQueue:
    def __init__(self):
        self.elements = []

    def is_empty(self):
        return not self.elements

    def put(self, item, priority):
        self.elements.append((priority, item))
        self.elements.sort(reverse=True)

    def get(self):
        return self.elements.pop()[1]  

def A_Star(graph, source, destination, heuristic):
    open_list = PriorityQueue()
    open_list.put(source, 0 + heuristic[source])
    predecessors = {source: None}
    costs = {source: 0}  # Cost from start to node

    while not open_list.is_empty():
        current = open_list.get()

        if current == destination:
            break

        for neighbor, weight in graph[current].items():
            new_cost = costs[current] + weight
            if neighbor not in costs or new_cost < costs[neighbor]:
                costs[neighbor] = new_cost
                priority = new_cost + heuristic[neighbor]
                open_list.put(neighbor, priority)
                predecessors[neighbor] = current

    return predecessors, reconstruct_path(predecessors, source, destination)

def reconstruct_path(predecessors, start, end):
    path = []
    while end is not None:
        path.append(end)
        end = predecessors[end]
    return path[::-1] 


# Example usage
graph = {
    0: {1: 1, 2: 4},
    1: {2: 2, 3: 5},
    2: {3: 1},
    3: {}
}
heuristic = {
    0: 7,
    1: 6,
    2: 2,
    3: 0
}
source = 0
destination = 3

predecessors, path = A_Star(graph, source, destination, heuristic)
print("Predecessors:", predecessors)
print("Path:", path)