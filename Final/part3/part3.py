
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

    # Exclude the source node from the predecessors dictionary before returning
    filtered_predecessors = {key: value for key, value in predecessors.items() if key != source}
    
    path = reconstruct_path(predecessors, source, destination)
    
    # If no path is found, return an empty list for the path and None for the cost.
    if not path:  
        return filtered_predecessors, path, None
    
    return filtered_predecessors, path

def reconstruct_path(predecessors, start, end):
    
    if end not in predecessors:
        return []  # Or return "Path not found" or similar message
    
    path = []
    while end is not None:
        path.append(end)
        end = predecessors.get(end)  # Use .get() to avoid KeyError if end is not in predecessors
    path.reverse()
    return path # Return the path and a flag indicating success



graph = {0: {1: 1}, 1: {2: 1}, 2: {}}
source = 0
destination = 2
heuristic = {0: 2, 1: 1, 2: 0}
expected = ([0, 1, 2], 2)
result = A_Star(graph, source, destination, heuristic)
print(result)

graph = {0: {1: 1, 2: 5}, 1: {3: 1}, 2: {3: 1}, 3: {}}
source = 0
destination = 3
heuristic = {0: 2, 1: 1, 2: 1, 3: 0}
expected = ([0, 1, 3], 2)
result = A_Star(graph, source, destination, heuristic)
print(result)

graph = {0: {1: 1}, 1: {2: 1}, 2: {1: 1, 3: 1}, 3: {}}
source = 0
heuristic = {0: 3, 1: 2, 2: 1, 3: 0}
expected = ([0, 1, 2, 3], 3)
result = A_Star(graph, source, destination, heuristic)
print(result)

graph = {0: {1: 1, 2: 2, 3: 3}, 1: {}, 2: {}, 3: {}}
source = 0
destination = 3
heuristic = {0: 1, 1: 2, 2: 1, 3: 0}
expected = ([0, 3], 3)
result = A_Star(graph, source, destination, heuristic)
print(result)

graph = {0: {1: 1}, 1: {}, 2: {3: 1}, 3: {}}
source = 0
destination = 3
heuristic = {0: 100, 1: 100, 2: 1, 3: 0}
expected = (None, float('inf'))
result = A_Star(graph, source, destination, heuristic)
print(result)

graph = {0: {1: 2, 2: 4}, 1: {2: 1, 3: 7}, 2: {3: 3}, 3: {}}
source = 0
destination = 3
heuristic = {0: 5, 1: 4, 2: 2, 3: 0}
expected = ([0, 1, 2, 3], 6)
result = A_Star(graph, source, destination, heuristic)
print(result)

graph = {0: {1: 70, 2: 150}, 1: {3: 100}, 2: {3: 80}, 3: {}}
source = 0
destination = 3
heuristic = {0: 120, 1: 100, 2: 60, 3: 0}
expected = ([0, 2, 3], 230)
result = A_Star(graph, source, destination, heuristic)
print(result)

graph = {0: {1: 60, 2: 120}, 1: {3: 90}, 2: {3: 70}, 3: {}}
source = 0
destination = 3
heuristic = {0: 110, 1: 80, 2: 50, 3: 0}
expected = ([0, 2, 3], 190)
result = A_Star(graph, source, destination, heuristic)
print(result)

graph = {0: {1: 80, 2: 140}, 1: {3: 110}, 2: {3: 100}, 3: {}}
source = 0
destination = 3
heuristic = {0: 130, 1: 90, 2: 60, 3: 0}
expected = ([0, 2, 3], 240)
result = A_Star(graph, source, destination, heuristic)
print(result)

graph = {0: {1: 50, 2: 100}, 1: {3: 80}, 2: {4: 60}, 3: {}, 4: {3: 40}}
source = 0
destination = 3
heuristic = {0: 100, 1: 70, 2: 40, 3: 0, 4: 20}
expected = ([0, 2, 4, 3], 200)
result = A_Star(graph, source, destination, heuristic)
print(result)

graph = {0: {1: 75, 2: 115}, 1: {3: 95}, 2: {3: 85}, 3: {}}
source = 0
destination = 3
heuristic = {0: 125, 1: 85, 2: 55, 3: 0}
expected = ([0, 2, 3], 200)
result = A_Star(graph, source, destination, heuristic)
print(result)

graph = {0: {1: 65, 2: 130}, 1: {2: 20, 3: 100}, 2: {3: 75}, 3: {}}
source = 0
destination = 3
heuristic = {0: 120, 1: 100, 2: 55, 3: 0}
expected = ([0, 2, 3], 205)
result = A_Star(graph, source, destination, heuristic)
print(result)


graph = {0: {1: 55, 2: 105}, 1: {3: 85}, 2: {1: 30, 3: 65}, 3: {}}
source = 0
destination = 3
heuristic = {0: 115, 1: 75, 2: 45, 3: 0}
expected = ([0, 2, 3], 170)
result = A_Star(graph, source, destination, heuristic)
print(result)

graph = {0: {1: 45, 2: 95}, 1: {3: 75}, 2: {3: 55}, 3: {}}
source = 0
destination = 3
heuristic = {0: 105, 1: 65, 2: 35, 3: 0}
expected = ([0, 2, 3], 150)
result = A_Star(graph, source, destination, heuristic)
print(result)

graph = {0: {1: 85, 2: 125}, 1: {2: 25, 3: 105}, 2: {3: 95}, 3: {}}
source = 0
destination = 3
heuristic = {0: 135, 1: 95, 2: 65, 3: 0}
expected = ([0, 2, 3], 220)
result = A_Star(graph, source, destination, heuristic)
print(result)

graph = {0: {1: 70, 2: 110}, 1: {3: 90}, 2: {1: 35, 3: 80}, 3: {}}
source = 0
destination = 3
heuristic = {0: 120, 1: 80, 2: 50, 3: 0}
expected = ([0, 2, 3], 190)
result = A_Star(graph, source, destination, heuristic)
print(result)