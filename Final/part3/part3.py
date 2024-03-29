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
    open_set = PriorityQueue()
    open_set.put(source, 0 + heuristic[source])
    predecessors = {source: None}
    actual_costs = {source: 0}

    while not open_set.is_empty():
        current_node = open_set.get()

        if current_node == destination:
            return predecessors, reconstruct_path(predecessors, source, destination)

        for neighbor, weight in graph[current_node].items():
            tentative_cost = actual_costs[current_node] + weight
            if neighbor not in actual_costs or tentative_cost < actual_costs[neighbor]:
                actual_costs[neighbor] = tentative_cost
                total_cost = tentative_cost + heuristic.get(neighbor, float('inf'))
                open_set.put(neighbor, total_cost)
                predecessors[neighbor] = current_node

    return predecessors, []

def reconstruct_path(predecessors, start, end):
    path = []
    while end is not None:
        path.append(end)
        end = predecessors[end]
    return path[::-1]  # Reverse path



# Define the graph as a dictionary. Each key is a node, and each value is another dictionary
# representing the neighboring nodes and the cost to reach them.
graph = {
    'A': {'B': 1, 'C': 3},
    'B': {'A': 1, 'D': 7, 'E': 5},
    'C': {'A': 3, 'F': 10},
    'D': {'B': 7, 'E': 2},
    'E': {'B': 5, 'D': 2, 'F': 1},
    'F': {'C': 10, 'E': 1}
}

# The heuristic function. In this example, it's just a dictionary with arbitrary values.
# In practice, this would calculate the estimated cost from each node to the destination.
heuristic = {
    'A': 10,
    'B': 8,
    'C': 5,
    'D': 7,
    'E': 3,
    'F': 0,  # Assuming 'F' is our destination
}

# Define the A* function and the PriorityQueue class here
# (Copy the A_Star function and the PriorityQueue class from the previous response)

# Assuming 'A' is the source and 'F' is the destination
source = 'A'
destination = 'F'

# Run the A* algorithm
predecessors, path = A_Star(graph, source, destination, heuristic)

# Display the results
print("Predecessors:", predecessors)
print("Shortest path from", source, "to", destination, ":", path)
