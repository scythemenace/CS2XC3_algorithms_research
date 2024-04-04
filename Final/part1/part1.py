import time
import random

def dijkstra(graph, source, k):
    dist = [float('inf')] * len(graph)
    dist[source] = 0
    path = {vertex: [] for vertex in range(len(graph))}
    path[source] = [source]
    unvisited = set(range(len(graph)))
    relaxations = {vertex: 0 for vertex in range(len(graph))}

    while unvisited:
        current_node = None
        current_distance = float('inf')
        for node in unvisited:
            if dist[node] < current_distance:
                current_node = node
                current_distance = dist[node]

        if current_node is None:
            break

        unvisited.remove(current_node)

        if relaxations[current_node] >= k:
            continue

        for neighbor in range(len(graph)):
            weight = graph[current_node][neighbor] 
            if weight > 0:
                new_distance = current_distance + weight
                if new_distance < dist[neighbor]:
                    dist[neighbor] = new_distance
                    path[neighbor] = path[current_node] + [neighbor]
                    relaxations[neighbor] += 1

    return dist, path

def bellman_ford(graph, source, k):
    dist = [float('inf')] * len(graph)
    dist[source] = 0
    path = {vertex: [] for vertex in range(len(graph))}
    path[source] = [source]
    relaxations = {vertex: 0 for vertex in range(len(graph))}

    for _ in range(len(graph) - 1):  
        for node in range(len(graph)):
            if relaxations[node] < k:  
                for neighbor in range(len(graph)):
                    weight = graph[node][neighbor]
                    if weight > 0: 
                        new_distance = dist[node] + weight
                        if new_distance < dist[neighbor]:
                            dist[neighbor] = new_distance
                            path[neighbor] = path[node] + [neighbor]
                            relaxations[neighbor] += 1

    return dist, path


def generate_random_graph(n, density):
    graph = [[0] * n for _ in range(n)]
    for i in range(n):
        for j in range(n):
            if i != j and random.random() < density:
                graph[i][j] = random.randint(1, 10)  # Random weights between 1 and 10
    return graph

def run_experiment(algorithm, graph_size, density, k):
    graph = generate_random_graph(graph_size, density)
    source = 0  # Assuming source is always 0 for simplicity

    start_time = time.time()
    dist, path = algorithm(graph, source, k)
    end_time = time.time()
    
    execution_time = end_time - start_time

    return execution_time

# Parameters for the experiment
graph_size = 100  # Adjust as needed
density = 0.3  # Adjust as needed
k = 3  # Limit of relaxations

# Run experiment for Dijkstra's algorithm
dijkstra_time = run_experiment(dijkstra, graph_size, density, k)
print("Dijkstra's Algorithm Execution Time:", dijkstra_time)

# Run experiment for Bellman-Ford algorithm
bellman_ford_time = run_experiment(bellman_ford, graph_size, density, k)
print("Bellman-Ford Algorithm Execution Time:", bellman_ford_time)