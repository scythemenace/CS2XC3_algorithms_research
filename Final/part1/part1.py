import time
import matplotlib.pyplot as plt
import random
import networkx as nx
import sys

def get_size(obj):
  size = sys.getsizeof(obj)
  if isinstance(obj, dict):
    size += sum(get_size(k) + get_size(v) for k, v in obj.items())
  if isinstance(obj, (list, set, tuple)): 
    size += sum(get_size(v) for v in obj)
  return size

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

def measure_space_complexity_dijkstra(graph, source, k):
    dist, path = dijkstra(graph, source, k)

    # Calculate space used by major data structures
    space_dist = get_size(dist)
    space_path = get_size(path)

    total_space = space_dist + space_path
    return total_space 

def measure_space_complexity_bellman_ford(graph, source, k):
    dist, path = bellman_ford(graph, source, k)

    # Calculate space used by major data structures
    space_dist = get_size(dist)
    space_path = get_size(path)

    total_space = space_dist + space_path
    return total_space 


def generate_random_graph(n, density):
    if density < 0 or density > 1:
        raise ValueError("Density must be in the range [0, 1]")
    graph = [[0] * n for _ in range(n)]
    for i in range(n):
        for j in range(i+1, n):
            if random.random() < density:
                graph[i][j] = random.randint(1, 10)
    return graph

def accuracy_calc(graph, source):
    dist = [float('inf')] * len(graph)
    dist[source] = 0
    queue = [source]
    visited = set()

    while queue:
        current_node = queue.pop(0)
        visited.add(current_node)

        for neighbor, weight in enumerate(graph[current_node]):
            if weight > 0 and neighbor not in visited:
                new_distance = dist[current_node] + weight
                if new_distance < dist[neighbor]:
                    dist[neighbor] = new_distance
                    queue.append(neighbor)

    return dist

def measure_accuracy(graph, source, algorithm, k=3):
    if algorithm == 'dijkstra':
        dist, _ = dijkstra(graph, source, k) 
    else: 
        dist, _ = bellman_ford(graph, source, k)

    true_distances = accuracy_calc(graph, source)
    correct_count = 0
    for i in range(len(graph)):
        if dist[i] == true_distances[i]:
            correct_count += 1

    accuracy = correct_count / len(graph)  # Percentage of correct distances
    return accuracy

def calc_time(algorithm, graph_size, density, k):
    graph = generate_random_graph(graph_size, density)
    source = 0 

    start_time = time.time()
    dist, path = algorithm(graph, source, k)
    end_time = time.time()
    
    execution_time = end_time - start_time

    return execution_time


"""We write three different experiments calculating time complexity
for different variables:- First graph size, second relaxation limit and third density."""

# Parameters for the first experiment:
densities = 0.5  # Fixed density
k = 3  # Fixed relaxation limit

# Experiment for different graph sizes:
graph_sizes = list(range(5, 101, 5))
dijkstra_times = []
bellman_ford_times = []

for graph_size in graph_sizes:
    dijkstra_time = calc_time(dijkstra, graph_size, densities, k)
    bellman_ford_time = calc_time(bellman_ford, graph_size, densities, k)
    dijkstra_times.append(dijkstra_time)
    bellman_ford_times.append(bellman_ford_time)

# Plotting the results
plt.plot(graph_sizes, dijkstra_times, label="Dijkstra's Algorithm")
plt.plot(graph_sizes, bellman_ford_times, label="Bellman-Ford Algorithm")
plt.xlabel('Graph Size')
plt.ylabel('Execution Time (seconds)')
plt.title('Execution Time Comparison for Different Graph Sizes')
plt.legend()
plt.grid(True)
plt.show()

# Parameters for the second experiment:
graph_size = 50  # Fixed graph size
densities = 0.5  # Fixed density

# Experiment for different relaxation limits (k):
ks = list(range(1, 16))
dijkstra_times = []
bellman_ford_times = []

for k in ks:
    dijkstra_time = calc_time(dijkstra, graph_size, densities, k)
    bellman_ford_time = calc_time(bellman_ford, graph_size, densities, k)
    dijkstra_times.append(dijkstra_time)
    bellman_ford_times.append(bellman_ford_time)

# Plotting the results
plt.plot(ks, dijkstra_times, label="Dijkstra's Algorithm")
plt.plot(ks, bellman_ford_times, label="Bellman-Ford Algorithm")
plt.xlabel('Relaxation Limit (k)')
plt.ylabel('Execution Time (seconds)')
plt.title('Execution Time Comparison for Different Relaxation Limits')
plt.legend()
plt.grid(True)
plt.show()

# Parameters for the third experiment:
graph_size = 50  # Fixed graph size
k = 3  # Fixed relaxation limit

# Experiment for different densities:
densities = [i / 10 for i in range(1, 10)]
dijkstra_times = []
bellman_ford_times = []

for density in densities:
    dijkstra_time = calc_time(dijkstra, graph_size, density, k)
    bellman_ford_time = calc_time(bellman_ford, graph_size, density, k)
    dijkstra_times.append(dijkstra_time)
    bellman_ford_times.append(bellman_ford_time)

# Plotting the results
plt.plot(densities, dijkstra_times, label="Dijkstra's Algorithm")
plt.plot(densities, bellman_ford_times, label="Bellman-Ford Algorithm")
plt.xlabel('Density')
plt.ylabel('Execution Time (seconds)')
plt.title('Execution Time Comparison for Different Densities')
plt.legend()
plt.grid(True)
plt.show()


"""We write three different experiments calculating space complexity
for different variables:- First graph size, second relaxation limit and third density."""

#For various graph sizes:-
sizes = range(5, 101, 5)
density = 0.5
relaxation_limit = 5

dijkstra_usage_s = []
bellman_ford_usage_s = []

for size in sizes:
    graph = generate_random_graph(size, density)
    dijkstra_usage_s.append(measure_space_complexity_dijkstra(graph, 0, relaxation_limit))
    bellman_ford_usage_s.append(measure_space_complexity_bellman_ford(graph, 0, relaxation_limit))

plt.plot(sizes, dijkstra_usage_s, label="Dijkstra")
plt.plot(sizes, bellman_ford_usage_s, label="Bellman-Ford")
plt.xlabel("Graph Size (Number of Vertices)")
plt.ylabel("Space Usage (Bytes)")
plt.title("Space Complexity vs. Graph Size")
plt.legend()
plt.show()


#For various relaxation limits:-
dijkstra_usage_rl = []
bellman_ford_usage_rl = []

graph_size = 50
density = 0.5

relaxation_limits = range(1, 16)
space_usage = []
for limit in relaxation_limits:
    graph = generate_random_graph(graph_size, density)
    dijkstra_usage_rl.append(measure_space_complexity_dijkstra(graph, 0, limit))
    bellman_ford_usage_rl.append(measure_space_complexity_bellman_ford(graph, 0, limit))

plt.plot(relaxation_limits, dijkstra_usage_rl, label="Dijkstra")
plt.plot(relaxation_limits, bellman_ford_usage_rl, label="Bellman-Ford")
plt.xlabel("Relaxation Limit (k)")
plt.ylabel("Space Usage (Bytes)")
plt.title("Space Complexity vs. Relaxation limit")
plt.legend()
plt.show()

#For various densities:-
dijkstra_usage_d = []
bellman_ford_usage_d = []

graph_size = 50
relaxation_limit = 5 

densities = [0.1 * i for i in range(1, 10)]  # 0.1 to 0.9
space_usage = []
for density in densities:
    graph = generate_random_graph(size, density)
    dijkstra_usage_d.append(measure_space_complexity_dijkstra(graph, 0, relaxation_limit))
    bellman_ford_usage_d.append(measure_space_complexity_bellman_ford(graph, 0, relaxation_limit))

plt.plot(densities, dijkstra_usage_d, label="Dijkstra")
plt.plot(densities, bellman_ford_usage_d, label="Bellman-Ford")
plt.xlabel("Densities")
plt.ylabel("Space Usage (Bytes)")
plt.title("Space Complexity vs. Densities")
plt.legend()
plt.show()

"""We write three different experiments calculating accuracy
for different variables:- First graph size, second relaxation limit and third density."""
#First experiment for different graph sizes:-
graph_size = 50
density = 0.5
sizes = range(5, 101, 5) 
density = 0.5  # Ensure density allows for valid paths 
relaxation_limit = 5 

dijkstra_accuracy_size = []
bellman_ford_accuracy_size = []

for size in sizes:
    graph = generate_random_graph(size, density)
    dijkstra_accuracy_size.append(measure_accuracy(graph, 0, 'dijkstra', relaxation_limit))
    bellman_ford_accuracy_size.append(measure_accuracy(graph, 0, 'bellman_ford', relaxation_limit))

plt.plot(sizes, dijkstra_accuracy_size, label="Dijkstra")
plt.plot(sizes, bellman_ford_accuracy_size, label="Bellman-Ford")
plt.xlabel("Sizes")
plt.ylabel("Accuracy")
plt.title("Accuracy vs. Sizes")
plt.legend()
plt.show()


#Second experiment for different relaxation limits:-
graph_size = 50 
density = 0.5
relaxation_limits = range(1, 16)

dijkstra_accuracy = []
bellman_ford_accuracy = []

for limits in relaxation_limits:
    graph = generate_random_graph(graph_size, density)
    dijkstra_accuracy.append(measure_accuracy(graph, 0, 'dijkstra', limits))
    bellman_ford_accuracy.append(measure_accuracy(graph, 0, 'bellman_ford', limits))

plt.plot(relaxation_limits, dijkstra_accuracy, label="Dijkstra")
plt.plot(relaxation_limits, bellman_ford_accuracy, label="Bellman-Ford")
plt.xlabel("Relaxation Limit (k)")
plt.ylabel("Accuracy")
plt.title("Accuracy vs. Relaxation Limit")
plt.legend()
plt.show()

#Third experiment for different densities:-
densities = [0.1 * i for i in range(1, 10)]  # Densities from 0.1 to 0.9
graph_size = 50  # Fixed graph size
relaxation_limit = 5 

dijkstra_accuracy = []
bellman_ford_accuracy = []

for density in densities:
    graph = generate_random_graph(graph_size, density)
    dijkstra_accuracy.append(measure_accuracy(graph, 0, 'dijkstra', relaxation_limit))
    bellman_ford_accuracy.append(measure_accuracy(graph, 0, 'bellman_ford', relaxation_limit))

plt.plot(densities, dijkstra_accuracy, label="Dijkstra")
plt.plot(densities, bellman_ford_accuracy, label="Bellman-Ford")
plt.xlabel("Density")
plt.ylabel("Accuracy")
plt.title("Accuracy vs. Density")
plt.legend()
plt.show()