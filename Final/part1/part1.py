import time
import matplotlib.pyplot as plt
import random

class MinPriorityQueue:
    def __init__(self):
        self.heap = []

    def parent(self, i):
        return (i - 1) // 2

    def left_child(self, i):
        return 2 * i + 1

    def right_child(self, i):
        return 2 * i + 2

    def insert(self, val):
        self.heap.append(val)
        self._heapify_up(len(self.heap) - 1)

    def delete_min(self):
        if len(self.heap) == 0:
            return None  # Handle empty queue case

        self.heap[0], self.heap[-1] = self.heap[-1], self.heap[0]
        min_val = self.heap.pop()
        self._heapify_down(0)
        return min_val

    def _heapify_up(self, index):
        while index > 0 and self.heap[self.parent(index)] > self.heap[index]:
            self.heap[index], self.heap[self.parent(index)] =self.heap[self.parent(index)], self.heap[index]
            index = self.parent(index)

    def _heapify_down(self, index):
        while index < len(self.heap):
            smallest = index
            left = self.left_child(index)
            right = self.right_child(index)

            if left < len(self.heap) and self.heap[left] < self.heap[smallest]:
                smallest = left
            if right < len(self.heap) and self.heap[right] < self.heap[smallest]:
                smallest = right

            if smallest != index:
                self.heap[index], self.heap[smallest] = self.heap[smallest], self.heap[index]
                index = smallest
            else:
                break

"""
queue = MinPriorityQueue()
queue.insert(10)
queue.insert(5)
queue.insert(20)
queue.insert(3)

print(queue.delete_min())
print(queue.delete_min())
"""

class Graph:
    def __init__(self, size, directed=False):
        self.adj_matrix = [[0] * size for _ in range(size)]
        self.size = size
        self.vertex_data = [''] * size
        self.directed = directed

    def add_edge(self, u, v, weight):
        if 0 <= u < self.size and 0 <= v < self.size:
            self.adj_matrix[u][v] = weight
            if not self.directed:
                self.adj_matrix[v][u] = weight  # For undirected graph

    def add_vertex_data(self, vertex, data):
        if 0 <= vertex < self.size:
            self.vertex_data[vertex] = data

    def num_edges(self):
        count = 0
        for row in self.adj_matrix:
            for edge in row:
                if edge != 0:
                    count += 1
        return count
    
    def num_vertices(self):
        return self.size

    def dijkstra(self, start_vertex_data, k):
        start_vertex = self.vertex_data.index(start_vertex_data)
        distances = [float('inf')] * self.size
        distances[start_vertex] = 0
        visited = [False] * self.size
        paths = {v: [] for v in range(self.size)}   # Store paths
        paths[start_vertex] = [start_vertex_data]   # Initial path for source
        counts = [0] * self.size  # Track relaxations per node

        for _ in range(self.size):
            min_distance = float('inf')
            u = None
            for i in range(self.size):
                if not visited[i] and distances[i] < min_distance and counts[i] < k:
                    min_distance = distances[i]
                    u = i

            if u is None:
                break

            visited[u] = True
            counts[u] += 1  # Increment relaxation count

            for v in range(self.size):
                if self.adj_matrix[u][v] != 0 and not visited[v]:
                    alt = distances[u] + self.adj_matrix[u][v]
                    if alt < distances[v]:
                        distances[v] = alt
                        # Update path
                        paths[v] = paths[u] + [self.vertex_data[v]] 
        return distances, paths
    
    def bellman_ford(self, start_vertex_data, k):
        start_vertex = self.vertex_data.index(start_vertex_data)
        distances = [float('inf')] * self.size
        distances[start_vertex] = 0
        paths = {v: [] for v in range(self.size)}
        paths[start_vertex] = [start_vertex_data]
        counts = [0] * self.size

        for i in range(self.size - 1):
            for u in range(self.size):
                for v in range(self.size):
                    if self.adj_matrix[u][v] != 0 and counts[v] < k:
                        if distances[u] + self.adj_matrix[u][v] < distances[v]:
                            distances[v] = distances[u] + self.adj_matrix[u][v]
                            paths[v] = paths[u] + [self.vertex_data[v]]
                            counts[v] += 1

        # Negative cycle detection
        for u in range(self.size):
            for v in range(self.size):
                if self.adj_matrix[u][v] != 0:
                    if distances[u] + self.adj_matrix[u][v] < distances[v]:
                        return True, [], {} 

        return False, distances, paths
    
    
"""
Running a very basic test in order to check if Dijkstra's and Bellman Ford algorithms are working as expected.
Below is the first test which is a basic test to check if the algorithms show the same distances for positive weights. Thus we'll do
this for an undirected graph.
"""

"""
my_graph = Graph(5)  # Create a graph with 5 vertices

my_graph.add_vertex_data(0, "City A")
my_graph.add_vertex_data(1, "City B")
my_graph.add_vertex_data(2, "City C")
my_graph.add_vertex_data(3, "City D")
my_graph.add_vertex_data(4, "City E")

my_graph.add_edge(0, 1, 10)  # Edge: City A to City B, distance 10
my_graph.add_edge(0, 2, 5)   # Edge: City A to City C, distance 5
my_graph.add_edge(1, 3, 3)   # Edge: City B to City D, distance 3
my_graph.add_edge(2, 3, 8)   # Edge: City C to City D, distance 8
my_graph.add_edge(2, 4, 2)   # Edge: City C to City E, distance 2
my_graph.add_edge(4, 3, 6)   # Edge: City E to City D, distance 6

# Find shortest paths from City A with up-to 2 relaxations allowed
distances_1_d, paths_1_d = my_graph.dijkstra("City A", 2) 

# Find shortest distances from City A using Bellman-Ford, allowing up-to 2 relaxations
has_negative_cycle_1_bf, distances_1_bf, paths_1_bf = my_graph.bellman_ford("City A", 2)

print("\nShortest distances from City A using Dijkstra's:")
for vertex_data, distance in zip(my_graph.vertex_data, distances_1_d):
    print(f"{vertex_data}: {distance}")
print("\nShortest paths from City A using Dijkstra's:")
for vertex, path in paths_1_d.items():
    print(f"To {my_graph.vertex_data[vertex]}: {' -> '.join(path)}")

print("\nShortest distances from City A using Bellman-Ford:")
for vertex_data, distance in zip(my_graph.vertex_data, distances_1_bf):
    print(f"{vertex_data}: {distance}")
print("\nShortest paths from City A using Bellman-Ford:")
for vertex, path in paths_1_bf.items():
    print(f"To {my_graph.vertex_data[vertex]}: {' -> '.join(path)}")

"""

"""
Now we need to check if the algorithms works as intended for negative weights. We'll create a graph with negative weights and check
that Dijkstra's fails and Bellman Ford works as intended. Not every negative graph will make Dijsktra's fail, but we'll create a graph
which gives it longer path which eventually has less weight but being a greedy algorithm it will fail to find it.
The graph kind of looks like this:
A ---- 4 ---> D
|             |
5             1
|             |
B --- -4 ---> C 
"""

"""
graph = Graph(4, True)

# Add vertex data
graph.add_vertex_data(0, 'A')
graph.add_vertex_data(1, 'B')
graph.add_vertex_data(2, 'C')
graph.add_vertex_data(3, 'D')

# Add edges with negative weights creating a negative cycle
graph.add_edge(0, 1, 5)
graph.add_edge(0, 3, 4)
graph.add_edge(1, 2, -4)
graph.add_edge(2, 3, 1)

# Test Bellman-Ford algorithm
has_negative_cycle_2_bf, distance_2_bf, paths_2_bf = graph.bellman_ford('A', 2)

if has_negative_cycle_2_bf:
    print("\nGraph contains a negative cycle.")
else:
    print("\nShortest distance from vertex A using Bellman Ford in example 2:")
    for vertex_data, distance in zip(graph.vertex_data, distance_2_bf):
        print(f"{vertex_data}: {distance}")
    print("\nShortest paths from vertex A using Bellman Ford in example 2::")
    for vertex, path in paths_2_bf.items():
        print(f"To {graph.vertex_data[vertex]}: {' -> '.join(path)}")

distancesd, pathsd = graph.dijkstra('A', 2) 
print("\nShortest distances from vertex A using Dijkstra's in example 2:")
for vertex_data, distance in zip(graph.vertex_data, distancesd):
    print(f"{vertex_data}: {distance}")
print("\nShortest paths from vertex A using Dijkstra's in example 2:")
for vertex, path in pathsd.items():
    print(f"To {graph.vertex_data[vertex]}: {' -> '.join(path)}")

"""

"""
Now we need to check if the algorithms works as intended for cyclic negative graph. We'll create a graph with negative weights and which is cyclic.
Bellman Ford should output True [], {} for no paths and distances
"""

"""
g = Graph(5, directed=True)

g.add_vertex_data(0, 'A')
g.add_vertex_data(1, 'B')
g.add_vertex_data(2, 'C')
g.add_vertex_data(3, 'D') 
g.add_vertex_data(4, 'E') 

g.add_edge(0, 1, 2)  # Edge from vertex 0 to vertex 1 with a weight of 2
g.add_edge(1, 2, 1)  # Edge from vertex 1 to vertex 2 with a weight of 1
g.add_edge(2, 0, -4) # Edge from vertex 2 to vertex 0 with a weight of -4
g.add_edge(2, 3, 3)  # Edge from vertex 2 to vertex 3 with a weight of 3
g.add_edge(3, 4, 1)  # Edge from vertex 3 to vertex 4 with a weight of 1

has_negative_cycle_3_bf, distances_3_bf, paths_3_bf = g.bellman_ford('A', 2)

if has_negative_cycle_3_bf:
    print("\nGraph contains a negative cycle so Bellman Ford won't process it.")
else:
    print("\nShortest distances from vertex A using Bellman Ford in example 3:")
    for vertex_data, distance in zip(g.vertex_data, distances_3_bf):
        print(f"{vertex_data}: {distance}")
    print("\nShortest paths from vertex A using Bellman Ford in example 3:")
    for vertex, path in paths_3_bf.items():
        print(f"To {g.vertex_data[vertex]}: {' -> '.join(path)}")

distances_3_d, paths_3_d = g.dijkstra('A', 2)

print("\nShortest distances from vertex A using Dijkstra's in example 3:")
for vertex_data, distance in zip(g.vertex_data, distances_3_d):
    print(f"{vertex_data}: {distance}")
print("\nShortest paths from vertex A using Dijkstra's in example 3:")
for vertex, path in paths_3_d.items():
    print(f"To {g.vertex_data[vertex]}: {' -> '.join(path)}")
print("\n")

"""

"""
After printing everything, it seems all out basic experimentation was passed successfully. Now we need to test the algorithms on a larger
scale to see how they perform.
Let's test the algorithms based on their time taken for three different variables:-
1. Size of the graph
2. Number of relaxations allowed
3. Graph density
"""

def generate_random_graph(size, density, directed=False, negative_weights=False):
    if density < 0 or density > 1:
        raise ValueError("Density must be between 0 and 1.")
    graph = Graph(size, directed)

    max_edges = size * (size - 1) // 2  # Maximum possible edges in an undirected graph 
    if directed:
        max_edges *= 2  # Double max edges for directed case

    num_edges = int(max_edges * density)

    for _ in range(num_edges):
        while True:
            u = random.randint(0, size - 1)
            v = random.randint(0, size - 1)
            if u != v and graph.adj_matrix[u][v] == 0:
                if negative_weights:
                    weight = random.randint(-5, 10)
                weight = random.randint(1, 10) 
                graph.add_edge(u, v, weight)
                break

    return graph

"""
#Testing if it works as needed
size = 5
density = 1
directed = False  # Set to True for a directed graph

random_graph = generate_random_graph(size, density, directed)

# Print out the adjacency matrix
for row in random_graph.adj_matrix:
    print(row)
"""
def time_algorithm(graph, source, algorithm='d', relaxation_limit=3):
    if algorithm.lower() == 'd':
        start_time = time.time()
        graph.dijkstra(source, k=relaxation_limit)
        end_time = time.time()
        return end_time - start_time
    start_time = time.time()
    graph.bellman_ford(source, k=relaxation_limit)
    end_time = time.time()
    return end_time - start_time


# Experiment 1: Variable Graph Size
graph_sizes = range(4, 20, 3)  # Graph sizes to test
density = 0.5   # Fixed density
dijkstra_times = []
bellman_ford_times = []

for size in graph_sizes:
    graph = generate_random_graph(size, density)
    source = random.choice(graph.vertex_data)  # Random source

    dijkstra_times.append(time_algorithm(graph, source, 'd', 4))
    bellman_ford_times.append(time_algorithm(graph, source, 'b', 4))

plt.figure(figsize=(10, 6))
plt.plot(graph_sizes, dijkstra_times, label='Dijkstra')
plt.plot(graph_sizes, bellman_ford_times, label='Bellman-Ford')
plt.xlabel('Graph Size')
plt.ylabel('Time (seconds)')
plt.title('Time Complexity: Graph Size Variation')
plt.legend()
plt.show()


# Experiment 2: Variable Density
graph_size = 50  # Fixed graph size
densities = [i / 10 for i in range(1, 10)]  # Densities to test

dijkstra_times = []
bellman_ford_times = []

for density in densities:
    graph = generate_random_graph(size, density)
    source = random.choice(graph.vertex_data)  # Random source

    dijkstra_times.append(time_algorithm(graph, source, 'd', 4))
    bellman_ford_times.append(time_algorithm(graph, source, 'b', 4))

plt.figure(figsize=(10, 6))
plt.plot(densities, dijkstra_times, label='Dijkstra')
plt.plot(densities, bellman_ford_times, label='Bellman-Ford')
plt.xlabel('Density')
plt.ylabel('Time (seconds)')
plt.title('Time Complexity: Density Variation')
plt.legend()
plt.show()

#Experiment 3: Variable Relaxation Limit
graph_size = 50  # Fixed graph size
density = 0.5  # Fixed density
relaxations = range(1, 10)  # Relaxations to test

dijkstra_times = []
bellman_ford_times = []
graph = generate_random_graph(size, density)

for k in relaxations:
    source = random.choice(graph.vertex_data)  # Random source

    dijkstra_times.append(time_algorithm(graph, source, 'd', k))
    bellman_ford_times.append(time_algorithm(graph, source, 'b', 10))

plt.figure(figsize=(10, 6))
plt.plot(relaxations, dijkstra_times, label='Dijkstra')
plt.plot(relaxations, bellman_ford_times, label='Bellman-Ford')
plt.xlabel('Relaxation Limits')
plt.ylabel('Time (seconds)')
plt.title('Time Complexity: Relaxation Limit Variation')
plt.legend()
plt.show()

"""
Next we test the accuracy of the algorithms by comparing the results of the two algorithms for again the same three cases.
We'll find the distance from before and after the relaxation limit and check if the distances are same or not.
"""

