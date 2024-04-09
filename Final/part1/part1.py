import time
import matplotlib.pyplot as plt
import random

class Graph:
    def __init__(self, size):
        self.adj_matrix = [[0] * size for _ in range(size)]
        self.size = size
        self.vertex_data = [''] * size

    def add_edge(self, u, v, weight, directed=False):
        if 0 <= u < self.size and 0 <= v < self.size:
            self.adj_matrix[u][v] = weight
            if not directed:
                self.adj_matrix[v][u] = weight  # For undirected graph

    def add_vertex_data(self, vertex, data):
        if 0 <= vertex < self.size:
            self.vertex_data[vertex] = data

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
Below is the first test which is a basic test to check if the algorithms show the same distances for positive weights
"""
    
graph = Graph(4)

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
has_negative_cycle, distances, paths = graph.bellman_ford('A', 2)

if has_negative_cycle:
    print("Graph contains a negative cycle.")
else:
    print("Shortest distances from vertex A:")
    for vertex_data, distance in zip(graph.vertex_data, distances):
        print(f"{vertex_data}: {distance}")
    print("\nShortest paths from vertex A:")
    for vertex, path in paths.items():
        print(f"To {graph.vertex_data[vertex]}: {' -> '.join(path)}")

distancesd, pathsd = graph.dijkstra('A', 2) 
print("Shortest distances from vertex A:")
for vertex_data, distance in zip(graph.vertex_data, distancesd):
    print(f"{vertex_data}: {distance}")
print("\nShortest paths from vertex A:")
for vertex, path in pathsd.items():
    print(f"To {graph.vertex_data[vertex]}: {' -> '.join(path)}")