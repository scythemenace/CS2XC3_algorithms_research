import random

class MinPriorityQueue:
    def __init__(self):
        self.heap = []

    def parent(self, i):
        return (i - 1) // 2
    
    def empty(self):
        return len(self.heap) == 0

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
            self.heap[index], self.heap[self.parent(index)] = self.heap[self.parent(index)], self.heap[index]
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

    def dijkstra_without_restriction(self, start_vertex_data):
        start_vertex = self.vertex_data.index(start_vertex_data)
        distances = [float('inf')] * self.size
        predecessors = [None] * self.size
        distances[start_vertex] = 0

        pq = MinPriorityQueue()
        pq.insert((0, start_vertex))

        while not pq.empty():
            (current_distance, u) = pq.delete_min()

            if current_distance > distances[u]:
                continue

            for v in range(self.size):
                if self.adj_matrix[u][v] != 0 and distances[v] > distances[u] + self.adj_matrix[u][v]:
                    alt = distances[u] + self.adj_matrix[u][v]
                    distances[v] = alt
                    predecessors[v] = u
                    pq.insert((alt, v))

        paths = {}
        for vertex in range(self.size):
            current_node = vertex
            path = [current_node]
            while predecessors[current_node] is not None:
                path.insert(0, predecessors[current_node])
                current_node = predecessors[current_node]
            paths[vertex] = path

        return distances, paths

    def bellman_ford_without_restriction(self, start_vertex_data):
        start_vertex = self.vertex_data.index(start_vertex_data)
        distances = [float('inf')] * self.size
        predecessors = [None] * self.size
        distances[start_vertex] = 0

        for i in range(self.size - 1):
            for u in range(self.size):
                for v in range(self.size):
                    if self.adj_matrix[u][v] != 0:
                        if distances[u] + self.adj_matrix[u][v] < distances[v]:
                            distances[v] = distances[u] + self.adj_matrix[u][v]
                            predecessors[v] = u

        # Negative cycle detection
        for u in range(self.size):
            for v in range(self.size):
                if self.adj_matrix[u][v] != 0:
                    if distances[u] + self.adj_matrix[u][v] < distances[v]:
                        raise ValueError("Graph contains a negative cycle")

        paths = {}
        for vertex in range(self.size):
            current_node = vertex
            path = [current_node]
            while predecessors[current_node] is not None:
                path.insert(0, predecessors[current_node])
                current_node = predecessors[current_node]
            paths[vertex] = path

        return distances, paths
    
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
                else:
                    weight = random.randint(1, 10) 
                graph.add_edge(u, v, weight)
                break

    return graph

def all_pairs_dijkstra(graph):
    """Assumes the graph only has positive edge weights"""
    shortest_paths = {}
    previous = {}
    distances = {} 

    for i in range(graph.size):
        source = graph.vertex_data[i]
        dist, paths = graph.dijkstra_without_restriction(source) 

        for vertex in range(graph.size):
            shortest_paths[(i, vertex)] = paths[vertex]
            previous[(i, vertex)] = paths[vertex][len(paths[vertex])-2]
            distances[(i, vertex)] = dist[vertex]

    return distances, shortest_paths, previous

def all_pairs_bellman_ford(graph):
    shortest_paths = {}
    previous = {}
    distances = {}

    for i in range(graph.size):
        source = graph.vertex_data[i]
        dist, paths = graph.bellman_ford_without_restriction(source)

        for vertex in range(graph.size):
            shortest_paths[(i, vertex)] = paths[vertex]
            previous[(i, vertex)] = paths[vertex][len(paths[vertex])-2]
            distances[(i, vertex)] = dist[vertex]

    return distances, shortest_paths, previous

graph = Graph(5)
graph.add_edge(0, 1, 4)
graph.add_edge(0, 2, 2)
graph.add_edge(1, 2, 1)
graph.add_edge(1, 3, 5)
graph.add_edge(2, 3, 2)
graph.add_edge(2, 4, 5)
graph.add_edge(3, 4, 1) 

for row in graph.adj_matrix:
    print(row)

distances, shortest_paths, previous = all_pairs_dijkstra(graph)

# Example: Path from vertex 0 to vertex 4
print("Shortest path from 0 to 4:", shortest_paths[(0, 4)]) 
# Output: Shortest path from 0 to 4: [0, 2, 3, 4] which means 0 -> 2 -> 3 -> 4

# Example: Distance from vertex 0 to vertex 4
print("Distance from 0 to 4:", distances[(0, 4)])
# Output: Distance from 0 to 4: 5

# Example: Second-to-last vertex on the path from 1 to 3
print("Second-to-last vertex from 1 to 3:", previous[(1, 3)]) 
# Output: Second-to-last vertex from 1 to 3: 2

# Create the graph
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

for row in graph.adj_matrix:
    print(row)

distances, shortest_paths, previous = all_pairs_bellman_ford(graph)

print("Shortest path from 0 to 4:", shortest_paths[(0, 3)])

print("Distance from 0 to 4:", distances[(0, 3)])