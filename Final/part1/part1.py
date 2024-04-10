import time
import matplotlib.pyplot as plt
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
        paths = {v: [] for v in range(self.size)}
        paths[start_vertex] = [start_vertex_data]
        counts = [0] * self.size

        pq = MinPriorityQueue()
        pq.insert((0, start_vertex))  # Insert (distance, vertex)

        while not pq.empty():  
            current_distance, u = pq.delete_min()

            if counts[u] >= k:
                continue

            visited[u] = True
            counts[u] += 1  

            for v in range(self.size):
                if self.adj_matrix[u][v] != 0 and not visited[v]:
                    alt = current_distance + self.adj_matrix[u][v]
                    if alt < distances[v]:
                        distances[v] = alt
                        paths[v] = paths[u] + [self.vertex_data[v]]
                        pq.insert((alt, v))

        return distances, paths

    
    def bellman_ford(self, start_vertex_data, k):
        start_vertex = self.vertex_data.index(start_vertex_data)
        distances = [float('inf')] * self.size
        distances[start_vertex] = 0
        paths = {v: [] for v in range(self.size)}

        counts = [0] * self.size
        for i in range(k):
            for u in range(self.size):
                for v in range(self.size):
                    if self.adj_matrix[u][v] != 0 :
                        if distances[u] + self.adj_matrix[u][v] < distances[v]:
                            distances[v] = distances[u] + self.adj_matrix[u][v]
                            paths[v] = paths[u] + [self.vertex_data[v]]
                            counts[v] += 1

        # Negative cycle detection
        for u in range(self.size):
            for v in range(self.size):
                if self.adj_matrix[u][v] != 0:
                    if distances[u] + self.adj_matrix[u][v] < distances[v]:
                        raise ValueError("Graph contains a negative cycle")
                    
        return distances, paths


"""
Running a very basic test in order to check if Dijkstra's and Bellman Ford algorithms are working as expected.
Below is the first test which is a basic test to check if the algorithms show the same distances for positive weights. Thus we'll do
this for an undirected graph.
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
distances_1_bf, paths_1_bf = my_graph.bellman_ford("City A", 2)

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
Now we need to check if the algorithms works as intended for negative weights. We'll create a graph with negative weights and check
that Dijkstra's fails and Bellman Ford works as intended. We've created a graph which gives it longer path which eventually has 
less weight but being a greedy algorithm it will fail to find it.
The graph kind of looks like this:
A ---- 4 ---> D
|             |
5             1
|             |
B --- -4 ---> C 

You'll see in the test below Dijkstra's fails to give the shortest path and choose A -> D because it has weight 4 whereas Bellman Ford
chooses A -> B -> C -> D which has weight 2 because even though initially it has weight 5, it has a negative weight of -4 which makes
it 1 and then finally the additional wieght of 1 from C -> D makes it 2.
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

# Test Dijkstra's algorithm
distancesd, pathsd = graph.dijkstra('A', 2) 
print("\nShortest distances from vertex A using Dijkstra's in example 2:")
for vertex_data, distance in zip(graph.vertex_data, distancesd):
    print(f"{vertex_data}: {distance}")
print("\nShortest paths from vertex A using Dijkstra's in example 2:")
for vertex, path in pathsd.items():
    print(f"To {graph.vertex_data[vertex]}: {' -> '.join(path)}")

# Test Bellman-Ford algorithm
distance_2_bf, paths_2_bf = graph.bellman_ford('A', 2)

print("\nShortest distance from vertex A using Bellman Ford in example 2:")
for vertex_data, distance in zip(graph.vertex_data, distance_2_bf):
    print(f"{vertex_data}: {distance}")
print("\nShortest paths from vertex A using Bellman Ford in example 2:")
for vertex, path in paths_2_bf.items():
    print(f"To {graph.vertex_data[vertex]}: {' -> '.join(path)}")


"""
Now we need to check if the algorithms works as intended for cyclic negative graph. We'll create a graph with negative weights and which is cyclic.
Bellman Ford should raise an error whenever that happens

Why are we raising an error instead of just outputting some distances. Well, essentially if something is in an infinite loop and you sum the negative
weights then logically, our distance can be -infinity for every vertex. Because they can be very skewed, if the graph is cyclic and has 
negative weights. For checking this just modify the bellman ford code above and instead of raising an error. Make it return True, 
distances, paths. You'll see that based on the relaxations we've allowed the paths are changing and the distances are also changing. 
For example, there is a graph below:-

A -- 2 --> B
 \         |
  \        |
   \       |
    \      |
    -4     1
      \    |
       \   |
        \  |
         \ |
          \| 
           C --- 3 ---> D ---- 1 ---> E

Now in the graph above if we allow 2 relaxations, the path for shortest distance from A to A will be:
A -> B -> C -> A -> B -> C -> A
which makes the distance as -2
but if we allow 3 relaxations, the path for shortest distance from A to A will be:
A -> B -> C -> A -> B -> C -> A -> B -> C -> A
which makes the distance as -3

Therefore, we choose to return nothing, which is [] and {} for distances and paths respectively.
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

distances_3_d, paths_3_d = g.dijkstra('A', 2)

print("\nShortest distances from vertex A using Dijkstra's in example 3:")
for vertex_data, distance in zip(g.vertex_data, distances_3_d):
    print(f"{vertex_data}: {distance}")
print("\nShortest paths from vertex A using Dijkstra's in example 3:")
for vertex, path in paths_3_d.items():
    print(f"To {g.vertex_data[vertex]}: {' -> '.join(path)}")

try:
    distances_3_bf, paths_3_bf = g.bellman_ford('A', 2)
    print("\nShortest distances from vertex A using Bellman Ford in example 3:")
    for vertex_data, distance in zip(g.vertex_data, distances_3_bf):
        print(f"{vertex_data}: {distance}")
    print("\nShortest paths from vertex A using Bellman Ford in example 3:")
    for vertex, path in paths_3_bf.items():
        print(f"To {g.vertex_data[vertex]}: {' -> '.join(path)}")
except ValueError as e:
    print(e)


"""
After printing everything, it seems all our basic experimentation was passed successfully. Now we need to test the algorithms on a larger
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
                else:
                    weight = random.randint(1, 10) 
                graph.add_edge(u, v, weight)
                break

    return graph


"""
#Testing if it works as needed
size = 5
density = 0.8
directed = False  # Set to True for a directed graph

random_graph = generate_random_graph(size, density, directed, negative_weights=True)

# Print out the adjacency matrix
for row in random_graph.adj_matrix:
    print(row)

#Note that the diagonals will always be 0 even thought density is 1 because we're not allowing self loops
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
graph_sizes = range(5, 100, 10)  # Graph sizes to test
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

#Only Dijkstra
plt.figure(figsize=(10, 6))
plt.plot(graph_sizes, dijkstra_times, label='Dijkstra')
plt.xlabel('Graph Size')
plt.ylabel('Time (seconds)')
plt.title('Time Complexity: Graph Size Variation for Dijkstra only')
plt.legend()
plt.show()

#Only Bellman Ford
plt.figure(figsize=(10, 6))
plt.plot(graph_sizes, bellman_ford_times, label='Dijkstra')
plt.xlabel('Graph Size')
plt.ylabel('Time (seconds)')
plt.title('Time Complexity: Graph Size Variation for Bellman Ford only')
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

#Only Dijkstra
plt.figure(figsize=(10, 6))
plt.plot(densities, dijkstra_times, label='Dijkstra')
plt.xlabel('Density')
plt.ylabel('Time (seconds)')
plt.title('Time Complexity: Density Variation for Dijkstra only')
plt.legend()
plt.show()

#Only Bellman Ford
plt.figure(figsize=(10, 6))
plt.plot(densities, bellman_ford_times, label='Dijkstra')
plt.xlabel('Density')
plt.ylabel('Time (seconds)')
plt.title('Time Complexity: Density Variation for Bellman Ford only')
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

#Only Dijkstra
plt.figure(figsize=(10, 6))
plt.plot(relaxations, dijkstra_times, label='Dijkstra')
plt.xlabel('Relaxation Limits')
plt.ylabel('Time (seconds)')
plt.title('Time Complexity: Relaxation Limit Variation for Dijkstra only')
plt.legend()
plt.show()

#Only Bellman Ford
plt.figure(figsize=(10, 6))
plt.plot(relaxations, bellman_ford_times, label='Dijkstra')
plt.xlabel('Relaxation Limits')
plt.ylabel('Time (seconds)')
plt.title('Time Complexity: Relaxation Limit Variation for Bellman Ford only')
plt.legend()
plt.show()

"""
Next we test the accuracy of the algorithms by comparing the results of the two algorithms for again the same three cases.
We'll find the distance from before and after the relaxation limit and check if the distances are same or not.

We'll create a normal Dijkstra's and a normal Bellman Ford without k limitations. Then we'll compare the results of the two algorithms
for the same source and check how close the distances are for the same source.
"""
def dijkstra_without_restriction(graph, start_vertex_data):
    start_vertex = graph.vertex_data.index(start_vertex_data)
    distances = [float('inf')] * graph.size
    predecessors = [None] * graph.size
    distances[start_vertex] = 0

    pq = MinPriorityQueue()
    pq.insert((0, start_vertex))

    while not pq.empty():
        (current_distance, u) = pq.delete_min()

        if current_distance > distances[u]:
            continue

        for v in range(graph.size):
            if graph.adj_matrix[u][v] != 0 and distances[v] > distances[u] + graph.adj_matrix[u][v]:
                alt = distances[u] + graph.adj_matrix[u][v]
                distances[v] = alt
                predecessors[v] = u
                pq.insert((alt, v))

    paths = {}
    for vertex in range(graph.size):
        current_node = vertex
        path = [current_node]
        while predecessors[current_node] is not None:
            path.insert(0, predecessors[current_node])
            current_node = predecessors[current_node]
        paths[vertex] = path

    return distances, paths

def bellman_ford_without_restriction(graph, start_vertex_data):
    start_vertex = graph.vertex_data.index(start_vertex_data)
    distances = [float('inf')] * graph.size
    predecessors = [None] * graph.size
    distances[start_vertex] = 0

    for i in range(graph.size - 1):
        for u in range(graph.size):
            for v in range(graph.size):
                if graph.adj_matrix[u][v] != 0:
                    if distances[u] + graph.adj_matrix[u][v] < distances[v]:
                        distances[v] = distances[u] + graph.adj_matrix[u][v]
                        predecessors[v] = u

    # Negative cycle detection
    for u in range(graph.size):
        for v in range(graph.size):
            if graph.adj_matrix[u][v] != 0:
                if distances[u] + graph.adj_matrix[u][v] < distances[v]:
                    raise ValueError("Graph contains a negative cycle")

    paths = {}
    for vertex in range(graph.size):
        current_node = vertex
        path = [current_node]
        while predecessors[current_node] is not None:
            path.insert(0, predecessors[current_node])
            current_node = predecessors[current_node]
        paths[vertex] = path

    return distances, paths

def measure_accuracy(graph, source, algorithm, relaxation_limit=4):
    if algorithm == 'dijkstra':
        distances, paths = graph.dijkstra(source, k=relaxation_limit)
        unrestricted_distances, unrestricted_paths = dijkstra_without_restriction(graph, source)
    elif algorithm == 'bellman_ford':
        distances, paths = graph.bellman_ford(source, k=relaxation_limit)
        unrestricted_distances, unrestricted_paths = bellman_ford_without_restriction(graph, source)
    else:
        return None 

    distance_error_sum = 0
    path_matches = 0

    for i in range(len(distances)):
        distance_error_sum += abs(distances[i] - unrestricted_distances[i])
        if paths[i] == unrestricted_paths[i]:
            path_matches += 1

    average_distance_error = distance_error_sum / len(distances)
    path_accuracy = path_matches / len(distances) * 100

    #We return distance error as well in case we want to do some more testing but we won't plot it. We'll only plot path accuracy for now.
    return average_distance_error, path_accuracy 

#Experiment 1: Variable Graph Size
graph_sizes = range(5, 101, 5)  
density = 0.5  
relaxation_limit = 5  

dijkstra_accuracies = [] 
bellman_ford_accuracies = []

for size in graph_sizes:
    graph = generate_random_graph(size, density)
    source = random.choice(graph.vertex_data)

    dijkstra_accuracies.append(measure_accuracy(graph, source, 'dijkstra', relaxation_limit))
    bellman_ford_accuracies.append(measure_accuracy(graph, source, 'bellman_ford', relaxation_limit))


# Plotting code 
plt.figure(figsize=(10, 6))
plt.plot(graph_sizes, dijkstra_accuracies, label='Dijkstra')
plt.plot(graph_sizes, bellman_ford_accuracies, label='Bellman-Ford')
plt.xlabel('Graph Size')
plt.ylabel('Path Accuracy (%)')
plt.title("Experiment 1: Path Accuracy vs. Graph Size")
plt.legend()
plt.show()

#Experiment 2: Variable Density
graph_sizes = 50
density = [i / 10 for i in range(1, 10)]
relaxation_limit = 5  

dijkstra_accuracies = [] 
bellman_ford_accuracies = []

for density in densities:
    graph = generate_random_graph(size, density)
    source = random.choice(graph.vertex_data)

    dijkstra_accuracies.append(measure_accuracy(graph, source, 'dijkstra', relaxation_limit))
    bellman_ford_accuracies.append(measure_accuracy(graph, source, 'bellman_ford', relaxation_limit))

# Plotting code 
plt.figure(figsize=(10, 6))
plt.plot(densities, dijkstra_accuracies, label='Dijkstra')
plt.plot(densities, bellman_ford_accuracies, label='Bellman-Ford')
plt.xlabel('Densities')
plt.ylabel('Path Accuracy (%)')
plt.title("Experiment 2: Path Accuracy vs. Densities")
plt.legend()
plt.show()

#Experiment 3: Variable Relaxation Limit
graph_sizes = 50
density = 0.6
relaxations = range(1, 10)
dijkstra_accuracies = [] 
bellman_ford_accuracies = []
graph = generate_random_graph(size, density)

for k in relaxations:
    source = random.choice(graph.vertex_data)

    dijkstra_accuracies.append(measure_accuracy(graph, source, 'dijkstra', k))
    bellman_ford_accuracies.append(measure_accuracy(graph, source, 'bellman_ford', k))

# Plotting code 
plt.figure(figsize=(10, 6))
plt.plot(relaxations, dijkstra_accuracies, label='Dijkstra')
plt.plot(relaxations, bellman_ford_accuracies, label='Bellman-Ford')
plt.xlabel('Relaxation Limit')
plt.ylabel('Path Accuracy (%)')
plt.title("Experiment 2: Path Accuracy vs. Relaxation Limit")
plt.legend()
plt.show()
plt.show()