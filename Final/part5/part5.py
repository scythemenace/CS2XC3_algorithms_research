from abc import ABC, abstractmethod
from typing import Dict, Tuple, List


class Graph(ABC):
    def __init__(self):
        self.edges = {}  # Stores adjacency list
        
    @property
    def nodes(self):
        return list(self.edges.keys())
    
    def add_node(self, node: int):
        if node not in self.edges:
            self.edges[node] = []
    
    def add_edge(self, src: int, dst: int):
        if src not in self.edges:
            self.add_node(src)
        if dst not in self.edges:
            self.add_node(dst)
        self.edges[src].append(dst)
        self.edges[dst].append(src)  # Assuming undirected graph for simplicity

    def get_num_of_nodes(self) -> int:
        return len(self.edges)

    def get_adj_nodes(self, node: int) -> List[int]:
        return self.edges.get(node, [])

    @abstractmethod
    def w(self, node: int) -> float:
        # Abstract method, should be implemented in child classes
        pass

    
class WeightedGraph(Graph):
    def __init__(self):
        super().__init__()
        self.edges = {}  # Overrides the base class to use a dict for weighted edges

    def add_edge(self, src: int, dst: int, weight: float):
        if src not in self.edges:
            self.edges[src] = {}
        if dst not in self.edges:
            self.edges[dst] = {}
        self.edges[src][dst] = weight
        self.edges[dst][src] = weight  # Assuming undirected graph for simplicity

    def w(self, node1: int, node2: int) -> float:
        try:
            return self.edges[node1][node2]
        except KeyError:
            return float('inf')  # No edge exists



class HeuristicGraph(WeightedGraph):
    def __init__(self, heuristic: Dict[int, float]):
        super().__init__()
        self.heuristic = heuristic

    def get_heuristic(self, node): 
        return self.heuristic.get(node, float('inf'))

class SPAlgorithm(ABC):
    def __init__(self, graph: Graph):
        self.graph = graph

    @abstractmethod
    def calc_sp(self, source: int, dest: int) -> Tuple[List[int], float]:
        pass

def dijkstra(graph, start, end):
    distance = {vertex: float('infinity') for vertex in graph}
    distance[start] = 0
    predecessor = {vertex: None for vertex in graph}
    pq = PriorityQueue()
    pq.put(start, 0)
    
    while not pq.is_empty():
        current_node = pq.get()
        
        if current_node == end:
            break

        for neighbor, weight in graph[current_node].items():
            alt_route = distance[current_node] + weight
            if alt_route < distance[neighbor]:
                distance[neighbor] = alt_route
                predecessor[neighbor] = current_node
                pq.put(neighbor, alt_route)
                
    # Reconstruct path from end to start using predecessors
    path = []
    current = end
    while current is not None:
        path.append(current)
        current = predecessor[current]
    path.reverse()  # Reverse the path to start from the beginning
    
    return distance[end], path

class Dijkstra(SPAlgorithm):
    def __init__(self, graph: WeightedGraph):
        super().__init__(graph)

    def calc_sp(self, source: int, dest: int) -> Tuple[List[int], float]:

        adapted_graph = {node: {} for node in self.graph.nodes}
    
        for src_node, neighbors in self.graph.edges.items():
            for dst_node, weight in neighbors.items():
                adapted_graph[src_node][dst_node] = weight
        
        # Running the Dijkstra's algorithm
        distance, path = dijkstra(adapted_graph, source, dest)
        
        return path, distance

class A_Star_Adapter(SPAlgorithm):
    def __init__(self, heuristic_graph: HeuristicGraph):
        super().__init__(heuristic_graph)
        self.heuristic_graph = heuristic_graph

    def calc_sp(self, source: int, dest: int) -> Tuple[List[int], float]:
        graph = {node: {} for node in self.heuristic_graph.nodes}
        for node, neighbors in self.heuristic_graph.edges.items():
            for neighbor in neighbors:
                if neighbor in self.heuristic_graph.edges[node]: 
                    graph[node][neighbor] = self.heuristic_graph.w(node, neighbor)
        
        heuristic = {node: self.heuristic_graph.get_heuristic(node) for node in self.heuristic_graph.nodes}

        _, path = self.a_star_algorithm(graph, source, dest, heuristic)

        if not path:
            return [], float('inf') 

        path_cost = sum([self.heuristic_graph.w(path[i], path[i+1]) for i in range(len(path) - 1)])
        
        return path, path_cost

    def a_star_algorithm(self, graph, source, destination, heuristic):
        open_list = PriorityQueue()
        open_list.put(source, 0 + heuristic[source])
        predecessors = {source: None}
        costs = {source: 0}

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

        path = self.reconstruct_path(predecessors, source, destination)
        if not path:
            return predecessors, [], "Destination not reachable"

        return predecessors, path

    def reconstruct_path(self, predecessors, start, end):
        if end not in predecessors:
            return []  # Path not found
        path = []
        while end is not None:
            path.append(end)
            end = predecessors.get(end)
        path.reverse()  # Reverse the path to start from the beginning
        return path

    
class BellmanFord(SPAlgorithm):
    def calc_sp(self, source: int, dest: int) -> Tuple[List[int], float]:
        # Initialize distance to all vertices as infinite and distance to source as 0
        distances = {v: float('inf') for v in self.graph.nodes}
        predecessors = {v: None for v in self.graph.nodes}
        distances[source] = 0

        # Relax edges repeatedly
        for _ in range(len(self.graph.nodes) - 1):
            for src in self.graph.edges:
                for dst, weight in self.graph.edges[src].items():
                    if distances[src] + weight < distances[dst]:
                        distances[dst] = distances[src] + weight
                        predecessors[dst] = src

        # Check for negative-weight cycles
        for src in self.graph.edges:
            for dst, weight in self.graph.edges[src].items():
                if distances[src] + weight < distances[dst]:
                    raise ValueError("Graph contains a negative-weight cycle")

        # Reconstruct the path from source to dest
        path = []
        current = dest
        while current is not None:
            path.append(current)
            current = predecessors[current]
        path.reverse()

        return path, distances[dest] if distances[dest] != float('inf') else None

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


class ShortPathFinder:
    def __init__(self, graph: Graph, algo: SPAlgorithm):
        self.graph = graph
        self.algo = algo

    def calc_short_path(self, source: int, dest: int) -> Tuple[List[int], float]:
        return self.algo.calc_sp(source, dest)

    def set_graph(self, graph: Graph):
        self.graph = graph
        self.algo.graph = graph

    def set_algorithm(self, algo: SPAlgorithm):
        self.algo = algo


"""Detailed testing to ensure that everything is working or not"""
def create_test_graph():
    graph = WeightedGraph()
    graph.add_edge(1, 2, 1.0)
    graph.add_edge(2, 3, 2.0)
    graph.add_edge(3, 4, 3.0)
    graph.add_edge(4, 5, 4.0)
    graph.add_edge(1, 5, 10.0)
    return graph

def create_heuristic_graph():
    heuristic = {1: 9, 2: 7, 3: 4, 4: 2, 5: 0}  # Example heuristic, assuming 5 is the goal
    graph = HeuristicGraph(heuristic)
    graph.add_edge(1, 2, 1.0)
    graph.add_edge(2, 3, 2.0)
    graph.add_edge(3, 4, 3.0)
    graph.add_edge(4, 5, 4.0)
    graph.add_edge(1, 5, 10.0)
    return graph

def test_dijkstra_simple_path():
    graph = create_test_graph()
    dijkstra_algo = Dijkstra(graph)
    finder = ShortPathFinder(graph, dijkstra_algo)
    path, cost = finder.calc_short_path(1, 5)
    print(path, cost)


def test_a_star_simple_path():
    graph = create_heuristic_graph()
    a_star_algo = A_Star_Adapter(graph)
    finder = ShortPathFinder(graph, a_star_algo)
    path, cost = finder.calc_short_path(1, 5)
    print(path, cost)


def test_dijkstra_disconnected_graph():
    graph = WeightedGraph()
    graph.add_edge(1, 2, 1.0)
    graph.add_edge(3, 4, 1.0)  # Disconnected component
    dijkstra_algo = Dijkstra(graph)
    finder = ShortPathFinder(graph, dijkstra_algo)
    path, cost = finder.calc_short_path(1, 4)
    print(path, cost)

def test_bellman_ford_negative_cycle():
    graph = WeightedGraph()
    graph.add_edge(1, 2, 4.0)
    graph.add_edge(2, 3, -6.0)
    graph.add_edge(3, 1, 2.0)
    bellman_ford_algo = BellmanFord(graph)
    finder = ShortPathFinder(graph, bellman_ford_algo)
    try:
        path, cost = finder.calc_short_path(1, 3)
        assert False, "Negative cycle detection failed"
    except ValueError as e:
        assert str(e) == "Graph contains a negative-weight cycle", "Incorrect error message"
    print("Test Case 4 Passed: Bellman-Ford Negative Weight Cycle")


if __name__ == "__main__":
    test_dijkstra_simple_path()
    test_a_star_simple_path()
    test_dijkstra_disconnected_graph()
    test_bellman_ford_negative_cycle()
