from abc import ABC, abstractmethod
from typing import Dict, Tuple, List

class Graph(ABC):
    def __init__(self):
        self.edges = {}  # Stores adjacency list
        
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
    def add_edge(self, src: int, dst: int, weight: float):
        super().add_edge(src, dst)  # Call to superclass to handle adjacency
        if src not in self.edges:
            self.edges[src] = {}
        if dst not in self.edges:  # For directed graph only
            self.edges[dst] = {}
        self.edges[src][dst] = weight
        
    def w(self, node1: int, node2: int) -> float:
        try:
            return self.edges[node1][node2]
        except KeyError:
            return float('inf')  # No edge exists.


class HeuristicGraph(WeightedGraph):
    def __init__(self, nodes, heuristic: Dict):
        super().__init__(nodes)
        self.heuristic = heuristic

    def get_heuristic(self, node): 
        return self.heuristic.get(node, float('inf'))

class SPAlgorithm(ABC):
    def __init__(self, graph: Graph):
        self.graph = graph

    @abstractmethod
    def calc_sp(self, source: int, dest: int) -> Tuple[List[int], float]:
        pass

class Dijkstra(SPAlgorithm):
    def __init__(self, graph: WeightedGraph):
        super().__init__(graph)

    def calc_sp(self, source: int, dest: int) -> Tuple[List[int], float]:
        # Adapted graph structure to what Dijkstra's expects
        adapted_graph = {node: {} for node in self.graph.nodes}
        for src_node in self.graph.edges:
            for dst_node, weight in self.graph.edges[src_node].items():
                adapted_graph[src_node][dst_node] = weight
        
        # Running the Dijkstra's algorithm
        distance, path = dijkstra(adapted_graph, source, dest)
        
        return path, distance

class A_Star(SPAlgorithm):
    def __init__(self, graph: HeuristicGraph):
        super().__init__(graph)

    def calc_sp(self, source: int, dest: int) -> Tuple[List[int], float]:
        heuristic = self.graph.get_heuristic  # Assuming `get_heuristic` is a method to fetch heuristic values
        
        # Preparing a local function to match the `heuristic` argument's expected input by A_Star
        def local_heuristic(node):
            return heuristic(node)
        
        # Adjusting the graph's structure to match what A_Star expects
        adapted_graph = {}
        for node in self.graph.nodes:
            adapted_graph[node] = {}
            for adj_node in self.graph.get_adj_nodes(node):
                adapted_graph[node][adj_node] = self.graph.w(node, adj_node)
        
        # Running the A_Star function with the adapted arguments
        _, path = A_Star(adapted_graph, source, dest, local_heuristic)
        
        if not path:
            return [], float('inf')  # Indicating no path was found
        
        # Assuming the path cost needs to be calculated, as A_Star originally didn't return the path cost
        path_cost = sum([self.graph.w(path[i], path[i + 1]) for i in range(len(path) - 1)])
        
        return path, path_cost
    
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
graph = WeightedGraph()
graph.add_edge(1, 2, 3.5)
print(graph.w(1, 2))