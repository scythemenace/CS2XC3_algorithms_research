from abc import ABC, abstractmethod
from typing import Dict, Tuple, List

class Graph(ABC):
    def __init__(self):
        self.edges = {}
        
    @abstractmethod
    def w(self, node: int) -> float:
        pass
    
class WeightedGraph(Graph):
    def add_edge(self, src: int, dst: int, weight: float = 1.0):
        if src not in self.edges:
            self.edges[src] = {}
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
    def calc_sp(self, source: int, dest: int) -> Tuple[List[int], float]:
        # Implementation of the Dijkstra algorithm
        return [], 0.0

class A_Star(SPAlgorithm):
    def calc_sp(self, source: int, dest: int) -> Tuple[List[int], float]:
        # Implementation of the A* algorithm
        return [], 0.0
    
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
