from abc import ABC, abstractmethod
from typing import Dict, Tuple, List


class Graph(ABC):
    def __init__(self):
        self.edges = {}  
        
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
        self.edges[dst].append(src) 

    def get_num_of_nodes(self) -> int:
        return len(self.edges)

    def get_adj_nodes(self, node: int) -> List[int]:
        return self.edges.get(node, [])

    @abstractmethod
    def w(self, node: int) -> float:

        pass

    
class WeightedGraph(Graph):
    def __init__(self):
        super().__init__()
        self.edges = {} 
    def add_edge(self, src: int, dst: int, weight: float):
        if src not in self.edges:
            self.edges[src] = {}
        if dst not in self.edges:
            self.edges[dst] = {}
        self.edges[src][dst] = weight
        self.edges[dst][src] = weight 

    def w(self, node1: int, node2: int) -> float:
        try:
            return self.edges[node1][node2]
        except KeyError:
            return float('inf')  



class HeuristicGraph(WeightedGraph):
    def __init__(self, _heuristic: Dict[int, float]):
        super().__init__()
        self._heuristic = _heuristic

    def get_heuristic(self, node) -> Dict[int, float]: 
        return self._heuristic.get(node, float('inf'))

class SPAlgorithm(ABC):
    def __init__(self, graph: Graph):
        self.graph = graph

    @abstractmethod
    def calc_sp(self, source: int, dest: int) -> float:
        pass

class Dijkstra(SPAlgorithm):
    def __init__(self, graph: WeightedGraph):
        super().__init__(graph)
    
    def calc_sp(self, source, dest) -> float:
        distance = {vertex: float('infinity') for vertex in self.graph.nodes}
        distance[source] = 0
        pq = MinPriorityQueue() 
        pq.put(source, 0)
        visited = set()
        predecessor = {vertex: None for vertex in self.graph.nodes}
        
        while not pq.is_empty():
            _, current_vertex = pq.delete_min()  
            
            if current_vertex in visited:
                continue
            visited.add(current_vertex)
            
            if current_vertex == dest:
                break
            
            for neighbor in self.graph.get_adj_nodes(current_vertex):
                if neighbor in visited:
                    continue
                alt_route = distance[current_vertex] + self.graph.w(current_vertex, neighbor)
                if alt_route < distance[neighbor]:
                    distance[neighbor] = alt_route
                    pq.put(neighbor, alt_route) 
                    predecessor[neighbor] = current_vertex
        
        # Reconstruct path
        path = []
        current = dest
        while current is not None and current in predecessor:
            path.append(current)
            current = predecessor[current]
        path.reverse()
        
        return distance[dest] if distance[dest] != float('infinity') else float('inf')


class A_Star_Adapter(SPAlgorithm):
    def __init__(self, heuristic_graph: HeuristicGraph):
        super().__init__(heuristic_graph)
        self.heuristic_graph = heuristic_graph

    def calc_sp(self, source: int, dest: int) -> float:
        graph = {node: {} for node in self.heuristic_graph.nodes}
        for node, neighbors in self.heuristic_graph.edges.items():
            for neighbor, weight in neighbors.items():
                graph[node][neighbor] = weight
        
        heuristic = {node: self.heuristic_graph.get_heuristic(node) for node in self.heuristic_graph.nodes}


        _, cost = self.a_star_algorithm(graph, source, dest, heuristic)

        return cost

    def a_star_algorithm(self, graph, source, destination, heuristic):
        open_list = MinPriorityQueue()
        open_list.put(source, heuristic[source])  
        predecessors = {source: None}
        costs = {source: 0}

        while not open_list.is_empty():
            _, current = open_list.delete_min()

            if current == destination:

                break

            for neighbor, weight in graph[current].items():
                new_cost = costs[current] + weight
                if neighbor not in costs or new_cost < costs[neighbor]:
                    costs[neighbor] = new_cost
                    priority = new_cost + heuristic[neighbor] 
                    open_list.put(neighbor, priority)
                    predecessors[neighbor] = current


        if destination not in costs:
            return [], float('inf')  


        return self.reconstruct_path(predecessors, source, destination), costs[destination]

    def reconstruct_path(self, predecessors, start, end):
        path = []
        current = end
        while current != start and current is not None:
            path.append(current)
            current = predecessors.get(current)
        if current is None: 
            return [], float('inf')
        path.append(start)
        path.reverse()
        return path




    
class BellmanFord(SPAlgorithm):
    def calc_sp(self, source: int, dest: int) -> float:
        distances = {v: float('inf') for v in self.graph.nodes}
        predecessors = {v: None for v in self.graph.nodes}
        distances[source] = 0

        for _ in range(len(self.graph.nodes) - 1):
            for src in self.graph.edges:
                for dst, weight in self.graph.edges[src].items():
                    if distances[src] + weight < distances[dst]:
                        distances[dst] = distances[src] + weight
                        predecessors[dst] = src


        for src in self.graph.edges:
            for dst, weight in self.graph.edges[src].items():
                if distances[src] + weight < distances[dst]:
                    raise ValueError("Graph contains a negative-weight cycle")


        path = []
        current = dest
        while current is not None:
            path.append(current)
            current = predecessors[current]
        path.reverse()

        return distances[dest] if distances[dest] != float('inf') else None


# For A* and Dijkstra
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
            return None

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


    def put(self, node, priority):
        self.insert((priority, node))

    def is_empty(self):
        return len(self.heap) == 0


class ShortPathFinder:
    def __init__(self, graph: Graph, algo: SPAlgorithm):
        self.graph = graph
        self.algo = algo

    def calc_short_path(self, source: int, dest: int) -> float:
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
    heuristic = {1: 9, 2: 7, 3: 4, 4: 2, 5: 0}  
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
    cost = finder.calc_short_path(1, 5) 
    print(cost)  

def test_dijkstra_disconnected_graph():
    graph = WeightedGraph()
    graph.add_edge(1, 2, 1.0)
    graph.add_edge(3, 4, 1.0)  
    dijkstra_algo = Dijkstra(graph)
    finder = ShortPathFinder(graph, dijkstra_algo)
    cost = finder.calc_short_path(1, 4)
    print(cost)  


def test_bellman_ford_negative_cycle():
    graph = WeightedGraph()
    graph.add_edge(1, 2, 4.0)
    graph.add_edge(2, 3, -6.0)
    graph.add_edge(3, 1, 2.0)
    bellman_ford_algo = BellmanFord(graph)
    finder = ShortPathFinder(graph, bellman_ford_algo)
    try:
        cost = finder.calc_short_path(1, 3)
        assert False, "Negative cycle detection failed"
    except ValueError as e:
        assert str(e) == "Graph contains a negative-weight cycle", "Incorrect error message"
    print("Test Case 4 Passed: Bellman-Ford Negative Weight Cycle")
def test_a_star_adapter():
    # Test case 1
    graph1 = HeuristicGraph({0: 2, 1: 1, 2: 0})
    graph1.add_edge(0, 1, 1)
    graph1.add_edge(1, 2, 1)
    a_star1 = A_Star_Adapter(graph1)
    print(a_star1.calc_sp(0, 2))  # Expected: ([0, 1, 2], 2)

    # Test case 2
    graph2 = HeuristicGraph({0: 2, 1: 1, 2: 1, 3: 0})
    graph2.add_edge(0, 1, 1)
    graph2.add_edge(0, 2, 5)
    graph2.add_edge(1, 3, 1)
    graph2.add_edge(2, 3, 1)
    a_star2 = A_Star_Adapter(graph2)
    print(a_star2.calc_sp(0, 3))  # Expected: ([0, 1, 3], 2)

    # Test case 3
    graph3 = HeuristicGraph({0: 3, 1: 2, 2: 1, 3: 0})
    graph3.add_edge(0, 1, 1)
    graph3.add_edge(1, 2, 1)
    graph3.add_edge(2, 1, 1)
    graph3.add_edge(2, 3, 1)
    a_star3 = A_Star_Adapter(graph3)
    print(a_star3.calc_sp(0, 3))  # Expected: ([0, 1, 2, 3], 3)

    # Test case 4
    graph4 = HeuristicGraph({0: 1, 1: 2, 2: 1, 3: 0})
    graph4.add_edge(0, 1, 1)
    graph4.add_edge(0, 2, 2)
    graph4.add_edge(0, 3, 3)
    a_star4 = A_Star_Adapter(graph4)
    print(a_star4.calc_sp(0, 3))  # Expected: ([0, 3], 3)
def test_a_star_simple_weighted_graph():
    heuristic = {0: 3, 1: 2, 2: 1, 3: 0}
    graph = HeuristicGraph(heuristic)
    graph.add_edge(0, 1, 1)
    graph.add_edge(1, 2, 1)
    graph.add_edge(2, 3, 1)
    graph.add_edge(0, 3, 4)

    a_star_algo = A_Star_Adapter(graph)
    finder = ShortPathFinder(graph, a_star_algo)

    cost = finder.calc_short_path(0, 3)
    print("Test A* Simple Weighted Graph:", "Passed" if cost == 3 else "Failed", "- Cost:", cost)
def test_bellman_ford_no_negative_cycle():
    graph = WeightedGraph()
    graph.add_edge(0, 1, 5)
    graph.add_edge(1, 2, 3)
    graph.add_edge(2, 0, 2)

    bellman_ford_algo = BellmanFord(graph)
    finder = ShortPathFinder(graph, bellman_ford_algo)

    cost = finder.calc_short_path(0, 2)
    print("Test Bellman-Ford No Negative Cycle:", "Passed" if cost == 8 else "Failed", "- Cost:", cost)
def test_sp_algorithm_variety_graph_types():
    graph = WeightedGraph()
    graph.add_edge(0, 1, 2)
    graph.add_edge(1, 2, 2)

    # Dijkstra
    dijkstra_algo = Dijkstra(graph)
    finder = ShortPathFinder(graph, dijkstra_algo)
    cost_dijkstra = finder.calc_short_path(0, 2)

    # Bellman-Ford
    bellman_ford_algo = BellmanFord(graph)
    finder.set_algorithm(bellman_ford_algo)
    cost_bellman = finder.calc_short_path(0, 2)

    print("Test SPAlgorithm with Dijkstra and Bellman-Ford:", "Passed" if cost_dijkstra == cost_bellman == 4 else "Failed", "- Dijkstra Cost:", cost_dijkstra, "- Bellman-Ford Cost:", cost_bellman)



test_dijkstra_simple_path()
test_dijkstra_disconnected_graph()
test_bellman_ford_negative_cycle()
test_a_star_adapter()
test_a_star_simple_weighted_graph()
test_bellman_ford_no_negative_cycle()
test_sp_algorithm_variety_graph_types()

