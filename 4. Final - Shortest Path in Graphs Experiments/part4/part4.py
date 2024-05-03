
import csv
from math import sqrt
import time
import matplotlib.pyplot as plt


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
    
def dijkstra(graph, start, end):
    if start not in graph or end not in graph:
        return None, [] 

    distance = {vertex: float('infinity') for vertex in graph}
    distance[start] = 0
    predecessor = {vertex: None for vertex in graph}
    pq = MinPriorityQueue()
    pq.put(start, 0)
    
    while not pq.is_empty():
        _, current_node = pq.delete_min()
        
        if current_node == end:
            break

        for neighbor, weight in graph.get(current_node, {}).items():
            alt_route = distance[current_node] + weight
            if alt_route < distance[neighbor]:
                distance[neighbor] = alt_route
                predecessor[neighbor] = current_node
                pq.put(neighbor, alt_route)

    if distance[end] == float('infinity'):  
        return None, []  

    path = []
    current = end
    while current is not None:
        path.append(current)
        current = predecessor[current]
    path.reverse()

    return distance[end], path



def euclidean_distance(coord1, coord2):
    lat1, lon1 = coord1
    lat2, lon2 = coord2

    delta_lat = lat1 - lat2
    delta_lon = lon1 - lon2

    distance = sqrt(delta_lat**2 + delta_lon**2)
    
    return distance


def heuristic(station_id, destination_id):
    station_coord = (stations[station_id]['latitude'], stations[station_id]['longitude'])
    destination_coord = (stations[destination_id]['latitude'], stations[destination_id]['longitude'])
    return euclidean_distance(station_coord, destination_coord)

def A_Star(graph, source, destination, heuristic):
    open_list = MinPriorityQueue()
    open_list.insert((0 + heuristic(source, destination), source))  # Corrected line
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
                priority = new_cost + heuristic(neighbor, destination)  # Corrected line
                open_list.insert((priority, neighbor))
                predecessors[neighbor] = current

    path = reconstruct_path(predecessors, source, destination)
    if not path:
        return predecessors, "Destination not reachable"

    return predecessors, path


def reconstruct_path(predecessors, start, end):
        if end not in predecessors:
            return []  # Path not found
        path = []
        while end is not None:
            path.append(end)
            end = predecessors.get(end)
        path.reverse()
        return path  


def parse_stations(file_path):
    stations = {}
    with open(file_path, mode='r', encoding='utf-8') as infile:
        reader = csv.DictReader(infile)
        for row in reader:
            station_id = int(row["id"])
            stations[station_id] = {
                "latitude": float(row["latitude"]),
                "longitude": float(row["longitude"]),
                "name": row["name"]
            }
    return stations


def parse_connections(file_path):
    connections = []
    with open(file_path, mode='r', encoding='utf-8') as infile:
        reader = csv.reader(infile)
        next(reader)  # Skip the header
        for row in reader:
            connection = (int(row[0]), int(row[1]), int(row[2]), float(row[3]))
            connections.append(connection)
    return connections

stations_file_path = 'london_stations.csv'
connections_file_path = 'london_connections.csv'

stations = parse_stations(stations_file_path)
connections = parse_connections(connections_file_path)




def build_graph(stations, connections):

    graph = {}

    for station1, station2, _, _ in connections:  
        if station1 not in graph:
            graph[station1] = {}
        if station2 not in graph:
            graph[station2] = {}

        coord1 = (stations[station1]['latitude'], stations[station1]['longitude'])
        coord2 = (stations[station2]['latitude'], stations[station2]['longitude'])
        distance = euclidean_distance(coord1, coord2)

        graph[station1][station2] = distance
        graph[station2][station1] = distance
    return graph


graph = build_graph(stations, connections)


def measure_performance(graph, start_id, end_id, heuristic):

    start_time = time.time()
    dijkstra(graph, start_id, end_id)
    dijkstra_time = time.time() - start_time

    start_time = time.time()
    A_Star(graph, start_id, end_id, heuristic)
    astar_time = time.time() - start_time

    return dijkstra_time, astar_time



def plot_performance_comparison(labels, dijkstra_times, astar_times, title, image_name):
    x = range(len(labels))
    width = 0.35  # Width of the bars

    fig, ax = plt.subplots(figsize=(14, 7))  # Larger figure size
    rects1 = ax.bar(x, dijkstra_times, width, label='Dijkstra')
    rects2 = ax.bar([p + width for p in x], astar_times, width, label='A*')

    ax.set_ylabel('Execution Time (seconds)', fontsize=12)
    ax.set_title(title, fontsize=14)
    ax.set_xticks([p + width / 2 for p in x])
    ax.set_xticklabels(labels, rotation=90, ha='right', fontsize=10) 
    ax.legend(fontsize=12)

    fig.tight_layout()
    plt.savefig(image_name)
    
def count_line(graph, source, destination, connections_filename):
    distance, shortest_path = dijkstra(graph, source, destination)
    
    if shortest_path == []:  
        return -1


    with open(connections_filename, mode='r') as file:
        csv_reader = csv.DictReader(file)
        connections = {(int(row['station1']), int(row['station2'])): int(row['line']) for row in csv_reader}

    num_line_transfers = 0
    current_line = None
    
    for i in range(len(shortest_path) - 1):
        station1, station2 = shortest_path[i], shortest_path[i + 1]
        line = connections.get((station1, station2)) or connections.get((station2, station1))
        
        if line is not None:
            if current_line is not None and line != current_line:
                num_line_transfers += 1
            current_line = line
    
    return num_line_transfers

def categorize_station_pairs_by_transfers(graph, stations, connections_filename):
    categories = {
        'same_line': [],
        'adjacent_transfer': [],
        'multiple_transfers': []
    }
    
    for start_id in stations:
        for end_id in stations:
            if start_id != end_id:
                num_transfers = count_line(graph, start_id, end_id, connections_filename)
                
                if num_transfers == 0:
                    categories['same_line'].append((start_id, end_id))
                elif num_transfers == 1:
                    categories['adjacent_transfer'].append((start_id, end_id))
                elif num_transfers > 1:
                    categories['multiple_transfers'].append((start_id, end_id))
    
    return categories

# The below commented code will give the station pairs that are same line, adjacent line and multiple transfers, So I hand picked some of the station pairs to compare the performance of Dijkstra's and A* algorithms
"""stations = list(range(1, 304))  # List of all station IDs
categories = categorize_station_pairs_by_transfers(graph, stations, 'london_connections.csv')

same_line_pairs = categories['same_line']
multi_transfer_pairs = categories['multiple_transfers']
adjacent_line_pairs = categories['adjacent_transfer']
"""

"""print(same_line_pairs)
print('-----------------------------------------------------------------------------------------------------------------')
print(multi_transfer_pairs)
print('-----------------------------------------------------------------------------------------------------------------')
print(adjacent_line_pairs)"""


same_pairs = [
    (1,5),
    (1,17),
    (1,30),
    (1,52),
    (2,280),
    (2, 199),
    (2,202),
    (2, 214),
    (3, 200),
    (3, 244),
    (3, 255),
    (7, 258),
    (7, 258),
    (8,22)
    
]

adjacent_pairs = [
    (1,6),
    (1,18),
    (1,53),
    (2,82),
    (2,89),
    (2,144),
    (3, 24),
    (3, 21),
    (3, 28),
    (3, 21),
    (4, 256),
    (4, 259),
    (4, 289),
    (5, 179),
    
]

multi_pairs = [
    (1,2),
    (1,10),
    (1, 39),
    (2, 8),
    (2, 15),
    (2, 16),
    (2, 17),
    (2, 40),
    (223, 21),
    (223, 27),
    (224, 170),
    (224, 172),
    (224, 183),
    (225, 98),

    
]


# No transfer (Same line)
labels_same_line = []
dijkstra_times_same_line = []
astar_times_same_line = []
station_pairs_adjacent_lines = []


print("Time Comparison between Dijkstra's and A* Algorithms: same line")
print("---------------------------------------------------------------------------------------------------")
for start_id, end_id in same_pairs:

    dijkstra_time1,_ = measure_performance(graph, start_id, end_id, lambda x, y: 0)
    astar_time1,_ = measure_performance(graph, start_id, end_id, heuristic)
    labels_same_line.append(f"{stations[start_id]['name']} to {stations[end_id]['name']}")
    dijkstra_times_same_line.append(dijkstra_time1)
    astar_times_same_line.append(astar_time1)
    print(f"Pair: {labels_same_line[-1]}, Dijkstra Time: {dijkstra_time1:.6f}, A* Time: {astar_time1:.6f}")
print("---------------------------------------------------------------------------------------------------")

plot_performance_comparison(labels_same_line, dijkstra_times_same_line, astar_times_same_line, "Performance Comparison: Same Line", "same_line.png")

# One transfer (Adjacent lines)
dijkstra_times_adjacent_lines = []
astar_times_adjacent_lines = []
labels_adjacent_lines = []
print("Time Comparison between Dijkstra's and A* Algorithms: Ajacent Line transfers")
print("---------------------------------------------------------------------------------------------------")
for start_id, end_id in adjacent_pairs:

    dijkstra_time3,_ = measure_performance(graph, start_id, end_id, lambda x, y: 0)
    astar_time3,_ = measure_performance(graph, start_id, end_id, heuristic)
    labels_adjacent_lines.append(f"{stations[start_id]['name']} to {stations[end_id]['name']}")
    dijkstra_times_adjacent_lines.append(dijkstra_time3)
    astar_times_adjacent_lines.append(astar_time3)
    print(f"Pair: {labels_adjacent_lines[-1]}, Dijkstra Time: {dijkstra_time3:.6f}, A* Time: {astar_time3:.6f}")

print("---------------------------------------------------------------------------------------------------")

plot_performance_comparison(labels_adjacent_lines, dijkstra_times_adjacent_lines, astar_times_adjacent_lines, "Performance Comparison: Adjacent lines", "adjacent_line.png")


# Multiple transfers (Multiple lines)
dijkstra_times_multiple_transfers = []
astar_times_multiple_transfers = []
labels_multiple_transfers = []  
print("Time Comparison between Dijkstra's and A* Algorithms: Multiple Line transfers")
print("---------------------------------------------------------------------------------------------------")
for start_id, end_id in multi_pairs:

    dijkstra_time2,_ = measure_performance(graph, start_id, end_id, lambda x, y: 0)
    astar_time2,_ = measure_performance(graph, start_id, end_id, heuristic)
    labels_multiple_transfers.append(f"{stations[start_id]['name']} to {stations[end_id]['name']}")
    dijkstra_times_multiple_transfers.append(dijkstra_time2)
    astar_times_multiple_transfers.append(astar_time2)
    

    print(f"Pair: {labels_multiple_transfers[-1]}, Dijkstra Time: {dijkstra_time2:.6f}, A* Time: {astar_time2:.6f}")
print("---------------------------------------------------------------------------------------------------")

plot_performance_comparison(labels_multiple_transfers, dijkstra_times_multiple_transfers, astar_times_multiple_transfers, "Performance Comparison: Multiple Transfers", "multiple_transfer.png")



# Testing the count_line function
#whether the function is working correctly or not
for i in range(len(same_pairs)): #Zero transfers
    print(count_line(graph, same_pairs[i][0], same_pairs[i][1], 'london_connections.csv'))
print('---------------------------------------------------------------------------------------------------')
for i in range(len(adjacent_pairs)): #One transfer
    print(count_line(graph, adjacent_pairs[i][0], adjacent_pairs[i][1], 'london_connections.csv'))
print('---------------------------------------------------------------------------------------------------')
for i in range(len(multi_pairs)): #Multiple transfers
    print(count_line(graph, multi_pairs[i][0], multi_pairs[i][1], 'london_connections.csv'))
print('---------------------------------------------------------------------------------------------------')










#Question 3 Line changes, I took some random pairs, and made line change function below, inorder to find the line changes between the stations.
station_pairs = [
    (1, 234),  # Short distance
    (10, 150),  # Medium distance
    (50, 200),  # Long distance
    (11, 212),  # Same Line, Short Distance
    (13, 301),  # Same Line, Long Distance
    (11, 87),   # Different Lines, No Transfers
    (3, 295),   # Different Lines, Multiple Transfers
    (117, 42),  # Heathrow Terminals 1, 2 & 3 to Canary Wharf
    (282, 247), # Wembley Park to Stratford
    (88, 299),  # Epping to Wimbledon
    (35, 192),  # Brixton to Oxford Circus
    (280, 167),  # Watford to Moorgate
    (6,70)

]


def count_line_changes(path, connections):
    station_pairs_to_lines = {}
    for conn in connections:
        station1, station2, line, _ = conn
        if (station1, station2) not in station_pairs_to_lines:
            station_pairs_to_lines[(station1, station2)] = set()
        station_pairs_to_lines[(station1, station2)].add(line)
        
        if (station2, station1) not in station_pairs_to_lines:
            station_pairs_to_lines[(station2, station1)] = set()
        station_pairs_to_lines[(station2, station1)].add(line)

    line_changes = 0
    current_line = None
    for i in range(len(path) - 1):
        station1, station2 = path[i], path[i + 1]
        possible_lines = station_pairs_to_lines.get((station1, station2), set())
        

        if current_line not in possible_lines:
  
            for line in possible_lines:
                current_line = line
                break  
            
            if i > 0:
                line_changes += 1
                
    return line_changes

labels = []
for start_id, end_id in station_pairs:
    labels.append(f"{stations[start_id]['name']} to {stations[end_id]['name']}")
line_changes_list = []
print("Line Change Comparison between Dijkstra's and A* Algorithms:")
print("---------------------------------------------------------------------------------------------------" )
for (start_id, end_id), label in zip(station_pairs, labels):
    _,dijkstra_path = dijkstra(graph, start_id, end_id)
    _, astar_path = A_Star(graph, start_id, end_id, heuristic)

   
    dijkstra_line_changes = count_line_changes(dijkstra_path, connections)
    astar_line_changes = count_line_changes(astar_path, connections)
    
    line_changes_list.append((dijkstra_line_changes, astar_line_changes))

    print(f"Pair: {label}, Dijkstra Line Changes: {dijkstra_line_changes}, A* Line Changes: {astar_line_changes}")
