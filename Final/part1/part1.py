def dijkstra(graph, source, k):
    dist = {node: float('inf') for node in graph}
    dist[source] = 0
    path = {node: [] for node in graph}
    path[source] = [source]
    unvisited = set(graph.keys())
    relaxations = {node: 0 for node in graph}

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

        for neighbor, weight in graph[current_node].items():
            new_distance = current_distance + weight

            if new_distance < dist[neighbor]:
                dist[neighbor] = new_distance
                path[neighbor] = path[current_node] + [neighbor]
                relaxations[neighbor] += 1

    return dist, path

graph = {
    0: {1: 4, 7: 8},
    1: {0: 4, 2: 8, 7: 11},
    2: {1: 8, 3: 7, 8: 2},
    3: {2: 7, 4: 9, 5: 14},
    4: {3: 9, 5: 10},
    5: {2: 4, 3: 14, 4: 10, 6: 2},
    6: {5: 2, 7: 1, 8: 6},
    7: {0: 8, 1: 11, 6: 1, 8: 7},
    8: {2: 2, 6: 6, 7: 7},
}

# Test the function
source = 0
k = 2
distances, paths = dijkstra(graph, source, k)
print("Distances:", distances)
print("Paths:", paths)