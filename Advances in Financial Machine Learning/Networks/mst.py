from collections import defaultdict
import heapq
class MinimumSpanningTree:
    def __init__(self, vertices, edges):
        self.vertices = vertices
        self.edges = edges
        self.parent = {}
        self.rank = {}
        self.init_table()
        self.__graph_edges = []
        self.__graph_map = defaultdict(dict)

    def init_table(self):
        for vertex in self.vertices:
            self.parent[vertex] = vertex
            self.rank[vertex] = 0

    @property
    def graph_edges(self):
        return self.__graph_edges

    @property
    def graph_map(self):
        return self.__graph_map

    def find_parent(self, vertex):
        if self.parent[vertex] != vertex:
            self.parent[vertex] = self.find_parent(self.parent[vertex])

        return self.parent[vertex]

    def check_connected(self, vertex_1, vertex_2):
        parent1 = self.find_parent(vertex_1)
        parent2 = self.find_parent(vertex_2)

        return parent1 == parent2

    def union_parent(self, vertex_1, vertex_2):
        parent1 = self.find_parent(vertex_1)
        parent2 = self.find_parent(vertex_2)

        if parent1 != parent2:
            self.parent[parent2] = parent1

    def build(self):
        for edge in self.edges:
            _, vertex_1, vertex_2 = edge
            if not self.check_connected(vertex_1, vertex_2):
                self.union_parent(vertex_1, vertex_2)
                self.__graph_edges.append(edge)

                if len(self.__graph_edges) == len(self.vertices)-1:
                    break
        graph_edges = self.__graph_edges.copy()

        for vertex in self.vertices:

            rm_ix = []
            if graph_edges:
                for i, (distance, vertex_1, vertex_2) in enumerate(graph_edges):
                    if vertex in (vertex_1, vertex_2):
                        rm_ix.append(i)
                        if vertex == vertex_1:
                            self.__graph_map[vertex][vertex_2] = distance
                        else:
                            self.__graph_map[vertex][vertex_1] = distance
        print(f"The building of tree is completed")
        return self.__graph_edges

    def find_path(self, start_vertex, end_vertex):
        graph = self.graph_map.copy()
        distances = {vertex: [float('inf'), start_vertex] for vertex in self.vertices}
        distances[start_vertex] = [0, start_vertex]
        queue = []
        heapq.heappush(queue, [distances[start_vertex][0], start_vertex])

        while queue:
            current_distance, current_vertex = heapq.heappop(queue)
            if distances[current_vertex][0] < current_distance:
                continue

            for adjacent, weight in graph[current_vertex].items():
                distance = current_distance + weight
                if distance < distances[adjacent][0]:
                    distances[adjacent] = [distance, current_vertex]
                    heapq.heappush(queue, [distance, adjacent])
        path = end_vertex
        path_record = []

        if distances[end_vertex][1] == start_vertex:
            path_record.append(end_vertex)
        else:
            while distances[path][1] != start_vertex:
                path_record.append(path)
                path = distances[path][1]
            path_record.append(path)
        path_record.append(start_vertex)
        path_record = path_record[::-1]
        distance = distances[end_vertex][0]
        return path_record, distance