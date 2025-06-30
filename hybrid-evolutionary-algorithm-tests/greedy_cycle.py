import random
import numpy as np

def greedy_cycle(matrix):
    n = len(matrix)
    nodes = list(range(n))

    start_point1 = random.choice(nodes)
    start_point2 = int(np.argmax(matrix[start_point1]))

    remaining_nodes = set(nodes) - {start_point1, start_point2}

    first_cycle = [start_point1, int(np.argmin(matrix[start_point1]))]
    second_cycle = [start_point2, int(np.argmin(matrix[start_point2]))]

    remaining_nodes -= {first_cycle[1], second_cycle[1]}

    target_len1 = (n + 1) // 2
    target_len2 = n // 2

    while remaining_nodes:
        best_cost = float('inf')
        best_node = None
        best_place = None
        best_cycle = None

        for node in remaining_nodes:
            for cycle, target_len in [(first_cycle, target_len1), (second_cycle, target_len2)]:
                if len(cycle) >= target_len + 1:
                    continue
                for i in range(0, len(cycle)):
                    prev = cycle[i - 1]
                    next = cycle[i]
                    cost = matrix[prev][node] + matrix[node][next] - matrix[prev][next]
                    if cost < best_cost:
                        best_cost = cost
                        best_node = node
                        best_place = i
                        best_cycle = cycle

        if best_node is not None:
            best_cycle.insert(best_place, best_node)
            remaining_nodes.remove(best_node)

    return first_cycle, second_cycle