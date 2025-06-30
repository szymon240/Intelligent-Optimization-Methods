import numpy as np

def initialize_cycles(distance_matrix: np.ndarray) -> tuple[list[int], list[int], set[int]]:
    n = len(distance_matrix)
    nodes = set(range(n))
    
    first_node = np.random.choice(list(nodes))
    nodes.remove(first_node)
    
    second_node = min(nodes, key=lambda x: distance_matrix[first_node][x])
    nodes.remove(second_node)
    
    cycle1 = [first_node]
    cycle2 = [second_node]
    
    return cycle1, cycle2, nodes

def calculate_regret(distance_matrix: np.ndarray, cycle: list[int], candidate: int) -> tuple[float, float, int]:
    best_increase = float('inf')
    second_best_increase = float('inf')
    best_pos = -1
    
    for i in range(len(cycle)):
        j = (i + 1) % len(cycle)
        increase = (
            distance_matrix[cycle[i]][candidate] +
            distance_matrix[candidate][cycle[j]] -
            distance_matrix[cycle[i]][cycle[j]]
        )
        
        if increase < best_increase:
            second_best_increase = best_increase
            best_increase = increase
            best_pos = j
        elif increase < second_best_increase:
            second_best_increase = increase
    
    regret = second_best_increase - best_increase
    return regret, best_increase, best_pos

def insert_into_cycle(distance_matrix: np.ndarray, cycle: list[int], candidate: int):
    _, _, pos = calculate_regret(distance_matrix, cycle, candidate)
    cycle.insert(pos, candidate)