import tsplib95
import math
import numpy as np
import random
from typing import Callable, Tuple, List, Set
import matplotlib.pyplot as plt
import json

def convert_tsp_to_array(tsp_coords: tsplib95.fields.IndexedCoordinatesField, size: int) -> np.ndarray:
    list_coords = []
    for i in range(1, size + 1):
        list_coords.append(tsp_coords[i])
    return np.array(list_coords)


def load_from_tsp(path: str) -> Tuple[np.ndarray, np.ndarray]:
    problem = tsplib95.load(path)
    coords = problem.node_coords
    n = problem.dimension
    distance_matrix = np.zeros((n, n), dtype=int)

    for i in range(1, n + 1):
        for j in range(1, n + 1):
            if i == j:
                distance_matrix[i - 1, j - 1] = 0
            else:
                x1, y1 = coords[i]
                x2, y2 = coords[j]
                distance = math.hypot(x1 - x2, y1 - y2)
                distance_matrix[i - 1, j - 1] = round(distance)

    coords_matrix = convert_tsp_to_array(coords, n)
    return distance_matrix, coords_matrix

def initialize_random_cycles(distance_matrix: np.ndarray) -> tuple[list[int], list[int], set[int]]:
    n = len(distance_matrix)
    nodes = list(range(n))

    start_point1 = random.choice(nodes)
    start_point2 = int(np.argmax(distance_matrix[start_point1]))

    remaining_nodes = list(set(nodes) - {start_point1, start_point2})
    random.shuffle(remaining_nodes)

    first_cycle_size = (n + 1) // 2
    second_cycle_size = n // 2

    first_cycle = [start_point1] + remaining_nodes[:first_cycle_size - 1]
    second_cycle = [start_point2] + remaining_nodes[first_cycle_size - 1:]

    first_cycle.append(start_point1)
    second_cycle.append(start_point2)

    return first_cycle, second_cycle, set(nodes)


def cycle_length(cycle, matrix):
    length = 0
    for i in range(1, len(cycle)):
        length += matrix[cycle[i - 1]][cycle[i]]
    return length

def target_function(cycle1: list[int], cycle2: list[int], matrix: np.ndarray) -> int:
    return cycle_length(cycle1, matrix) + cycle_length(cycle2, matrix)

def load_regret_results(results_path: str) -> Tuple[list[int], list[int], list[int], list[int]]:
    with open(results_path, 'r') as f:
        loaded_results = json.load(f)

    kroa200_cycle1_loaded = loaded_results["kroA200_cycle1"]
    kroa200_cycle2_loaded = loaded_results["kroA200_cycle2"]
    krob200_cycle1_loaded = loaded_results["kroB200_cycle1"]
    krob200_cycle2_loaded = loaded_results["kroB200_cycle2"]
    
    return kroa200_cycle1_loaded, kroa200_cycle2_loaded, krob200_cycle1_loaded, krob200_cycle2_loaded

def experiment(
            matrix: np.ndarray,
            cycle1: list[int],
            cycle2: list[int],
            alg: Callable[[np.ndarray, list[int], list[int], float], tuple[tuple[list[int], list[int]], int, float]],
            min_time: float | None = None,
            runs = 100
            ):
    lengths = []
    times = []
    solutions = []

    for _ in range(runs):
        if min_time is None:
            solution, length, time = alg(matrix, cycle1, cycle2)
        else:
            solution, length = alg(matrix, cycle1, cycle2, min_time)
            time = min_time
        lengths.append(length)
        times.append(time)
        solutions.append(solution)
    
    avg_length = sum(lengths) / runs
    avg_times = sum(times) / runs
    best_solution = solutions[lengths.index(min(lengths))]

    return avg_length, min(lengths), max(lengths), avg_times, min(times), max(times), best_solution

def run_test(
            alg_name: str,
            matrix: np.ndarray,
            coords: np.ndarray,
            cycle1: list[int],
            cycle2: list[int],
            alg: Callable[[np.ndarray, list[int], list[int], float], tuple[tuple[list[int], list[int]], int, float]],
            min_time_alg: float | None = None
            ) -> float:
    init_length = target_function(cycle1, cycle2, matrix)

    avg_length, min_length, max_length, avg_time, min_time, max_time, best_solution = experiment(matrix, cycle1, cycle2, alg, min_time_alg)

    print(f"\n{alg_name}:")
    print(f"Initial solution length: {init_length}")
    print(f"Average solution length: {avg_length}")
    print(f"Best solution length: {min_length}")
    print(f"Worst solution length: {max_length}")
    print(f"Average execution time: {avg_time:.4f} seconds")
    print(f"Best execution time: {min_time:.4f} seconds")
    print(f"Worst execution time: {max_time:.4f} seconds")

    visualize_cycles(cycle1, cycle2, coords, f"{alg_name} - before", init_length, save=True)
    visualize_cycles(best_solution[0], best_solution[1], coords, alg_name, min_length, save=True)

    return min_time

def visualize_cycles(cycle1: list[int], cycle2: list[int], positions: np.ndarray, algorithm_name: str, cycles_length: int, save = False):
    plt.figure(figsize=(16, 9))
    
    def plot_cycle(cycle, color):
        x = [positions[i, 0] for i in cycle] + [positions[cycle[0], 0]]
        y = [positions[i, 1] for i in cycle] + [positions[cycle[0], 1]]
        plt.plot(x, y, marker='o', color=color, linestyle='-')
    
    plot_cycle(cycle1, 'blue')
    plot_cycle(cycle2, 'red')
    
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.title(f"{algorithm_name} - Length: {cycles_length}")
    
    if save:
        plt.savefig(f"visualizations/{algorithm_name}.png")
    else:
        plt.show()
    
    plt.close()