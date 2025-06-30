from local_search import steepest_descent
from utils import cycle_length, target_function
import time
import numpy as np

def steepest_full(distance_matrix: np.ndarray, cycle1: list[int], cycle2: list[int]) -> tuple[
        tuple[list[int], list[int]],
        int,
        float]:
    start_time = time.time()
    initial_cycles = (cycle1, cycle2)
    result = steepest_descent(initial_cycles, distance_matrix)
    total_cost = target_function(result[0], result[1], distance_matrix)
    duration = time.time() - start_time
    return result, total_cost, duration