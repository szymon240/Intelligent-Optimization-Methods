from swaps import swap_nodes_within_cycle, swap_edges_within_cycle, swap_nodes_between_cycles
from utils import load_from_tsp, initialize_random_cycles, cycle_length
import random
import time

def greedy_within_cycle(distance_matrix, first_cycle, second_cycle, randomize=True):
    start_time = time.time()
    best_length = cycle_length(first_cycle, distance_matrix) + cycle_length(second_cycle, distance_matrix)

    improved = True
    while improved:
        improved = False

        # Sprawdź zamiany w obrębie pierwszego cyklu (wymiany wierzchołków)
        swaps = list(swap_nodes_within_cycle(first_cycle, distance_matrix))
        if randomize:
            random.shuffle(swaps)
        for new_cycle, delta in swaps:
            if best_length + delta < best_length:
                first_cycle = new_cycle
                best_length += delta
                improved = True
                break

        # Sprawdź zamiany w obrębie drugiego cyklu (wymiany wierzchołków)
        swaps = list(swap_nodes_within_cycle(second_cycle, distance_matrix))
        if randomize:
            random.shuffle(swaps)
        for new_cycle, delta in swaps:
            if best_length + delta < best_length:
                second_cycle = new_cycle
                best_length += delta
                improved = True
                break

        # Sprawdź zamiany krawędzi w obrębie pierwszego cyklu (2-opt move)
        swaps = list(swap_edges_within_cycle(first_cycle, distance_matrix))
        if randomize:
            random.shuffle(swaps)
        for new_cycle, delta in swaps:
            if best_length + delta < best_length:
                first_cycle = new_cycle
                best_length += delta
                improved = True
                break

        # Sprawdź zamiany krawędzi w obrębie drugiego cyklu (2-opt move)
        swaps = list(swap_edges_within_cycle(second_cycle, distance_matrix))
        if randomize:
            random.shuffle(swaps)
        for new_cycle, delta in swaps:
            if best_length + delta < best_length:
                second_cycle = new_cycle
                best_length += delta
                improved = True
                break

    execution_time = time.time() - start_time
    return (first_cycle, second_cycle), best_length, execution_time


def greedy_between_cycles(distance_matrix, first_cycle, second_cycle, randomize=True):
    start_time = time.time()  # Start timer
    best_length = cycle_length(first_cycle, distance_matrix) + cycle_length(second_cycle, distance_matrix)

    improved = True
    while improved:
        improved = False

        # Sprawdź zamiany między cyklami (wymiany wierzchołków)
        swaps = list(swap_nodes_between_cycles(first_cycle, second_cycle, distance_matrix))
        if randomize:
            random.shuffle(swaps)
        for new_first, new_second, delta in swaps:
            if best_length + delta < best_length:
                first_cycle, second_cycle = new_first, new_second
                best_length += delta
                improved = True
                break

        # Sprawdź zamiany w obrębie pierwszego cyklu (wymiany wierzchołków)
        swaps = list(swap_nodes_within_cycle(first_cycle, distance_matrix))
        if randomize:
            random.shuffle(swaps)
        for new_cycle, delta in swaps:
            if best_length + delta < best_length:
                first_cycle = new_cycle
                best_length += delta
                improved = True
                break

        # Sprawdź zamiany w obrębie drugiego cyklu (wymiany wierzchołków)
        swaps = list(swap_nodes_within_cycle(second_cycle, distance_matrix))
        if randomize:
            random.shuffle(swaps)
        for new_cycle, delta in swaps:
            if best_length + delta < best_length:
                second_cycle = new_cycle
                best_length += delta
                improved = True
                break

        # Sprawdź zamiany krawędzi w obrębie pierwszego cyklu (2-opt move)
        swaps = list(swap_edges_within_cycle(first_cycle, distance_matrix))
        if randomize:
            random.shuffle(swaps)
        for new_cycle, delta in swaps:
            if best_length + delta < best_length:
                first_cycle = new_cycle
                best_length += delta
                improved = True
                break

        # Sprawdź zamiany krawędzi w obrębie drugiego cyklu (2-opt move)
        swaps = list(swap_edges_within_cycle(second_cycle, distance_matrix))
        if randomize:
            random.shuffle(swaps)
        for new_cycle, delta in swaps:
            if best_length + delta < best_length:
                second_cycle = new_cycle
                best_length += delta
                improved = True
                break

    execution_time = time.time() - start_time
    return (first_cycle, second_cycle), best_length, execution_time