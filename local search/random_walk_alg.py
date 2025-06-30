from swaps import swap_nodes_within_cycle, swap_edges_within_cycle, swap_nodes_between_cycles
from utils import load_from_tsp, initialize_random_cycles, cycle_length
import random
import time

def random_walk(distance_matrix, first_cycle, second_cycle, max_time):
    best_solution = (first_cycle, second_cycle)
    best_length = cycle_length(first_cycle, distance_matrix) + cycle_length(second_cycle, distance_matrix)

    start_time = time.time()
    while time.time() - start_time < max_time:
        move_type = random.choice(['within', 'between', 'edges'])

        if move_type == 'within':
            cycle_choice = random.choice(['first', 'second'])
            if cycle_choice == 'first':
                new_cycle, _ = random.choice(list(swap_nodes_within_cycle(first_cycle, distance_matrix)))
                first_cycle = new_cycle
            else:
                new_cycle, _ = random.choice(list(swap_nodes_within_cycle(second_cycle, distance_matrix)))
                second_cycle = new_cycle

        elif move_type == 'between':
            new_first, new_second, _ = random.choice(
                list(swap_nodes_between_cycles(first_cycle, second_cycle, distance_matrix)))
            first_cycle, second_cycle = new_first, new_second

        elif move_type == 'edges':
            cycle_choice = random.choice(['first', 'second'])
            if cycle_choice == 'first':
                new_cycle, _ = random.choice(list(swap_edges_within_cycle(first_cycle, distance_matrix)))
                first_cycle = new_cycle
            else:
                new_cycle, _ = random.choice(list(swap_edges_within_cycle(second_cycle, distance_matrix)))
                second_cycle = new_cycle

        current_length = cycle_length(first_cycle, distance_matrix) + cycle_length(second_cycle, distance_matrix)
        if current_length < best_length:
            best_solution = (first_cycle, second_cycle)
            best_length = current_length

    return best_solution, best_length