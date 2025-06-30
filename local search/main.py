from greedy_alg import greedy_within_cycle, greedy_between_cycles
from random_walk_alg import random_walk
from steepest_alg import steepest_within_cycle_and_edges, steepest_between_cycles
from utils import load_from_tsp
import utils

if __name__ == "__main__":
    kroa200_matrix, kroa200_coords = load_from_tsp('datasets/kroA200.tsp')
    krob200_matrix, krob200_coords = load_from_tsp('datasets/kroB200.tsp')
    kroa200_cycle1, kroa200_cycle2, krob200_cycle1, krob200_cycle2 = utils.load_regret_results('datasets/results.json')
    kroa200_cycle1_random, kroa200_cycle2_random, _ = utils.initialize_random_cycles(kroa200_matrix)
    krob200_cycle1_random, krob200_cycle2_random, _ = utils.initialize_random_cycles(krob200_matrix)


    # Run algorithms on kroA200 dataset
    kroa200_times = []

    kroa200_times.append(
        utils.run_test(
            "kroA: Greedy search (node and edge swaps)",
            kroa200_matrix,
            kroa200_coords,
            kroa200_cycle1,
            kroa200_cycle2,
            greedy_within_cycle
        )
    )

    kroa200_times.append(
        utils.run_test(
            "kroA: Greedy search (node and edge swaps) random",
            kroa200_matrix,
            kroa200_coords,
            kroa200_cycle1_random,
            kroa200_cycle2_random,
            greedy_within_cycle
        )
    )

    kroa200_times.append(
        utils.run_test(
            "kroA: Greedy search (node swap between cycles)",
            kroa200_matrix,
            kroa200_coords,
            kroa200_cycle1,
            kroa200_cycle2,
            greedy_between_cycles
        )
    )

    kroa200_times.append(
        utils.run_test(
            "kroA: Greedy search (node swap between cycles) random",
            kroa200_matrix,
            kroa200_coords,
            kroa200_cycle1_random,
            kroa200_cycle2_random,
            greedy_between_cycles
        )
    )

    kroa200_times.append(
        utils.run_test(
            "kroA: Steepest local search (node and edge swaps within cycle)",
            kroa200_matrix,
            kroa200_coords,
            kroa200_cycle1,
            kroa200_cycle2,
            steepest_within_cycle_and_edges
        )
    )

    kroa200_times.append(
        utils.run_test(
            "kroA: Steepest local search (node and edge swaps within cycle) random",
            kroa200_matrix,
            kroa200_coords,
            kroa200_cycle1_random,
            kroa200_cycle2_random,
            steepest_within_cycle_and_edges
        )
    )

    kroa200_times.append(
        utils.run_test(
            "kroA: Steepest local search (node swaps between cycles)",
            kroa200_matrix,
            kroa200_coords,
            kroa200_cycle1,
            kroa200_cycle2,
            steepest_between_cycles
        )
    )

    kroa200_times.append(
        utils.run_test(
            "kroA: Steepest local search (node swaps between cycles) random",
            kroa200_matrix,
            kroa200_coords,
            kroa200_cycle1_random,
            kroa200_cycle2_random,
            steepest_between_cycles
        )
    )

    min_time = min(kroa200_times)
    utils.run_test(
        "kroA: Random walk search",
        kroa200_matrix,
        kroa200_coords,
        kroa200_cycle1,
        kroa200_cycle2,
        random_walk,
        min_time
    )

    # Run algorithms on kroB200 dataset
    krob200_times = []

    krob200_times.append(
        utils.run_test(
            "kroB: Greedy search (node and edge swaps)",
            krob200_matrix,
            krob200_coords,
            krob200_cycle1,
            krob200_cycle2,
            greedy_within_cycle
        )
    )

    krob200_times.append(
        utils.run_test(
            "kroB: Greedy search (node and edge swaps) random",
            krob200_matrix,
            krob200_coords,
            krob200_cycle1_random,
            krob200_cycle2_random,
            greedy_within_cycle
        )
    )

    krob200_times.append(
        utils.run_test(
            "kroB: Greedy search (node swap between cycles)",
            krob200_matrix,
            krob200_coords,
            krob200_cycle1,
            krob200_cycle2,
            greedy_between_cycles
        )
    )

    krob200_times.append(
        utils.run_test(
            "kroB: Greedy search (node swap between cycles) random",
            krob200_matrix,
            krob200_coords,
            krob200_cycle1_random,
            krob200_cycle2_random,
            greedy_between_cycles
        )
    )

    krob200_times.append(
        utils.run_test(
            "kroB: Steepest local search (node and edge swaps within cycle)",
            krob200_matrix,
            krob200_coords,
            krob200_cycle1,
            krob200_cycle2,
            steepest_within_cycle_and_edges
        )
    )

    krob200_times.append(
        utils.run_test(
            "kroB: Steepest local search (node and edge swaps within cycle) random",
            krob200_matrix,
            krob200_coords,
            krob200_cycle1_random,
            krob200_cycle2_random,
            steepest_within_cycle_and_edges
        )
    )

    krob200_times.append(
        utils.run_test(
            "kroB: Steepest local search (node swaps between cycles)",
            krob200_matrix,
            krob200_coords,
            krob200_cycle1,
            krob200_cycle2,
            steepest_between_cycles
        )
    )

    krob200_times.append(
        utils.run_test(
            "kroB: Steepest local search (node swaps between cycles) random",
            krob200_matrix,
            krob200_coords,
            krob200_cycle1_random,
            krob200_cycle2_random,
            steepest_between_cycles
        )
    )

    min_time = min(krob200_times)
    utils.run_test(
        "kroB: Random walk search",
        krob200_matrix,
        krob200_coords,
        krob200_cycle1,
        krob200_cycle2,
        random_walk,
        min_time
    )