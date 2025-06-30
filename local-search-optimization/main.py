import utils
import steepest_alg
import local_search
from utils import load_from_tsp
from weighted_regret_heuristic import weighted_regret_heuristic

if __name__ == "__main__":
    kroa200_matrix, kroa200_coords = load_from_tsp('datasets/kroA200.tsp')
    krob200_matrix, krob200_coords = load_from_tsp('datasets/kroB200.tsp')
    kroa200_cycle1_random, kroa200_cycle2_random, _ = utils.initialize_random_cycles(kroa200_matrix)
    krob200_cycle1_random, krob200_cycle2_random, _ = utils.initialize_random_cycles(krob200_matrix)

    utils.run_test_lab1("Weighted Regret Heuristic kroA200", kroa200_matrix, kroa200_coords, weighted_regret_heuristic)
    utils.run_test_lab1("Weighted Regret Heuristic kroB200", krob200_matrix, krob200_coords, weighted_regret_heuristic)

    utils.run_test_lab2(
        "kroA: Steepest search (original)",
        kroa200_matrix,
        kroa200_coords,
        kroa200_cycle1_random,
        kroa200_cycle2_random,
        local_search.steepest_original
    )

    utils.run_test_lab2(
        "kroB: Steepest search (original)",
        krob200_matrix,
        krob200_coords,
        krob200_cycle1_random,
        krob200_cycle2_random,
        local_search.steepest_original
    )

    utils.run_test_lab2(
        "Steepest local search (all moves) for kroA",
        kroa200_matrix,
        kroa200_coords,
        kroa200_cycle1_random,
        kroa200_cycle2_random,
        steepest_alg.steepest_full
    )

    utils.run_test_lab2(
        "Steepest local search (all moves) for kroB",
        krob200_matrix,
        krob200_coords,
        krob200_cycle1_random,
        krob200_cycle2_random,
        steepest_alg.steepest_full
    )

    utils.run_test_lab2(
        "Steepest candidate for kroA",
        kroa200_matrix,
        kroa200_coords,
        kroa200_cycle1_random,
        kroa200_cycle2_random,
        local_search.steepest_original_with_candidates
    )

    utils.run_test_lab2(
        "Steepest candidate for kroB",
        krob200_matrix,
        krob200_coords,
        krob200_cycle1_random,
        krob200_cycle2_random,
        local_search.steepest_original_with_candidates
    )