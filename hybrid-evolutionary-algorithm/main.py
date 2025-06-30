import utils
from hae import hae
from utils import load_from_tsp

from msls import multiple_start_local_search
from ils import iterated_local_search
from lns import large_neighborhood_search

# Resztę algorytmów puszczamy na 60s. Patrzymy ile wyszło iteracji i mnożymy to *22, żeby wyszło ~22min
MAX_TIME = 60

def msls_wrapper(matrix, cycle1, cycle2):
    # 1 iteracja to średnio 10s. A więc 200 iteracji to średnio 2000s ~ 33min
    # POWIEDZMY że wyszło nam 1300s ~ 22min
    NUM_STARTS = 200
    best_cycles, best_length, total_time = multiple_start_local_search(matrix, num_starts=NUM_STARTS)
    return best_cycles, best_length, total_time, NUM_STARTS

def ils_wrapper(matrix, c1, c2):
    return iterated_local_search(matrix, c1, c2, max_time=MAX_TIME, perturbation_size=3)

def lns_wrapper_with_ln(matrix, c1, c2):
    return large_neighborhood_search(matrix, c1, c2, max_time=MAX_TIME, destroy_ratio=0.3, do_local_search=True)

def lns_wrapper_without_ln(matrix, c1, c2):
    return large_neighborhood_search(matrix, c1, c2, max_time=MAX_TIME, destroy_ratio=0.3, do_local_search=False)

def hae_wrapper_without_ln(matrix, c1, c2):
    # Wariant bez LS po rekombinacji
    return hae(matrix, max_time=MAX_TIME, use_local_search_after_recomb=False)

def hae_wrapper_with_ln(matrix, c1, c2):
    # Wariant z LS po rekombinacji
    return hae(matrix, max_time=MAX_TIME, use_local_search_after_recomb=True)

if __name__ == "__main__":
    kroa200_matrix, kroa200_coords = load_from_tsp('datasets/kroA200.tsp')
    krob200_matrix, krob200_coords = load_from_tsp('datasets/kroB200.tsp')

    kroa200_cycle1_random, kroa200_cycle2_random, _ = utils.initialize_random_cycles(kroa200_matrix)
    krob200_cycle1_random, krob200_cycle2_random, _ = utils.initialize_random_cycles(krob200_matrix)

    # # 1) MSLS: Multiple Start Local Search (10 uruchomień, każdorazowo 200 startów LS)
    # utils.run_test_lab4(
    #     "kroA: MSLS (200 starts)",
    #     kroa200_matrix,
    #     kroa200_coords,
    #     kroa200_cycle1_random,
    #     kroa200_cycle2_random,
    #     msls_wrapper
    # )
    # utils.run_test_lab4(
    #     "kroB: MSLS (200 starts)",
    #     krob200_matrix,
    #     krob200_coords,
    #     krob200_cycle1_random,
    #     krob200_cycle2_random,
    #     msls_wrapper
    # )

    # # 2) ILS: Iterated Local Search
    # utils.run_test_lab4(
    #     "kroA: ILS",
    #     kroa200_matrix,
    #     kroa200_coords,
    #     kroa200_cycle1_random,
    #     kroa200_cycle2_random,
    #     ils_wrapper
    # )

    # utils.run_test_lab4(
    #     "kroB: ILS",
    #     krob200_matrix,
    #     krob200_coords,
    #     krob200_cycle1_random,
    #     krob200_cycle2_random,
    #     ils_wrapper
    # )

    # 3) LNS: Large Neighborhood Search (NO LS)
    utils.run_test_lab4(
        "kroA: LNS (NO LS)",
        kroa200_matrix,
        kroa200_coords,
        kroa200_cycle1_random,
        kroa200_cycle2_random,
        lns_wrapper_without_ln
    )
    utils.run_test_lab4(
        "kroB: LNS (NO LS)",
        krob200_matrix,
        krob200_coords,
        krob200_cycle1_random,
        krob200_cycle2_random,
        lns_wrapper_without_ln
    )

    # # 4) LNS: Large Neighborhood Search + LS
    # utils.run_test_lab4(
    #     "kroA: LNS (WITH LS)",
    #     kroa200_matrix,
    #     kroa200_coords,
    #     kroa200_cycle1_random,
    #     kroa200_cycle2_random,
    #     lns_wrapper_with_ln
    # )
    # utils.run_test_lab4(
    #     "kroB: LNS (WITH LS)",
    #     krob200_matrix,
    #     krob200_coords,
    #     krob200_cycle1_random,
    #     krob200_cycle2_random,
    #     lns_wrapper_with_ln
    # )

    # # 5) HAE: Hybrid Algorithm with Evolution (NO LS)
    # utils.run_test_lab4(
    #     "kroA: HAE (NO LS)",
    #     kroa200_matrix,
    #     kroa200_coords,
    #     kroa200_cycle1_random,
    #     kroa200_cycle2_random,
    #     hae_wrapper_without_ln
    # )
    # utils.run_test_lab4(
    #     "kroB: HAE (NO LS)",
    #     krob200_matrix,
    #     krob200_coords,
    #     krob200_cycle1_random,
    #     krob200_cycle2_random,
    #     hae_wrapper_without_ln
    # )

    # # 6) HAE: Hybrid Algorithm with Evolution + LS
    # utils.run_test_lab4(
    #     "kroA: HAE (WITH LS)",
    #     kroa200_matrix,
    #     kroa200_coords,
    #     kroa200_cycle1_random,
    #     kroa200_cycle2_random,
    #     hae_wrapper_with_ln
    # )
    # utils.run_test_lab4(
    #     "kroB: HAE (WITH LS)",
    #     krob200_matrix,
    #     krob200_coords,
    #     krob200_cycle1_random,
    #     krob200_cycle2_random,
    #     hae_wrapper_with_ln
    # )