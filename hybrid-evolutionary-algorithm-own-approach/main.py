import utils
from hae import hae
from utils import load_from_tsp

MAX_TIME = 60

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

    # 6) HAE: Hybrid Algorithm with Evolution + LS
    utils.run_test_lab4(
        "kroA: HAE (WITH LS)",
        kroa200_matrix,
        kroa200_coords,
        kroa200_cycle1_random,
        kroa200_cycle2_random,
        hae_wrapper_with_ln
    )
    # utils.run_test_lab4(
    #     "kroB: HAE (WITH LS)",
    #     krob200_matrix,
    #     krob200_coords,
    #     krob200_cycle1_random,
    #     krob200_cycle2_random,
    #     hae_wrapper_with_ln
    # )