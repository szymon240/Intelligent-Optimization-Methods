from weighted_regret_heurestic import weighted_regret_heuristic
import utils
import json

if __name__ == "__main__":
    kroa200_matrix, kroa200_coords = utils.load_from_tsp('datasets/kroA200.tsp')
    krob200_matrix, krob200_coords = utils.load_from_tsp('datasets/kroB200.tsp')

    kroa200_cycle1, kroa200_cycle2 = weighted_regret_heuristic(kroa200_matrix)
    krob200_cycle1, krob200_cycle2 = weighted_regret_heuristic(krob200_matrix)

    results = {
        "kroA200_cycle1": list(map(int, kroa200_cycle1)),
        "kroA200_cycle2": list(map(int, kroa200_cycle2)),
        "kroB200_cycle1": list(map(int, krob200_cycle1)),
        "kroB200_cycle2": list(map(int, krob200_cycle2)),
    }

    with open('datasets/results.json', 'w') as f:
        json.dump(results, f)
    
    utils.visualize_cycles(kroa200_cycle1, kroa200_cycle2, kroa200_coords, "Weighted Regret Heuristic", 0, 0, 0, save=False)
    utils.visualize_cycles(krob200_cycle1, krob200_cycle2, krob200_coords, "Weighted Regret Heuristic", 0, 0, 0, save=False)

    with open('results.json', 'r') as f:
        loaded_results = json.load(f)

    kroa200_cycle1_loaded = loaded_results["kroA200_cycle1"]
    kroa200_cycle2_loaded = loaded_results["kroA200_cycle2"]
    krob200_cycle1_loaded = loaded_results["kroB200_cycle1"]
    krob200_cycle2_loaded = loaded_results["kroB200_cycle2"]