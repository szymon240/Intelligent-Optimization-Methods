import numpy as np
import matplotlib.pyplot as plt
import time
from utils import (
    load_from_tsp,
    target_function
)
from local_search import steepest_original
from hae import hae
from greedy_cycle import greedy_cycle

import matplotlib
matplotlib.use('TkAgg')

# --- PARAMETRY ---
NUM_LOCAL_OPTIMA = 200
DATASET_PATH = 'datasets/kroA200.tsp'

# --- GENEROWANIE LOKALNYCH OPTIMUM ---
def generate_local_optima(distance_matrix):
    local_optima = []
    for i in range(NUM_LOCAL_OPTIMA):
        print(f"Generuję lokalne optimum {i+1}/{NUM_LOCAL_OPTIMA}")
        c1, c2 = greedy_cycle(distance_matrix)
        (c1, c2), _, _ = steepest_original(distance_matrix, c1, c2)
        cost = target_function(c1, c2, distance_matrix)
        local_optima.append((c1, c2, cost))
    return local_optima

# --- NAJLEPSZE ROZWIĄZANIE HAE ---
def find_best_solution(distance_matrix):
    (c1, c2), cost, _, _ = hae(distance_matrix)
    print(f"Najlepszy koszt (HAE): {cost}")
    return c1, c2, cost

# --- MIARY PODOBIEŃSTWA ---
def pair_similarity(sol_a, sol_b):
    def single(a, b):
        c1a, c2a = a
        c1b, c2b = b
        cluster_a = {v:1 for v in c1a[1:-1]}
        cluster_a.update({v:2 for v in c2a[1:-1]})
        cluster_b = {v:1 for v in c1b[1:-1]}
        cluster_b.update({v:2 for v in c2b[1:-1]})
        verts = list(set(cluster_a.keys()) & set(cluster_b.keys()))
        count = 0
        for i in range(len(verts)):
            for j in range(i+1, len(verts)):
                u, v = verts[i], verts[j]
                if cluster_a[u] == cluster_a[v] and cluster_b[u] == cluster_b[v]:
                    count += 1
        return count

    # oryginalne dopasowanie
    sim1 = single(sol_a, sol_b)
    # dopasowanie z zamianą cykli w sol_b
    swapped = (sol_b[1], sol_b[0])
    sim2 = single(sol_a, swapped)
    return max(sim1, sim2)


def edge_similarity(sol_a, sol_b):
    def get_edges(cycle_pair):
        e = set()
        for cyc in cycle_pair:
            for u, v in zip(cyc, cyc[1:]):
                e.add(tuple(sorted((u, v))))
        return e

    edges_a = get_edges(sol_a)
    edges_b = get_edges(sol_b)

    return len(edges_a & edges_b)


# --- ANALIZA I WYKRESY ---
def main():
    dist, coords = load_from_tsp(DATASET_PATH)

    print("1) Generuję lokalne optima...")
    local_opts = generate_local_optima(dist)

    print("2) Znajduję najlepsze rozwiązanie...")
    best_c1, best_c2, best_cost = find_best_solution(dist)
    best = (best_c1, best_c2)

    costs, sim_pair_best, sim_edge_best, sim_pair_mean, sim_edge_mean = ([], [], [], [], [])

    print("3) Obliczam podobieństwa...")
    for c1, c2, cost in local_opts:
        sim_pair_best.append(pair_similarity((c1, c2), best))
        sim_edge_best.append(edge_similarity((c1, c2), best))
        tp = te = 0
        for d1, d2, _ in local_opts:
            if d1==c1 and d2==c2: continue
            tp += pair_similarity((c1,c2),(d1,d2))
            te += edge_similarity((c1,c2),(d1,d2))
        sim_pair_mean.append(tp / (len(local_opts)-1))
        sim_edge_mean.append(te / (len(local_opts)-1))
        costs.append(cost)

    print("4) Korelacje:")
    print(f"  koszt vs pair_mean: {np.corrcoef(costs, sim_pair_mean)[0,1]:.4f}")
    print(f"  koszt vs edge_mean: {np.corrcoef(costs, sim_edge_mean)[0,1]:.4f}")
    print(f"  koszt vs pair_best: {np.corrcoef(costs, sim_pair_best)[0,1]:.4f}")
    print(f"  koszt vs edge_best: {np.corrcoef(costs, sim_edge_best)[0,1]:.4f}")

    # wykresy
    for ys, label, title in [
        (sim_pair_mean, 'Śr. par', 'Cost vs Mean Pair'),
        (sim_edge_mean, 'Śr. edge', 'Cost vs Mean Edge'),
        (sim_pair_best, 'Par do best', 'Cost vs Best Pair'),
        (sim_edge_best, 'Edge do best', 'Cost vs Best Edge')
    ]:
        plt.figure(); plt.scatter(costs, ys)
        plt.xlabel('Koszt'); plt.ylabel(label); plt.title(title)
        plt.show()

if __name__ == '__main__':
    start=time.time(); main(); print(f"Total: {time.time()-start:.2f}s")
