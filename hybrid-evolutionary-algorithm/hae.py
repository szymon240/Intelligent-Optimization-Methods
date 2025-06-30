import random
import time
import local_search
from utils import (
    calculate_regret,
    target_function,
    initialize_random_cycles
)

# --- Configuration ---
ELITE_SIZE = 5
MAX_TIME = 65

# --- Repair function from LNS ---
def repair_solution(distance_matrix, cycle1, cycle2, removed_nodes):
    #print(f"[Repair] Repairing with {len(removed_nodes)} removed nodes...")
    for node in removed_nodes:
        regret1, increase1, pos1 = calculate_regret(distance_matrix, cycle1, node)
        regret2, increase2, pos2 = calculate_regret(distance_matrix, cycle2, node)
        if regret1 > regret2 or (regret1 == regret2 and increase1 < increase2):
            cycle1.insert(pos1, node)
        else:
            cycle2.insert(pos2, node)
    if cycle1[-1] != cycle1[0]:
        cycle1.append(cycle1[0])
    if cycle2[-1] != cycle2[0]:
        cycle2.append(cycle2[0])
    return cycle1, cycle2

# --- HAE components ---
def generate_initial_population(distance_matrix):
    print("[Init] Generating initial population...")
    population = []
    for i in range(ELITE_SIZE):
        print(f"[Init] Generating individual {i+1}/{ELITE_SIZE}")
        c1, c2, _ = initialize_random_cycles(distance_matrix)
        (c1, c2), length, _ = local_search.steepest_original(distance_matrix, c1, c2)
        population.append((c1, c2, length))
    population.sort(key=lambda x: x[2])
    print("[Init] Initial population ready.")
    return population


def select_parents(population):
    i, j = random.sample(range(len(population)), 2)
    #print(f"[Select] Selected parents {i} and {j}")
    return population[i], population[j]


def recombine(parent1, parent2, distance_matrix):
    #print("[Recombine] Starting recombination...")
    p1_c1, p1_c2, _ = parent1
    p2_c1, p2_c2, _ = parent2
    y1 = p1_c1.copy()
    y2 = p1_c2.copy()
    # Zbiór krawędzi z drugiego rodzica
    edges2 = set()
    for cycle in (p2_c1, p2_c2):
        for u, v in zip(cycle, cycle[1:]):
            edges2.add((u, v)); edges2.add((v, u))

    def prune_cycle(cycle):
        new = [cycle[0]]
        for u, v in zip(cycle, cycle[1:]):
            if (u, v) in edges2:
                new.append(v)
        if new[-1] != new[0]: new.append(new[0])
        return new

    y1 = prune_cycle(y1)
    y2 = prune_cycle(y2)
    all_nodes = set(p1_c1[1:-1] + p1_c2[1:-1])
    kept = set(y1[1:-1] + y2[1:-1])
    removed = list(all_nodes - kept)
    #print(f"[Recombine] Removed {len(removed)} nodes during pruning.")
    y1, y2 = repair_solution(distance_matrix, y1, y2, removed)
    #print("[Recombine] Recombination finished.")
    return y1, y2


def is_unique_and_better(offspring, population):
    _, _, length = offspring
    if length >= population[-1][2]: return False
    for c1, c2, l in population:
        if l == length and c1 == offspring[0] and c2 == offspring[1]:
            return False
    return True


def hae(distance_matrix, max_time=MAX_TIME, use_local_search_after_recomb=True):
    print("[HAE] Starting algorithm...")
    population = generate_initial_population(distance_matrix)
    start = time.time(); iter_count = 0
    while time.time() - start < max_time:
        iter_count += 1 # print(f"[HAE] Iter {iter_count}")
        p1, p2 = select_parents(population)
        y1, y2 = recombine(p1, p2, distance_matrix)
        if use_local_search_after_recomb:
            #print("[HAE] Local search...")
            (y1, y2), length, _ = local_search.steepest_original(distance_matrix, y1, y2)
        else:
            #print("[HAE] No local search.")
            length = target_function(y1, y2, distance_matrix)
        offspring = (y1, y2, length)
        if is_unique_and_better(offspring, population):
            #print(f"[HAE] Replaced worst with {length:.2f}")
            population[-1] = offspring; population.sort(key=lambda x: x[2])
        #else:
            #print("[HAE] Discarded offspring")
    best = population[0]; total = time.time() - start
    print(f"[HAE] Done iters={iter_count}, time={total:.2f}s, best={best[2]:.2f}")
    return (best[0], best[1]), best[2], total, iter_count