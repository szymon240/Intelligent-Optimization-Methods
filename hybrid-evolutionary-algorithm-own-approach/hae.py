import random
import time
import local_search
from utils import (
    calculate_regret,
    target_function,
    initialize_random_cycles
)
from weighted_regret_heuristic import weighted_regret_heuristic  # import weighted regret function

# --- Configuration ---
ELITE_SIZE = 5
MAX_TIME = 65
TOURNAMENT_SIZE = 3  # liczba uczestników w turnieju

# --- Global memory of edges ---
EDGE_FREQ = {}  # maps (u,v) -> count across population

# --- Repair function from LNS ---
def repair_solution(distance_matrix, cycle1, cycle2, removed_nodes):
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

# --- Tournament selection ---
def tournament_select(population, k=TOURNAMENT_SIZE):
    # losuj k rozwiązań i wybierz najlepsze (najniższa wartość) z nich
    contestants = random.sample(population, k)
    # każdy osobnik to (c1, c2, length)
    winner = min(contestants, key=lambda ind: ind[2])
    return winner

# --- HAE components ---
from lns import destroy_solution  # import LNS destroy

# --- LNS perturbation parameters ---
PERTURB_INTERVAL = 10     # co ile iteracji wywołać perturbację
DESTROY_RATIO = 0.3       # odsetek węzłów do usunięcia w perturbacji
def generate_initial_population(distance_matrix):
    print("[Init] Generating initial population with weighted regret heuristic...")
    global EDGE_FREQ
    population = []
    for i in range(ELITE_SIZE):
        print(f"[Init] Generating individual {i+1}/{ELITE_SIZE}")
        c1, c2 = weighted_regret_heuristic(distance_matrix)
        (c1, c2), length, _ = local_search.steepest_original(distance_matrix, c1, c2)
        population.append((c1, c2, length))
    population.sort(key=lambda x: x[2])
    # initialize EDGE_FREQ from initial population
    EDGE_FREQ.clear()
    for c1, c2, _ in population:
        for cycle in (c1, c2):
            for u, v in zip(cycle, cycle[1:]):
                EDGE_FREQ[(u, v)] = EDGE_FREQ.get((u, v), 0) + 1
    print("[Init] Initial population ready.")
    return population

# --- Parent selection now uses tournament ---
def select_parents(population):
    parent1 = tournament_select(population)
    parent2 = tournament_select(population)
    # upewnij się, że nie są tym samym osobnikiem
    while parent2 == parent1:
        parent2 = tournament_select(population)
    return parent1, parent2

# --- Rest of HAE algorithm ---
def recombine(parent1, parent2, distance_matrix):
    p1_c1, p1_c2, _ = parent1
    p2_c1, p2_c2, _ = parent2
    y1 = p1_c1.copy()
    y2 = p1_c2.copy()
    # edges from second parent
    edges2 = set()
    for cycle in (p2_c1, p2_c2):
        for u, v in zip(cycle, cycle[1:]):
            edges2.add((u, v)); edges2.add((v, u))
    # include globally frequent edges
    threshold = ELITE_SIZE // 2
    mem_edges = {edge for edge, cnt in EDGE_FREQ.items() if cnt >= threshold}
    combined_edges = edges2.union(mem_edges)

    def prune_cycle(cycle):
        new = [cycle[0]]
        for u, v in zip(cycle, cycle[1:]):
            if (u, v) in combined_edges:
                new.append(v)
        if new[-1] != new[0]: new.append(new[0])
        return new

    y1 = prune_cycle(y1)
    y2 = prune_cycle(y2)
    all_nodes = set(p1_c1[1:-1] + p1_c2[1:-1])
    kept = set(y1[1:-1] + y2[1:-1])
    removed = list(all_nodes - kept)
    y1, y2 = repair_solution(distance_matrix, y1, y2, removed)
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
        iter_count += 1
        p1, p2 = select_parents(population)
        y1, y2 = recombine(p1, p2, distance_matrix)
        # --- LNS perturbation ---
        if iter_count % PERTURB_INTERVAL == 0:
            y1, y2, removed = destroy_solution(y1.copy(), y2.copy(), DESTROY_RATIO)
            y1, y2 = repair_solution(distance_matrix, y1, y2, removed)
        # local search after (possibly perturbed) recombination
        if use_local_search_after_recomb:
            (y1, y2), length, _ = local_search.steepest_original(distance_matrix, y1, y2)
        else:
            length = target_function(y1, y2, distance_matrix)
        offspring = (y1, y2, length)
        # candidate replacement: track worst for memory update
        worst = population[-1]
        if is_unique_and_better(offspring, population):
            # replace worst in population
            population[-1] = offspring; population.sort(key=lambda x: x[2])
            # update EDGE_FREQ: remove worst edges, add offspring edges
            for cycle in (worst[0], worst[1]):
                for u, v in zip(cycle, cycle[1:]):
                    EDGE_FREQ[(u, v)] = EDGE_FREQ.get((u, v), 1) - 1
            for cycle in (y1, y2):
                for u, v in zip(cycle, cycle[1:]):
                    EDGE_FREQ[(u, v)] = EDGE_FREQ.get((u, v), 0) + 1
    best = population[0]; total = time.time() - start
    print(f"[HAE] Done iters={iter_count}, time={total:.2f}s, best={best[2]:.2f}")
    return (best[0], best[1]), best[2], total, iter_count
