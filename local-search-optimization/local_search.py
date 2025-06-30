from moves import generate_all_moves
import copy
import numpy as np
import time
import swaps as og_swaps
from utils import cycle_length

def apply_move(move, cycles):
    new_cycles = copy.deepcopy(cycles)

    if move['type'] == 'within_nodes':
        i, j = move['i'], move['j']
        c = move['cycle_index']
        new_cycles[c][i], new_cycles[c][j] = new_cycles[c][j], new_cycles[c][i]

    elif move['type'] == 'within_edges':
        i, j = move['i'], move['j']
        c = move['cycle_index']
        new_cycles[c][i:j + 1] = list(reversed(new_cycles[c][i:j + 1]))

    elif move['type'] == 'between_nodes':
        i, j = move['i'], move['j']
        new_cycles[0][i], new_cycles[1][j] = new_cycles[1][j], new_cycles[0][i]

    return new_cycles

def get_removed_edges(move, cycles):
    edges = []
    if move['type'] == 'within_nodes':
        c = move['cycle_index']
        i, j = move['i'], move['j']
        cycle = cycles[c]
        edges.append((cycle[i - 1], cycle[i]))
        edges.append((cycle[i], cycle[i + 1]))
        edges.append((cycle[j - 1], cycle[j]))
        edges.append((cycle[j], cycle[j + 1]))

    elif move['type'] == 'within_edges':
        c = move['cycle_index']
        i, j = move['i'], move['j']
        cycle = cycles[c]
        edges.append((cycle[i - 1], cycle[i]))
        edges.append((cycle[j], cycle[j + 1]))

    elif move['type'] == 'between_nodes':
        i, j = move['i'], move['j']
        cycle0 = cycles[0]
        cycle1 = cycles[1]
        edges.append((cycle0[i - 1], cycle0[i]))
        edges.append((cycle0[i], cycle0[i + 1]))
        edges.append((cycle1[j - 1], cycle1[j]))
        edges.append((cycle1[j], cycle1[j + 1]))

    return edges

def is_edge_in_cycles(edge, cycles):
    for cycle in cycles:
        for i in range(len(cycle) - 1):
            if (cycle[i], cycle[i + 1]) == edge:
                return 1  # taki sam kierunek
            if (cycle[i + 1], cycle[i]) == edge:
                return -1  # odwrotny kierunek
    return 0  # brak krawędzi

def steepest_descent(cycles: tuple[list[int], list[int]], distance_matrix: np.ndarray) -> tuple[list[int], list[int]]:
    LM = []  # lista krawędzi (z kierunkiem)
    improved = True

    while improved:
        improved = False
        new_moves = generate_all_moves(cycles, distance_matrix)
        new_moves = [m for m in new_moves if m['delta'] < 0]
        new_moves.sort(key=lambda x: x['delta'])

        for move in new_moves:
            removed_edges = get_removed_edges(move, cycles)
            edge_statuses = [is_edge_in_cycles(edge, cycles) for edge in removed_edges]

            # Sprawdzamy warunki aplikowalności ruchu
            if all(status == 1 for status in edge_statuses):
                # Krawędzie w tym samym kierunku – aplikujemy
                cycles = apply_move(move, cycles)

                # Dodajemy krawędzie i ich odwrotności do LM
                for e in removed_edges:
                    if e not in LM:
                        LM.append(e)
                    rev = (e[1], e[0])
                    if rev not in LM:
                        LM.append(rev)

                improved = True
                break

            elif all(is_edge_in_cycles(edge, cycles) == 0 for edge in removed_edges):
                # Krawędzi już nie ma – usuwamy ruch z LM
                for e in removed_edges:
                    if e in LM:
                        LM.remove(e)
                    rev = (e[1], e[0])
                    if rev in LM:
                        LM.remove(rev)
                continue

            else:
                # Krawędzie są, ale nie we właściwym kierunku – ignorujemy
                continue

    return cycles


def steepest_original(distance_matrix, first_cycle, second_cycle):
    start_time = time.time()
    best_length = cycle_length(first_cycle, distance_matrix) + cycle_length(second_cycle, distance_matrix)

    improved = True
    while improved:
        improved = False
        best_delta = 0
        best_move = None

        # Sprawdzanie zamian między cyklami
        for new_first, new_second, delta in og_swaps.swap_nodes_between_cycles(first_cycle, second_cycle, distance_matrix):
            if delta < best_delta:
                best_delta = delta
                best_move = ('both', new_first, new_second)

        # Sprawdzanie zamian krawędzi w obrębie pierwszego cyklu (2-opt move)
        for new_cycle, delta in og_swaps.swap_edges_within_cycle(first_cycle, distance_matrix):
            if delta < best_delta:
                best_delta = delta
                best_move = ('first', new_cycle)

        # Sprawdzanie zamian krawędzi w obrębie drugiego cyklu (2-opt move)
        for new_cycle, delta in og_swaps.swap_edges_within_cycle(second_cycle, distance_matrix):
            if delta < best_delta:
                best_delta = delta
                best_move = ('second', new_cycle)

        if best_move is not None:
            if best_move[0] == 'both':
                first_cycle, second_cycle = best_move[1], best_move[2]
            elif best_move[0] == 'first':
                first_cycle = best_move[1]
            elif best_move[0] == 'second':
                second_cycle = best_move[1]

            best_length += best_delta
            improved = True

    execution_time = time.time() - start_time
    return (first_cycle, second_cycle), best_length, execution_time

def get_candidate_edges(distance_matrix, k=10):
    """
    Wyznacza krawędzie kandydackie dla każdego wierzchołka.
    """
    candidate_edges = {}
    num_nodes = distance_matrix.shape[0]
    for i in range(num_nodes):
        nearest_neighbors = np.argsort(distance_matrix[i])[:k + 1]  # +1, aby uwzględnić wierzchołek i
        candidate_edges[i] = [j for j in nearest_neighbors if j != i]
    return candidate_edges

def steepest_original_with_candidates(distance_matrix, first_cycle, second_cycle, k=10):
    """
    Wersja lokalnego przeszukiwania z mechanizmem kandydackim, wykorzystująca funkcje z swaps.py.
    """
    start_time = time.time()
    best_length = cycle_length(first_cycle, distance_matrix) + cycle_length(second_cycle, distance_matrix)

    # Wyznacz krawędzie kandydackie
    candidate_edges = get_candidate_edges(distance_matrix, k)

    improved = True
    while improved:
        improved = False
        best_delta = 0
        best_move = None

        # Sprawdzanie zamian między cyklami
        for new_first, new_second, delta in og_swaps.swap_nodes_between_cycles_with_candidates(first_cycle, second_cycle, distance_matrix, candidate_edges):
            if delta < best_delta:
                best_delta = delta
                best_move = ('both', new_first, new_second)

        # Sprawdzanie zamian krawędzi w obrębie pierwszego cyklu (2-opt move)
        for new_cycle, delta in og_swaps.swap_edges_within_cycle_with_candidates(first_cycle, distance_matrix, candidate_edges):
            if delta < best_delta:
                best_delta = delta
                best_move = ('first', new_cycle)

        # Sprawdzanie zamian krawędzi w obrębie drugiego cyklu (2-opt move)
        for new_cycle, delta in og_swaps.swap_edges_within_cycle_with_candidates(second_cycle, distance_matrix, candidate_edges):
            if delta < best_delta:
                best_delta = delta
                best_move = ('second', new_cycle)

        # Wykonanie najlepszego ruchu
        if best_move is not None:
            if best_move[0] == 'both':
                first_cycle, second_cycle = best_move[1], best_move[2]
            elif best_move[0] == 'first':
                first_cycle = best_move[1]
            elif best_move[0] == 'second':
                second_cycle = best_move[1]

            best_length += best_delta
            improved = True

    execution_time = time.time() - start_time
    return (first_cycle, second_cycle), best_length, execution_time