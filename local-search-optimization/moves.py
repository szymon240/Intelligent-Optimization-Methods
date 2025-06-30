def swap_nodes_within_cycle(cycle_index, cycles, distance_matrix):
    cycle = cycles[cycle_index]
    n = len(cycle) - 1
    for i in range(1, n):
        for j in range(i + 1, n):
            new_cycle = cycle[:]
            new_cycle[i], new_cycle[j] = new_cycle[j], new_cycle[i]

            delta = (
                distance_matrix[new_cycle[i - 1]][new_cycle[i]] +
                distance_matrix[new_cycle[i]][new_cycle[i + 1]] +
                distance_matrix[new_cycle[j - 1]][new_cycle[j]] +
                distance_matrix[new_cycle[j]][new_cycle[j + 1]]
            ) - (
                distance_matrix[cycle[i - 1]][cycle[i]] +
                distance_matrix[cycle[i]][cycle[i + 1]] +
                distance_matrix[cycle[j - 1]][cycle[j]] +
                distance_matrix[cycle[j]][cycle[j + 1]]
            )

            yield {
                'type': 'within_nodes',
                'cycle_index': cycle_index,
                'i': i,
                'j': j,
                'delta': delta
            }

def swap_edges_within_cycle(cycle_index, cycles, distance_matrix):
    cycle = cycles[cycle_index]
    n = len(cycle) - 1
    for i in range(1, n - 1):
        for j in range(i + 2, n):
            new_cycle = cycle[:]
            new_cycle[i:j + 1] = reversed(new_cycle[i:j + 1])

            delta = (
                distance_matrix[new_cycle[i - 1]][new_cycle[i]] +
                distance_matrix[new_cycle[j]][new_cycle[j + 1]]
            ) - (
                distance_matrix[cycle[i - 1]][cycle[i]] +
                distance_matrix[cycle[j]][cycle[j + 1]]
            )

            yield {
                'type': 'within_edges',
                'cycle_index': cycle_index,
                'i': i,
                'j': j,
                'delta': delta
            }

def swap_nodes_between_cycles(cycles, distance_matrix):
    first, second = cycles[0], cycles[1]
    n1, n2 = len(first) - 1, len(second) - 1
    for i in range(1, n1):
        for j in range(1, n2):
            new_first = first[:]
            new_second = second[:]
            new_first[i], new_second[j] = new_second[j], new_first[i]

            delta = (
                distance_matrix[new_first[i - 1]][new_first[i]] +
                distance_matrix[new_first[i]][new_first[i + 1]] +
                distance_matrix[new_second[j - 1]][new_second[j]] +
                distance_matrix[new_second[j]][new_second[j + 1]]
            ) - (
                distance_matrix[first[i - 1]][first[i]] +
                distance_matrix[first[i]][first[i + 1]] +
                distance_matrix[second[j - 1]][second[j]] +
                distance_matrix[second[j]][second[j + 1]]
            )

            yield {
                'type': 'between_nodes',
                'i': i,
                'j': j,
                'delta': delta
            }

def generate_all_moves(cycles, distance_matrix):
    moves = []

    for cycle_index in [0, 1]:
        moves.extend(swap_edges_within_cycle(cycle_index, cycles, distance_matrix))

    moves.extend(swap_nodes_between_cycles(cycles, distance_matrix))

    return moves