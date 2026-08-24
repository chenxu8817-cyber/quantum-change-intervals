from __future__ import annotations

import itertools
import unittest

import numpy as np


UPPER_EDGES = ((0, 1), (0, 2), (1, 2))


def candidates(s: int) -> list[tuple[int, int, int]]:
    return list(itertools.combinations(range(s + 1), 3))


def monotone_edge_sets(
    m: int,
    row: int = 0,
    last_right: int = 0,
    rows: tuple[tuple[int, ...], ...] = (),
):
    if row == m:
        yield tuple(
            (left, right)
            for left, neighbors in enumerate(rows)
            for right in neighbors
        )
        return

    allowed = [right for right in range(last_right, m) if right != row]
    for count in range(len(allowed) + 1):
        for neighbors in itertools.combinations(allowed, count):
            next_right = max(neighbors) if neighbors else last_right
            yield from monotone_edge_sets(
                m,
                row + 1,
                next_right,
                rows + (neighbors,),
            )


def forest_separator_exponent(
    m: int,
    edges: tuple[tuple[int, int], ...],
) -> tuple[bool, int]:
    adjacency = [set() for _ in range(2 * m)]
    for left, right in edges:
        adjacency[left].add(m + right)
        adjacency[m + right].add(left)

    seen: set[int] = set()
    components = 0
    for vertex, neighbors in enumerate(adjacency):
        if neighbors and vertex not in seen:
            components += 1
            seen.add(vertex)
            stack = [vertex]
            while stack:
                current = stack.pop()
                for neighbor in adjacency[current]:
                    if neighbor not in seen:
                        seen.add(neighbor)
                        stack.append(neighbor)

    incident_vertices = sum(bool(neighbors) for neighbors in adjacency)
    is_forest = len(edges) == incident_vertices - components
    untouched = sum(
        not adjacency[index] and not adjacency[m + index]
        for index in range(m)
    )
    return is_forest, components + untouched


def overlap_profile(left: int, right: int, delta: int) -> int:
    return max(
        0,
        min(left, left - delta + right) - max(0, left - delta),
    )


def depth(
    edge: tuple[int, int],
    u: tuple[int, int, int],
    v: tuple[int, int, int],
    lengths: tuple[int, int, int],
) -> int:
    j, k = edge
    prefix = (0, lengths[0], lengths[0] + lengths[1], sum(lengths))
    return u[j] - v[k] + prefix[j + 1] - prefix[k]


def displacement(
    edge: tuple[int, int],
    delta: int,
    lengths: tuple[int, int, int],
) -> int:
    j, k = edge
    prefix = (0, lengths[0], lengths[0] + lengths[1], sum(lengths))
    return delta - prefix[j + 1] + prefix[k]


def q(length: int, r: float, x: int, y: int, s: int) -> float:
    if not (0 <= x <= s and 0 <= y <= s):
        return 0.0
    return r ** min(abs(x - y), length)


def square_root_kernel(length: int, r: float, s: int) -> np.ndarray:
    kernel = np.array(
        [[q(length, r, x, y, s) for y in range(s + 1)] for x in range(s + 1)]
    )
    values, vectors = np.linalg.eigh(kernel)
    return (vectors * np.sqrt(np.maximum(values, 0.0))) @ vectors.T


def corresponding_kernel(
    u: tuple[int, int, int],
    v: tuple[int, int, int],
    lengths: tuple[int, int, int],
    r: float,
    s: int,
) -> float:
    return np.prod([q(lengths[p], r, u[p], v[p], s) for p in range(3)])


def target_layer(
    points: list[tuple[int, int, int]],
    forest: tuple[tuple[int, int], ...],
    deltas: tuple[int, ...],
    lengths: tuple[int, int, int],
    r: float,
    s: int,
) -> np.ndarray:
    result = np.zeros((len(points), len(points)))
    for row, u in enumerate(points):
        for col, v in enumerate(points):
            if all(
                depth(edge, u, v, lengths) == delta
                for edge, delta in zip(forest, deltas, strict=True)
            ):
                result[row, col] = corresponding_kernel(u, v, lengths, r, s)
    return result


def explicit_factors(
    points: list[tuple[int, int, int]],
    forest: tuple[tuple[int, int], ...],
    deltas: tuple[int, ...],
    lengths: tuple[int, int, int],
    r: float,
    s: int,
) -> tuple[np.ndarray, np.ndarray]:
    ds = {
        edge: displacement(edge, delta, lengths)
        for edge, delta in zip(forest, deltas, strict=True)
    }
    size = s + 1

    if len(forest) == 1:
        j, k = forest[0]
        h = ({0, 1, 2} - {j, k}).pop()
        root = square_root_kernel(lengths[h], r, s)
        left = np.zeros((len(points), size * size))
        right = np.zeros((size * size, len(points)))
        for row, u in enumerate(points):
            for t in range(size):
                if u[j] == t + ds[(j, k)]:
                    for z in range(size):
                        left[row, t * size + z] = (
                            q(lengths[k], r, u[k], t, s) * root[u[h], z]
                        )
        for col, v in enumerate(points):
            for t in range(size):
                if v[k] == t:
                    for z in range(size):
                        right[t * size + z, col] = (
                            q(lengths[j], r, t + ds[(j, k)], v[j], s)
                            * root[z, v[h]]
                        )
        return left, right

    if forest == ((0, 1), (0, 2)):
        left = np.zeros((len(points), size))
        right = np.zeros((size, len(points)))
        for row, u in enumerate(points):
            for t in range(size):
                if u[0] == t:
                    left[row, t] = q(lengths[1], r, u[1], t - ds[(0, 1)], s) * q(
                        lengths[2], r, u[2], t - ds[(0, 2)], s
                    )
        for col, v in enumerate(points):
            for t in range(size):
                if v[1] == t - ds[(0, 1)] and v[2] == t - ds[(0, 2)]:
                    right[t, col] = q(lengths[0], r, t, v[0], s)
        return left, right

    if forest == ((0, 2), (1, 2)):
        left = np.zeros((len(points), size))
        right = np.zeros((size, len(points)))
        for row, u in enumerate(points):
            for t in range(size):
                if u[0] == t + ds[(0, 2)] and u[1] == t + ds[(1, 2)]:
                    left[row, t] = q(lengths[2], r, u[2], t, s)
        for col, v in enumerate(points):
            for t in range(size):
                if v[2] == t:
                    right[t, col] = q(
                        lengths[0], r, t + ds[(0, 2)], v[0], s
                    ) * q(lengths[1], r, t + ds[(1, 2)], v[1], s)
        return left, right

    if forest == ((0, 1), (1, 2)):
        left = np.zeros((len(points), size * size))
        right = np.zeros((size * size, len(points)))
        for row, u in enumerate(points):
            for t in range(size):
                for w in range(size):
                    if u[0] == t + ds[(0, 1)] and u[1] == w + ds[(1, 2)]:
                        left[row, t * size + w] = q(lengths[2], r, u[2], w, s)
        for col, v in enumerate(points):
            for t in range(size):
                for w in range(size):
                    if v[1] == t and v[2] == w:
                        right[t * size + w, col] = q(
                            lengths[0], r, t + ds[(0, 1)], v[0], s
                        ) * q(lengths[1], r, w + ds[(1, 2)], t, s)
        return left, right

    if forest == UPPER_EDGES:
        left = np.zeros((len(points), size))
        right = np.zeros((size, len(points)))
        for row, u in enumerate(points):
            for t in range(size):
                if u[0] == t + ds[(0, 2)] and u[1] == t + ds[(1, 2)]:
                    left[row, t] = q(
                        lengths[1],
                        r,
                        t + ds[(1, 2)],
                        t + ds[(0, 2)] - ds[(0, 1)],
                        s,
                    ) * q(lengths[2], r, u[2], t, s)
        for col, v in enumerate(points):
            for t in range(size):
                if v[2] == t and v[1] == t + ds[(0, 2)] - ds[(0, 1)]:
                    right[t, col] = q(
                        lengths[0], r, t + ds[(0, 2)], v[0], s
                    )
        return left, right

    raise AssertionError(f"unhandled forest {forest}")


class M3ForestFactorizationTests(unittest.TestCase):
    def test_two_sided_energy_cover_through_m4(self) -> None:
        for m in range(2, 5):
            for lengths in itertools.product(range(1, 4), repeat=m):
                prefix = (0,) + tuple(itertools.accumulate(lengths))
                for s in range(m - 1, m + 3):
                    points = list(itertools.combinations(range(s + 1), m))
                    for u in points:
                        for v in points:
                            diagonal_mismatch = [
                                min(abs(u[index] - v[index]), lengths[index])
                                for index in range(m)
                            ]
                            cross_edges: list[tuple[int, int, int]] = []
                            for left in range(m):
                                left_start = u[left] + prefix[left]
                                left_end = left_start + lengths[left]
                                for right in range(m):
                                    if left == right:
                                        continue
                                    right_start = v[right] + prefix[right]
                                    right_end = right_start + lengths[right]
                                    overlap = max(
                                        0,
                                        min(left_end, right_end)
                                        - max(left_start, right_start),
                                    )
                                    if overlap:
                                        cross_edges.append((left, right, overlap))

                            left_incident = {left for left, _, _ in cross_edges}
                            right_incident = {right for _, right, _ in cross_edges}
                            total_cross = sum(
                                overlap for _, _, overlap in cross_edges
                            )
                            self.assertGreaterEqual(
                                sum(diagonal_mismatch[p] for p in left_incident),
                                total_cross,
                            )
                            self.assertGreaterEqual(
                                sum(diagonal_mismatch[p] for p in right_incident),
                                total_cross,
                            )

    def test_general_monotone_forest_separator_exponent_through_m6(self) -> None:
        for m in range(2, 7):
            for edges in monotone_edge_sets(m):
                if not edges:
                    continue
                is_forest, exponent = forest_separator_exponent(m, edges)
                if is_forest:
                    self.assertLessEqual(exponent, m - 1)

    def test_upper_and_lower_cross_edges_do_not_coexist(self) -> None:
        for lengths in itertools.product(range(1, 4), repeat=3):
            for s in range(2, 7):
                points = candidates(s)
                for u in points:
                    for v in points:
                        upper = any(
                            overlap_profile(
                                lengths[j],
                                lengths[k],
                                depth((j, k), u, v, lengths),
                            )
                            > 0
                            for j, k in UPPER_EDGES
                        )
                        lower = any(
                            overlap_profile(
                                lengths[k],
                                lengths[j],
                                depth((k, j), u, v, lengths),
                            )
                            > 0
                            for j, k in UPPER_EDGES
                        )
                        self.assertFalse(upper and lower)

    def test_all_seven_upper_forest_layers(self) -> None:
        lengths = (3, 2, 4)
        r = 0.43
        s = 6
        points = candidates(s)
        forests = tuple(
            forest
            for count in range(1, 4)
            for forest in itertools.combinations(UPPER_EDGES, count)
        )
        for forest in forests:
            depth_ranges = [
                range(1, lengths[j] + lengths[k]) for j, k in forest
            ]
            for deltas in itertools.product(*depth_ranges):
                target = target_layer(points, forest, deltas, lengths, r, s)
                left, right = explicit_factors(
                    points, forest, deltas, lengths, r, s
                )
                np.testing.assert_allclose(left @ right, target, atol=2e-12)

    def test_subset_expansion_reconstructs_oriented_correction(self) -> None:
        lengths = (3, 2, 4)
        r = 0.43
        s = 6
        points = candidates(s)
        direct = np.zeros((len(points), len(points)))
        expanded = np.zeros_like(direct)

        for row, u in enumerate(points):
            for col, v in enumerate(points):
                total_overlap = sum(
                    overlap_profile(
                        lengths[j], lengths[k], depth((j, k), u, v, lengths)
                    )
                    for j, k in UPPER_EDGES
                )
                direct[row, col] = corresponding_kernel(
                    u, v, lengths, r, s
                ) * (r ** (-total_overlap) - 1.0)

        for count in range(1, 4):
            for forest in itertools.combinations(UPPER_EDGES, count):
                depth_ranges = [
                    range(1, lengths[j] + lengths[k]) for j, k in forest
                ]
                for deltas in itertools.product(*depth_ranges):
                    coefficient = np.prod(
                        [
                            r
                            ** (-overlap_profile(lengths[j], lengths[k], delta))
                            - 1.0
                            for (j, k), delta in zip(
                                forest, deltas, strict=True
                            )
                        ]
                    )
                    left, right = explicit_factors(
                        points, forest, deltas, lengths, r, s
                    )
                    expanded += coefficient * (left @ right)

        np.testing.assert_allclose(expanded, direct, atol=5e-11)


if __name__ == "__main__":
    unittest.main()
