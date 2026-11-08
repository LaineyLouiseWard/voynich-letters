"""Tests of scripts/read_letters.py's forced alignment: a known best path, a double letter on two boxes, too many
symbols, and on small random tables the same rows as trying every placement.
Usage: python scripts/test_read_letters.py
"""
import itertools
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from read_letters import align  # noqa: E402


def total(logp, targets, rows):
    return sum(logp[q, t] for q, t in zip(rows, targets)) + sum(logp[q, 0] for q in set(range(len(logp))) - set(rows))


def main():
    p = np.full((5, 4), 0.01)
    p[:, 0] = 0.9
    p[1, 2], p[3, 3] = 0.95, 0.9
    assert align(np.log(p), [2, 3]) == [1, 3]

    p = np.full((5, 6), 0.01)
    p[:, 0] = 0.9
    p[2, 5], p[3, 5] = 0.8, 0.7
    assert align(np.log(p), [5, 5]) == [2, 3]

    assert align(np.log(np.full((2, 3), 0.5)), [1, 2, 1]) is None

    rng = np.random.default_rng(0)
    for _ in range(200):
        n, m = rng.integers(1, 8), rng.integers(1, 5)
        if m > n:
            continue
        logp = np.log(rng.uniform(0.01, 1, (n, 5)))
        targets = list(rng.integers(1, 5, m))
        rows = align(logp, targets)
        best = max(itertools.combinations(range(n), m), key=lambda c: total(logp, targets, c))
        assert len(rows) == m and rows == sorted(rows)
        assert abs(total(logp, targets, rows) - total(logp, targets, best)) < 1e-9
    print("read_letters: 4 checks pass")


if __name__ == "__main__":
    main()
