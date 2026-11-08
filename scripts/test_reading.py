"""Tests of scripts/reading.py: the alignment's edits equal the edit distance, letters are counted by what happened to
them, a transcription's doubtful or missing lines are left out, and the model's readings are read with their end spaces
removed.
Usage: python scripts/test_reading.py
"""
import random
import sys
import tempfile
from pathlib import Path

import editdistance

sys.path.insert(0, str(Path(__file__).resolve().parent))
import reading  # noqa: E402


def main():
    rng = random.Random(0)
    for _ in range(300):
        a = "".join(rng.choice("aoe y") for _ in range(rng.randint(0, 12)))
        b = "".join(rng.choice("aoe y") for _ in range(rng.randint(0, 12)))
        pairs = reading.align(a, b)
        assert sum(r != h for r, h in pairs) == editdistance.eval(a, b)
        assert "".join(r for r, _ in pairs if r) == a and "".join(h for _, h in pairs if h) == b

    rows, letters, ins = reading.compare({"1": "axc", "2": "ac", "3": "abbc"}, {"1": "abc", "2": "abc", "3": "abc"})
    assert [r[2] for r in rows] == [1, 1, 1] and ins == 1
    assert letters["b"] == {"count": 3, "sub": 1, "del": 1} and letters["a"]["count"] == 3

    with tempfile.TemporaryDirectory() as tmp:
        f = Path(tmp) / "it.txt"
        f.write_text("<f1r.1,@P0>       <%>fachys.ykal<$>\n"
                     "<f1r.2,+P0>       sory.ckhar\n"
                     "<f1r.3,+P0>       cth?ar.daiin\n"
                     "<f1r.4,+P0>       [a:o]kar\n"
                     "<f1r.5,+P0>       o'kar\n")
        keys = [f"1r_{n}.png" for n in range(1, 7)]
        out, left = reading.transcription(f, keys)
        assert out == {"1r_1.png": "faCys ykal", "1r_2.png": "sory Kar"}
        assert left == {"doubtful reading": 3, "not in the transcription": 1}

        (Path(tmp) / "1r_1.json").write_text('{"predicted_text": " faCys ykal "}')
        out, left = reading.model_reading(Path(tmp), ["1r_1.png", "1r_2.png"])
        assert out == {"1r_1.png": "faCys ykal"} and left == {"not read": 1}

    lo_hi, n = reading.sheet_range([("1r_1.png", 10, 1), ("2r_1.png", 10, 1)], {"1r": "A-1", "2r": "A-2"}, 50, 0)
    assert n == 2 and abs(lo_hi[0] - 0.1) < 1e-12 and abs(lo_hi[1] - 0.1) < 1e-12

    g = reading.groups([("1r_1.png", 10, 1), ("1v_1.png", 20, 4), ("2r_1.png", 5, 0)], {"1r": "A-1", "1v": "A-1", "2r": "A-2"},
                       {"1r_1.png": "A", "1v_1.png": "B", "2r_1.png": "A"})
    assert g == [("language A", 2, 15, 1), ("language B", 1, 20, 4), ("sheet A-1", 2, 30, 5), ("sheet A-2", 1, 5, 0)]
    print("reading: 8 checks pass")


if __name__ == "__main__":
    main()
