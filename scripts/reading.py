"""Reading accuracy on the test sheets: how far one reading of each test line is from another,
counted as the character error rate is (edits over the reference's characters, word spaces included).

Readings of the test lines:
- zl: the dataset's labels (annotation.json, split test).
- takahashi: Takahashi's transcription (config/paths.yaml), read and cleaned as ZL is (scripts/make_lines.py: tags and
  comments removed, both spaces as one) and encoded as ZL is (scripts/make_dataset.py: composites as one capital each).
  A line with a doubtful reading (?, [a:b], @nnn; or {...}) or a ' is left out, as ZL's are.
- RUN, any other name: that run's model's own reading, as its authors' prediction script decodes it (predicted_text in
  data/runs/RUN/predict/<line>.json, scripts/letter_model.py predict), with spaces at its ends removed as ZL's labels have none.
Lines compared: the test lines both readings have. The 95% range comes from resampling the test sheets (parchment
sheets, as the split uses) BOOT times with seed SEED (config/thresholds.yaml).
Output, in config/paths.yaml's reading folder: <hyp>_vs_<ref>.csv (one row per line), <hyp>_vs_<ref>_letters.csv (per
reference letter: count, substituted, deleted, error share), <hyp>_vs_<ref>_sheets.csv (per test sheet and per Currier
language: lines, characters, edits, error) and <hyp>_vs_<ref>.txt (the summary, also printed).
Usage: python scripts/reading.py HYP REF   (each zl, takahashi or a run name)
"""
import csv
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from make_dataset import encode  # noqa: E402
from make_lines import CFG, T, UNSURE, clean, zl_lines  # noqa: E402

READINGS = ("zl", "takahashi")


def test_lines():
    """The dataset's test lines: line -> its entry (ZL label, Currier language, ...)."""
    ann = json.load(open(CFG["dataset"] / "annotation.json"))
    return {k: v for k, v in ann.items() if v["split"] == "test"}


def test_keys():
    """Test line -> its ZL label, from the dataset."""
    return {k: v["label"] for k, v in test_lines().items()}


def groups(rows, sheet, language):
    """(group, lines, reference characters, edits) per test sheet, then per Currier language."""
    out = defaultdict(lambda: [0, 0, 0])
    for k, n, e in rows:
        for g in (f"sheet {sheet[key_of(k)[0]]}", f"language {language[k]}"):
            out[g][0] += 1
            out[g][1] += n
            out[g][2] += e
    return sorted((g, *v) for g, v in out.items())


def key_of(key):
    """Dataset file name (100r_15.png) -> (page, ZL line number)."""
    page, line = Path(key).stem.rsplit("_", 1)
    return page, int(line)


def transcription(path, keys):
    """Test line -> its label in another IVTFF transcription, cleaned and encoded as ZL's; lines it lacks or doubts are
    left out. Also the number of lines left out for each reason."""
    loci = {(page, n): raw for page, rows in zl_lines(path).items() for n, _, raw in rows}
    out, left = {}, Counter()
    for k in keys:
        raw = loci.get(key_of(k))
        if raw is None:
            left["not in the transcription"] += 1
            continue
        label = clean(raw).replace(".", " ").strip()
        if UNSURE.search(re.sub(r"<[^>]*>", "", raw)) or "'" in label:
            left["doubtful reading"] += 1
            continue
        out[k] = encode(label)
    return out, left


def model_reading(folder, keys):
    """Test line -> the model's reading from its prediction files; a line without one is left out."""
    out, left = {}, Counter()
    for k in keys:
        f = folder / f"{Path(k).stem}.json"
        if not f.exists():
            left["not read"] += 1
            continue
        out[k] = json.load(open(f))["predicted_text"].strip(" ")
    return out, left


def reading(name, keys):
    if name == "zl":
        return dict(keys), Counter()
    if name in READINGS:
        return transcription(CFG[name], keys)
    folder = CFG["runs"] / name / "predict"
    if not (folder / "made.json").exists():
        sys.exit(f"{folder} has no made.json: run scripts/letter_model.py predict {name}")
    return model_reading(folder, keys)


def align(ref, hyp):
    """A least-edit alignment as (reference character or None, reading's character or None) pairs."""
    n, m = len(ref), len(hyp)
    d = np.zeros((n + 1, m + 1), int)
    d[:, 0], d[0, :] = range(n + 1), range(m + 1)
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            d[i, j] = min(d[i - 1, j] + 1, d[i, j - 1] + 1, d[i - 1, j - 1] + (ref[i - 1] != hyp[j - 1]))
    pairs, i, j = [], n, m
    while i or j:
        if i and j and d[i, j] == d[i - 1, j - 1] + (ref[i - 1] != hyp[j - 1]):
            pairs.append((ref[i - 1], hyp[j - 1]))
            i, j = i - 1, j - 1
        elif i and d[i, j] == d[i - 1, j] + 1:
            pairs.append((ref[i - 1], None))
            i -= 1
        else:
            pairs.append((None, hyp[j - 1]))
            j -= 1
    return pairs[::-1]


def compare(hyp, ref):
    """Per line (key, reference characters, edits) and per reference letter Counter(count, sub, del), plus insertions."""
    rows, letters, ins = [], defaultdict(Counter), 0
    for k in sorted(set(hyp) & set(ref)):
        pairs = align(ref[k], hyp[k])
        rows.append((k, len(ref[k]), sum(r != h for r, h in pairs)))
        for r, h in pairs:
            if r is None:
                ins += 1
                continue
            letters[r]["count"] += 1
            letters[r]["sub"] += h is not None and h != r
            letters[r]["del"] += h is None
    return rows, letters, ins


def sheet_range(rows, sheet, boot, seed):
    """95% range of the error rate over resamples of the sheets."""
    per = defaultdict(lambda: np.zeros(2))
    for k, n, e in rows:
        per[sheet[key_of(k)[0]]] += (e, n)
    tot = np.array(list(per.values()))
    rng = np.random.default_rng(seed)
    draws = tot[rng.integers(0, len(tot), (boot, len(tot)))].sum(axis=1)
    return np.percentile(draws[:, 0] / draws[:, 1], [2.5, 97.5]), len(tot)


def main():
    if len(sys.argv) != 3:
        sys.exit(__doc__)
    hyp_name, ref_name = sys.argv[1:]
    keys = test_keys()
    (hyp, hyp_left), (ref, ref_left) = reading(hyp_name, keys), reading(ref_name, keys)
    rows, letters, ins = compare(hyp, ref)
    sheet = {r["page"]: f'{r["quire"]}-{r["bifolium"]}' for r in csv.DictReader(open(CFG["page_labels"]))}
    (lo, hi), n_sheets = sheet_range(rows, sheet, T["BOOT"], T["SEED"])
    n, e = sum(r[1] for r in rows), sum(r[2] for r in rows)
    out = CFG["reading"]
    out.mkdir(parents=True, exist_ok=True)
    stem = out / f"{hyp_name}_vs_{ref_name}"
    with open(f"{stem}.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["line", "ref_chars", "edits", "ref", "hyp"])
        w.writerows([k, nk, ek, ref[k], hyp[k]] for k, nk, ek in rows)
    with open(f"{stem}_letters.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["letter", "count", "substituted", "deleted", "error_share"])
        for ch, c in sorted(letters.items(), key=lambda x: -x[1]["count"]):
            w.writerow([ch, c["count"], c["sub"], c["del"], round((c["sub"] + c["del"]) / c["count"], 4)])
    language = {k: v["language"] for k, v in test_lines().items()}
    with open(f"{stem}_sheets.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["group", "lines", "ref_chars", "edits", "error"])
        w.writerows([g, nl, nc, ne, round(ne / nc, 4)] for g, nl, nc, ne in groups(rows, sheet, language))
    left = {**{f"{hyp_name}: {r}": v for r, v in hyp_left.items()}, **{f"{ref_name}: {r}": v for r, v in ref_left.items()}}
    summary = (f"{hyp_name} against {ref_name} on the test lines: {e / n:.2%} of {n} characters "
               f"(95% range {lo:.2%} to {hi:.2%}, {n_sheets} sheets resampled {T['BOOT']} times); "
               f"{len(rows)} of {len(keys)} test lines compared; left out: {left or 'none'}; "
               f"insertions {ins} ({ins / n:.2%} of the reference's characters)")
    Path(f"{stem}.txt").write_text(summary + "\n")
    print(summary)


if __name__ == "__main__":
    main()
