"""The README's three figures, from a run's letter pictures and the letter export (data/letters/).

- line_boxes.png: the first held-back line, in page order, with at least LINE_LETTERS letters, on the page scan with
  its letter boxes.
- alphabet.png: every letter's learned picture, stretched to its mean box width and height in the run's reading of the
  dataset (predict/transcribe.json, as the authors' grid.jpg is), under the letter and its number of training examples;
  most common first, the letters with fewer than RARE examples on their own row (a layout choice only).
- letters.png: for k, d, q and ch, the learned picture, then SHOW crops drawn at random (seed SEED) from that letter's
  crops on held-back sheets, each scaled to one height.
- scribes.png: under each letter with at least RARE training examples from every one of SCRIBES (most common first),
  each scribe's learned picture from the run's per-scribe step 2 (scribes/<scribe>/sprites/, last saved pass),
  stretched to that scribe's mean box width and height.
Usage: python scripts/figures.py RUN
"""
import csv
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import yaml  # noqa: E402
from PIL import Image  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from make_dataset import COMPOSITES  # noqa: E402
from make_lines import CFG, T  # noqa: E402

OUT = ROOT / "figures"
LETTERS = ROOT / "data/letters"
LINE_LETTERS, SHOW, ROWS, RARE = 30, 12, ("k", "d", "q", "ch"), 200
SCRIBES = yaml.safe_load(open(ROOT / "config/model.yaml"))["scribes"]
DECODE = {cap: eva for eva, cap in COMPOSITES}
plt.rcParams.update({"font.family": "serif", "font.serif": ["cmr10"], "axes.unicode_minus": False,
                     "mathtext.fontset": "cm", "axes.formatter.use_mathtext": True})


def pictures(run):
    """EVA letter -> (learned picture as an array, mean box width, mean box height)."""
    pred = CFG["runs"] / run / "predict"
    stats = json.load(open(pred / "transcribe.json"))
    out = {}
    for i, v in stats.items():
        if v["character"] == " ":
            continue
        b = v["bbox_stats"]
        out[DECODE.get(v["character"], v["character"])] = (np.asarray(Image.open(pred / "sprite_final" / f"{i}.png")),
                                                            b["mean_width"] or 32, b["mean_height"] or 32)
    return out


def stretched(pic, w, h, height=64):
    return np.asarray(Image.fromarray(pic).resize((max(1, round(height * w / h)), height), Image.Resampling.BILINEAR))


def line_boxes(rows):
    by_line = defaultdict(list)
    for r in rows:
        by_line[r["line"]].append(r)
    test = [k for k in by_line if by_line[k][0]["split"] == "test" and len(by_line[k]) >= LINE_LETTERS]
    line = min(test, key=lambda k: (int(k.split("_")[0][:-1]), k.split("_")[0][-1], int(k.split("_")[1])))
    boxes = [tuple(int(r[c]) for c in ("x0", "y0", "x1", "y1")) for r in by_line[line] if r["x0"]]
    page = by_line[line][0]["page"]
    image = {r["page"]: r["iipimage"] for r in csv.DictReader(open(CFG["word_boxes"]))}[page]
    x0, y0 = min(b[0] for b in boxes) - 10, min(b[1] for b in boxes) - 10
    x1, y1 = max(b[2] for b in boxes) + 10, max(b[3] for b in boxes) + 10
    scan = Image.open(CFG["images"] / image).convert("RGB").crop((x0, y0, x1, y1))
    fig, ax = plt.subplots(figsize=(12, 12 * scan.height / scan.width))
    ax.imshow(scan)
    for k, (a, b, c, d) in enumerate(boxes):
        ax.add_patch(plt.Rectangle((a - x0, b - y0), c - a, d - b, fill=False, lw=1,
                                   edgecolor=("#1f77b4", "#d62728")[k % 2]))
    ax.axis("off")
    fig.savefig(OUT / "line_boxes.png", dpi=150, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)
    return line


def alphabet(pics, counts):
    order = sorted(pics, key=lambda ch: -counts[ch])
    cols = 10
    rows = []
    for group in ([c for c in order if counts[c] >= RARE], [c for c in order if counts[c] < RARE]):
        rows += [group[k:k + cols] for k in range(0, len(group), cols)]
    fig, axes = plt.subplots(len(rows), cols, figsize=(cols * 1.1, 1.6 * len(rows)))
    for ax in axes.flat:
        ax.axis("off")
    for ax, ch in ((axes[r, k], ch) for r, row in enumerate(rows) for k, ch in enumerate(row)):
        pic, w, h = pics[ch]
        ax.imshow(stretched(pic, w, h), cmap="gray", vmin=0, vmax=255)
        ax.set_title(f"{ch}\n{counts[ch]:,}", fontsize=10)
    fig.subplots_adjust(hspace=0.6)
    fig.savefig(OUT / "alphabet.png", dpi=150, bbox_inches="tight", pad_inches=0.05)
    plt.close(fig)


def letters(pics, rows):
    rng = np.random.default_rng(T["SEED"])
    fig, axes = plt.subplots(len(ROWS), SHOW + 1, figsize=((SHOW + 1) * 0.9, len(ROWS) * 1.1))
    for row, ch in zip(axes, ROWS):
        pic, w, h = pics[ch]
        row[0].imshow(stretched(pic, w, h), cmap="gray", vmin=0, vmax=255)
        row[0].set_ylabel(ch, rotation=0, fontsize=13, labelpad=14, va="center")
        pool = [r for r in rows if r["zl_letter"] == ch and r["split"] == "test" and r["x0"]]
        for ax, k in zip(row[1:], rng.choice(len(pool), SHOW, replace=False)):
            r = pool[k]
            ax.imshow(Image.open(LETTERS / "crops" / f"{r['line']}_{int(r['letter']):03d}.png").convert("L"),
                      cmap="gray", vmin=0, vmax=255)
        for ax in row:
            ax.set_xticks([])
            ax.set_yticks([])
            for s in ax.spines.values():
                s.set_visible(False)
    fig.savefig(OUT / "letters.png", dpi=150, bbox_inches="tight", pad_inches=0.05)
    plt.close(fig)


def scribes(run):
    ann = json.load(open(CFG["scribe_dataset"] / "annotation.json"))
    counts = {s: Counter(DECODE.get(c, c) for v in ann.values() if v["split"] == "train" and v["script"] == s
                         for c in v["label"] if c != " ") for s in SCRIBES}
    pics = {}
    for s in SCRIBES:
        d = CFG["runs"] / run / "scribes" / s
        last = max((p for p in (d / "sprites").iterdir() if p.name.isdigit()), key=lambda p: int(p.name))
        pics[s] = {DECODE.get(v["character"], v["character"]): (np.asarray(Image.open(last / f"{i}.png")),
                                                                v["bbox_stats"]["mean_width"] or 32,
                                                                v["bbox_stats"]["mean_height"] or 32)
                   for i, v in json.load(open(d / "transcribe.json")).items() if v["character"] != " "}
    shown = sorted((ch for ch in pics[SCRIBES[0]] if all(counts[s][ch] >= RARE for s in SCRIBES)),
                   key=lambda ch: -sum(counts[s][ch] for s in SCRIBES))
    fig, axes = plt.subplots(len(SCRIBES), len(shown), figsize=(len(shown) * 0.9, len(SCRIBES) * 1.1))
    for row, s in zip(axes, SCRIBES):
        for ax, ch in zip(row, shown):
            ax.imshow(stretched(*pics[s][ch]), cmap="gray", vmin=0, vmax=255)
            ax.set_xticks([])
            ax.set_yticks([])
            for sp in ax.spines.values():
                sp.set_visible(False)
        row[0].set_ylabel(f"Scribe {s}", rotation=0, fontsize=11, labelpad=30, va="center")
    for ax, ch in zip(axes[0], shown):
        ax.set_title(ch, fontsize=13)
    fig.savefig(OUT / "scribes.png", dpi=150, bbox_inches="tight", pad_inches=0.05)
    plt.close(fig)
    return shown


def main():
    if len(sys.argv) != 2:
        sys.exit(__doc__)
    rows = list(csv.DictReader(open(LETTERS / "letters.csv")))
    ann = json.load(open(CFG["dataset"] / "annotation.json"))
    counts = Counter(DECODE.get(c, c) for v in ann.values() if v["split"] == "train" for c in v["label"] if c != " ")
    pics = pictures(sys.argv[1])
    OUT.mkdir(exist_ok=True)
    line = line_boxes(rows)
    alphabet(pics, counts)
    letters(pics, rows)
    shown = scribes(sys.argv[1])
    print(f"figures/: line_boxes.png (line {line}), alphabet.png, letters.png, scribes.png ({' '.join(shown)})")


if __name__ == "__main__":
    main()
