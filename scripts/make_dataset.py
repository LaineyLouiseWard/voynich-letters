"""Build the letter model's dataset folder from the line images.

Lines: train=1 rows of data/lines/lines.csv (no unreadable glyph, alternative reading, rare-glyph code or ligature
brace in ZL), without the 16 that hold EVA's apostrophe mark, too rare to learn. Label: the ZL text with the frequent
EVA composites written as one capital each (cth T, ckh K, cph P, cfh F, ch C, sh S, as in Ponzi 2023), so each is one
symbol to the model. Word spaces stay in the labels: the model reads them as its authors' method does (config/model.yaml
space_index); no space is exported as a letter.

Splits are by whole parchment sheet (bifolium, named by quire and bifolium number), never single pages, so that
held-back lines come from sheets the model has not seen. Sheets are shuffled with SEED.
- annotation.json, for the reference model: sheets assigned in turn to test until it holds TEST of the lines, then to
  val until it holds VAL; the rest train. "script" is Currier's language (A or B).

Output: config/paths.yaml's dataset folder (images/ linking to data/lines/img/, and annotation.json), and its
scribe_dataset folder for the per-scribe step 2 (the same lines and splits, "script" = Davis's scribe, images/ linking
to the dataset folder's). The model writes its letter list (charset*.json) in a folder on first use; a changed
annotation.json deletes it, so a changed dataset never reuses an old list.
Usage: python scripts/make_dataset.py
"""
import csv
import json
import random
from collections import Counter
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
CFG = {k: ROOT / v for k, v in yaml.safe_load(open(ROOT / "config/paths.yaml")).items()}
OUT = CFG["dataset"]
COMPOSITES = [("cth", "T"), ("ckh", "K"), ("cph", "P"), ("cfh", "F"), ("ch", "C"), ("sh", "S")]
T = yaml.safe_load(open(ROOT / "config/thresholds.yaml"))
SEED, TEST, VAL = T["SEED"], T["TEST"], T["VAL"]


def encode(label):
    for eva, cap in COMPOSITES:
        label = label.replace(eva, cap)
    return label


def fill(order, per_sheet, total, split, quotas):
    """Assign the sheets of order not yet in split to each (name, share) in turn until it holds share of total lines;
    the rest go to train."""
    filled = Counter()
    for s, part in split.items():
        filled[part] += per_sheet[s]
    for s in order:
        if s in split:
            continue
        part = next((name for name, share in quotas if filled[name] < share * total), "train")
        split[s] = part
        filled[part] += per_sheet[s]
    return split


def write(out, lines, sheet, split, script):
    """annotation.json in out, with script's column of lines.csv as "script". When it changes, the model's letter list
    there (charset*.json) is deleted, so a changed dataset never reuses an old list."""
    ann = {r["file"]: {"split": split[sheet[r["page"]]], "label": encode(r["label"]), "script": r[script],
                       "hand": r["davis_hand"], "language": r["language"], "page": r["page"]} for r in lines}
    path, text = out / "annotation.json", json.dumps(ann, indent=1)
    if not path.exists() or path.read_text() != text:
        path.write_text(text)
        for old in out.glob("charset*.json"):
            old.unlink()
    by = Counter(v["split"] for v in ann.values())
    print(f"{path}: " + ", ".join(f"{s} {by[s]}" for s in ("train", "val", "test")))
    return ann


def main():
    lines = [r for r in csv.DictReader(open(CFG["lines"] / "lines.csv")) if r["train"] == "1" and "'" not in r["label"]]
    sheet = {r["page"]: f'{r["quire"]}-{r["bifolium"]}' for r in csv.DictReader(open(CFG["page_labels"]))}
    per_sheet = Counter(sheet[r["page"]] for r in lines)
    order = sorted(per_sheet)
    random.Random(SEED).shuffle(order)

    (OUT / "images").mkdir(parents=True, exist_ok=True)
    for r in lines:
        link = OUT / "images" / r["file"]
        if not link.exists():
            link.symlink_to((CFG["lines"] / "img" / r["file"]).resolve())
    print(f"{len(lines)} lines on {len(per_sheet)} sheets")

    main_split = fill(order, per_sheet, len(lines), {}, [("test", TEST), ("val", VAL)])
    ann = write(OUT, lines, sheet, main_split, "language")
    print("symbols:", "".join(sorted(set("".join(v["label"] for v in ann.values())) - {" "})))

    scribes = CFG["scribe_dataset"]
    scribes.mkdir(parents=True, exist_ok=True)
    if not (scribes / "images").is_symlink():
        (scribes / "images").symlink_to(Path("..") / OUT.name / "images", target_is_directory=True)
    write(scribes, lines, sheet, main_split, "davis_hand")


if __name__ == "__main__":
    main()
