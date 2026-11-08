"""Place every ZL letter of the dataset's lines on the page scan with a trained letter model.

The export covers the lines of the dataset's annotation.json (train, val and test sheets); the lines of lines.csv with
doubtful readings (train=0) or EVA's apostrophe are not in it. Each line is read as the authors' prediction script
reads it (predict.py from config/model.yaml's predict_commit, process_item: the dataset's resize keeping the line's
proportions, one line at a time, get_bboxes_scores, boxes at zero counted as no letter). Of the model's 100 boxes,
sorted left to right, the letter head (even rows; class 0 no letter, class k the dataset's letter k-1) carries the
line's encoded ZL text, word spaces included, by forced alignment (align): each symbol on one box, in order, the boxes
chosen to make the whole line most probable, every other box counted as no letter. Before any letter is kept, each
line's reading from the same scores (the letter head's most probable class per box) must equal predicted_text in the
run's predict/<line>.json, and the accent head must read no accent: the export then reads the model as predict.py did.

Line image to page scan: lines.csv keeps no position on the page, so each page's Kraken lines are assigned to its box
lines again with scripts/make_lines.py's own functions and extracted again from the grey page, recording the warp
Kraken applies (a mesh, or a rotation for two-point baselines). The same warp applied to maps of page x and y gives
each line-image pixel's place on the page; the rebuilt line image must equal the stored one. A letter's box on the
page is the bounding box of the page places of the line-image pixels inside its box.

Outputs (data/letters/): letters.csv, one row per letter (spaces are aligned, not exported): page, line, letter (its
index among the line's letters), zl_letter, model_letter (the letter head's most probable class at its box, empty for
no letter; composites written back as cth ckh cph cfh ch sh), model_prob (that class's probability), zl_prob (ZL's
letter's probability there), x0 y0 x1 y1 (page-scan pixels, x1 and y1 exclusive; empty where the box covers only a gap
between Kraken lines), split, scribe (Davis), language (Currier), section (illustration type in zl_page_labels.csv);
extra.csv, the boxes the model reads as a letter where the alignment places none (page, line, model_letter,
model_prob, x0 y0 x1 y1); crops/<line>_<letter>.png, the page scan inside the box at true size; square/<line>_<letter>.png,
that crop padded to a centred square on white and resized to SQUARE pixels by nearest neighbour, as in
voynich-glyph-embeddings' scripts/square_crops.py.
Usage: python scripts/read_letters.py RUN [--limit N]   (RUN: a run with step 1 and predict/ done)
"""
import argparse
import csv
import json
import os
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import letter_model  # noqa: E402
import make_lines  # noqa: E402
from kraken.containers import BaselineLine, Segmentation  # noqa: E402
from kraken.lib.segmentation import extract_polygons  # noqa: E402
from make_dataset import COMPOSITES  # noqa: E402

CFG = make_lines.CFG
OUT = ROOT / "data/letters"
DECODE = {cap: eva for eva, cap in COMPOSITES}
SQUARE = make_lines.T["SQUARE"]


def align(logp, targets, blank=0):
    """Forced alignment: for each target class in order, one row of logp (boxes left to right), the rows chosen to
    maximise the log probabilities of the target classes on their rows plus those of no letter (blank) on every other
    row. Returns the rows, or None when there are more targets than rows."""
    n, m = len(logp), len(targets)
    if m > n:
        return None
    gain = logp[:, targets] - logp[:, [blank]]
    best = np.full((n + 1, m + 1), -np.inf)
    best[:, 0] = 0
    for q in range(1, n + 1):
        best[q, 1:] = np.maximum(best[q - 1, 1:], best[q - 1, :-1] + gain[q - 1])
    rows, j = [], m
    for q in range(n, 0, -1):
        if j and best[q, j] != best[q - 1, j]:
            rows.append(q - 1)
            j -= 1
    return rows[::-1]


def load(run):
    """RUN's step-1 model, the dataset, and the code's box function, built as predict.py builds them."""
    letter_model.use_dataset()
    os.chdir(letter_model.CODE)
    sys.path.insert(0, str(letter_model.CODE))
    import util.misc as utils
    from datasets import build_dataset
    from main_synthetic import build_model_main
    from reconstruction.utils import get_bboxes_scores
    from util.slconfig import load_model
    c = letter_model.MODEL["common"]
    args = argparse.Namespace(dataset_file=c["dataset_file"], data_folder=c["data_folder"], split="all", document=None,
                              script=None, line=None, reverse_labels=False, old_data_augmentation=False,
                              preserve_line_aspect_ratio=True, line_resize_h_ref=c["line_resize_h_ref"],
                              line_resize_max_width=c["line_resize_max_width"])
    ds = build_dataset(image_set="train", args=args)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, _ = load_model(str(letter_model.CODE / c["model_config_path"]), device, str(letter_model.PATHS["runs"] / run / "step1/model.pth"),
                          build_model_main, ds.charset, expand_bbox=False, resume=False, num_fine_classes=2,
                          dataset_train=ds, init=False, num_sprites_per_letter=1)
    model = model.eval().to(device)

    def read(i):
        """Letter-head and accent-head scores (rows 0::2 and 1::2), the letter-head boxes (cx, cy, w, h) in line-image
        pixels, as predict.py computes them."""
        samples, targets = utils.collate_fn([ds[i]])
        with torch.no_grad():
            bboxes, scores, *_ = get_bboxes_scores(samples.to(device), model, num_fine_classes=c["num_fine_classes"])
        s, b = scores[0].cpu().numpy(), bboxes[0].cpu().numpy()
        bad = (np.abs(b[:, 0]) <= 1e-3) | (np.abs(b[:, 1]) <= 1e-3)
        s = s.copy()
        s[bad, 0], s[bad, 1:] = 1.0, 1e-6
        b = np.maximum(b, 1.0)
        size, orig = targets[0]["size"].numpy(), targets[0]["orig_size"].numpy()
        r_h, r_w = orig[0] / size[0], orig[1] / size[1]
        return s[0::2], s[1::2], b[0::2] * [r_w, r_h, r_w, r_h]

    return ds, read


def warped(gray, segs):
    """Each Kraken line in segs extracted from gray as in make_lines, with its page x and y maps warped the same way."""
    calls, transform = [], Image.Image.transform

    def spy(self, size, method, data=None, *args, **kwargs):
        out = transform(self, size, method, data, *args, **kwargs)
        calls.append((self.size, size, method, data, out))
        return out

    seg = Segmentation(type="baselines", imagename="page", text_direction="horizontal-lr", script_detection=False,
                       lines=[BaselineLine(id=str(k), baseline=s["baseline"], boundary=s["boundary"])
                              for k, s in enumerate(segs)])
    Image.Image.transform = spy
    try:
        images = [np.array(im.convert("L")) for im, _ in extract_polygons(gray, seg)]
    finally:
        Image.Image.transform = transform
    assert len(calls) == len(segs)
    parts = []
    for s, im, (patch, size, method, data, out) in zip(segs, images, calls):
        box, bound = out.getbbox(), np.array(s["boundary"])
        rows, cols = np.indices(patch[::-1], dtype=np.float32)
        maps = []
        for grid, origin in ((cols, int(bound[:, 0].min())), (rows, int(bound[:, 1].min()))):
            a = np.array(Image.fromarray(grid).transform(size, method, data, Image.Resampling.NEAREST,
                                                         fillcolor=-1).crop(box))
            maps.append(np.where(a < 0, np.nan, a + origin))
        parts.append((make_lines.parchment_outside(im), *maps))
    return parts


def line_maps(gray, segs, gap, stored):
    """Page x and y of each pixel of a line image (NaN in the gaps), rebuilt as make_lines joins its Kraken lines."""
    parts = warped(gray, segs)
    img = make_lines.join([p[0] for p in parts], gap)
    assert np.array_equal(img, stored), "rebuilt line image differs from the stored one"
    xs, ys = np.full(img.shape, np.nan, np.float32), np.full(img.shape, np.nan, np.float32)
    left = 0
    for im, mx, my in parts:
        top = (img.shape[0] - im.shape[0]) // 2
        xs[top:top + im.shape[0], left:left + im.shape[1]] = mx
        ys[top:top + im.shape[0], left:left + im.shape[1]] = my
        left += im.shape[1] + gap
    return xs, ys


def square(crop):
    side = max(crop.size)
    out = Image.new(crop.mode, (side, side), "white")
    out.paste(crop, ((side - crop.width) // 2, (side - crop.height) // 2))
    return out.resize((SQUARE, SQUARE), Image.Resampling.NEAREST)


def page_box(box, xs, ys):
    """Page-scan box (x0, y0, x1, y1) of the line-image pixels inside a (cx, cy, w, h) box, or None over a gap only."""
    cx, cy, w, h = box
    H, W = xs.shape
    x0, x1 = max(0, int(np.floor(cx - w / 2))), min(W, int(np.ceil(cx + w / 2)))
    y0, y1 = max(0, int(np.floor(cy - h / 2))), min(H, int(np.ceil(cy + h / 2)))
    px, py = xs[y0:y1, x0:x1], ys[y0:y1, x0:x1]
    px, py = px[~np.isnan(px)], py[~np.isnan(py)]
    if not len(px):
        return None
    return int(px.min()), int(py.min()), int(px.max()) + 1, int(py.max()) + 1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run")
    ap.add_argument("--limit", type=int)
    args = ap.parse_args()
    predicted = letter_model.PATHS["runs"] / args.run / "predict"
    if not (predicted / "made.json").exists():
        sys.exit(f"{predicted} has no made.json: run scripts/letter_model.py predict {args.run} first")

    ann = json.load(open(CFG["dataset"] / "annotation.json"))
    lines = {r["file"]: r for r in csv.DictReader(open(CFG["lines"] / "lines.csv"))}
    section = {r["page"]: r["illustration"] for r in csv.DictReader(open(CFG["page_labels"]))}
    words, image_of = defaultdict(lambda: defaultdict(list)), {}
    for r in csv.DictReader(open(CFG["word_boxes"])):
        if r["kept"] == "1":
            words[r["page"]][int(r["line"])].append(
                (int(r["index"]), r["text"], *(float(r[k]) for k in ("x0", "y0", "x1", "y1"))))
            image_of[r["page"]] = r["iipimage"]
    by_page = defaultdict(list)
    for name in list(ann)[:args.limit]:
        by_page[ann[name]["page"]].append(name)

    ds, read = load(args.run)
    charset = ds.charset
    index = {key: i for i, (key, _) in enumerate(ds.data)}
    show = lambda k: "" if k == 0 else DECODE.get(charset[k - 1], charset[k - 1])
    for d in ("crops", "square"):
        (OUT / d).mkdir(parents=True, exist_ok=True)
    rows, extra, gaps = [], [], 0
    for page, names in by_page.items():
        scan = Image.open(CFG["images"] / image_of[page])
        gray = scan.convert("L").point(lambda v: max(v, 1))
        segs = json.load(open(CFG["seg"] / f"{page}.json"))["lines"]
        box_lines = [sorted(words[page][k]) for k in sorted(words[page])]
        cen = [np.mean([(w[3] + w[5]) / 2 for w in ln]) for ln in box_lines]
        spacing = float(np.median(np.diff(sorted(cen))))
        owned, _ = make_lines.assign(box_lines, segs, spacing)
        for name in names:
            r, text = lines[name], ann[name]["label"]
            letter, accent, boxes = read(index[name])
            free = letter.argmax(1)
            reading = "".join(charset[k - 1] for k in free if k)
            saved = json.load(open(predicted / f"{Path(name).stem}.json"))["predicted_text"]
            if accent.argmax(1).any() or reading != saved:
                sys.exit(f"{name}: the model's reading here ({reading!r}) is not predict.py's ({saved!r}), or the "
                         f"accent head reads an accent: the export would not read the model as predict.py does")
            logp = np.log(np.clip(letter, 1e-12, None))
            targets = [charset.index(ch) + 1 for ch in text]
            placed = align(logp, targets)
            if placed is None:
                sys.exit(f"{name}: {len(targets)} symbols, more than the model's {len(logp)} boxes")
            mine =sorted(owned[int(r["box_line"])][0], key=lambda si: min(p[0] for p in segs[si]["baseline"]))
            stored = np.array(Image.open(CFG["lines"] / "img" / name))
            xs, ys = line_maps(gray, [segs[si] for si in mine], int(spacing / 2), stored)
            stem, j = Path(name).stem, 0
            for ch, col, q in zip(text, targets, placed):
                if ch == " ":
                    continue
                k = int(free[q])
                row = {"page": page, "line": stem, "letter": j, "zl_letter": DECODE.get(ch, ch),
                       "model_letter": show(k), "model_prob": round(float(letter[q, k]), 4),
                       "zl_prob": round(float(letter[q, col]), 4), "x0": "", "y0": "", "x1": "", "y1": "",
                       "split": ann[name]["split"], "scribe": r["davis_hand"], "language": r["language"],
                       "section": section.get(page, "")}
                box = page_box(boxes[q], xs, ys)
                if box:
                    row.update(zip(("x0", "y0", "x1", "y1"), box))
                    crop = scan.crop(box)
                    crop.save(OUT / "crops" / f"{stem}_{j:03d}.png")
                    square(crop).save(OUT / "square" / f"{stem}_{j:03d}.png")
                else:
                    gaps += 1
                rows.append(row)
                j += 1
            for q in sorted(set(np.flatnonzero(free)) - set(placed)):
                k = int(free[q])
                box = page_box(boxes[q], xs, ys) or ("", "", "", "")
                extra.append({"page": page, "line": stem, "model_letter": show(k),
                              "model_prob": round(float(letter[q, k]), 4), **dict(zip(("x0", "y0", "x1", "y1"), box))})

    extra_fields = ["page", "line", "model_letter", "model_prob", "x0", "y0", "x1", "y1"]
    for f, data, fields in (("letters.csv", rows, list(rows[0])), ("extra.csv", extra, extra_fields)):
        with open(OUT / f, "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=fields)
            w.writeheader()
            w.writerows(data)
    agree = sum(r["model_letter"] == r["zl_letter"] for r in rows)
    print(f"{len(rows)} letters on {len({r['line'] for r in rows})} lines; the model's letter at the box equals ZL's for "
          f"{agree} ({agree / len(rows):.1%}); {gaps} boxes over a gap only (no page box); {len(extra)} boxes read as a "
          f"letter where ZL has none")


if __name__ == "__main__":
    main()
