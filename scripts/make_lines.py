"""Make one grey image per ZL paragraph line, from Kraken's lines, on the pages with placed word boxes.

Matching text: each word-box line (kept pages of voynich-glyph-embeddings' aligned.csv) is matched to a ZL line of the
same page by text similarity, keeping page order (dynamic programming; a match needs similarity >= MIN_SIM). Only ZL
paragraph loci (type P) are kept.

Matching ink: Kraken's lines (scripts/segment_pages.py) follow the ink; the word boxes are placed less exactly (up to
half a line off on some pages, drifting further toward one side of the page), so they are used only to say which
Kraken line is which. Each box is moved by an offset fitted as a line across the page, a + b*x, to the boxes a first
pass (one constant offset, the median distance from a box centre to the nearest Kraken baseline) already assigns to
their own box line's Kraken line, with one round of outlier rejection; a page with too few such boxes (LINE_MIN)
keeps the constant offset instead. Each box then votes for the Kraken line whose baseline is nearest after its own
offset, if within VOTE_TOL line spacings. A Kraken line belongs to the box line with most of its votes, if at least
AGREE of its votes agree. A box line is kept when its own Kraken lines hold at least COVER of its words' votes and do
not overlap left to right; its Kraken lines (more than one where a drawing splits the line) are straightened by
Kraken's line extractor and joined left to right with a gap of half a line spacing. Pixels outside Kraken's outline
are set to parchment grey.

Width test: a line's image width per letter of its label is compared with its page's median. A line outside
WIDTH_BAND is dropped, and a page with more than PAGE_FAIL of its lines outside is dropped whole: if a page's boxes sat
a whole line off, every line would carry its neighbour's text, which the voting above cannot see, but about a third of
such lines fall outside the band against about 1 in 100 correct lines.

Label: the ZL text with tags and comments removed, a gap for a drawing (<->) and both certain (.) and uncertain (,)
spaces written as a space. Lines with an unreadable glyph (?), an alternative reading ([a:b]), a rare-glyph code
(@nnn;) or a ligature brace ({...}) are kept in the table with train=0.

Outputs: <out>/lines.csv, <out>/img/<page>_<line>.png, <out>/dropped.csv (each box line not kept, with the reason);
a per-page summary is printed. <out> defaults to the configured data/lines.
Usage: python scripts/make_lines.py [--pages 103r 1r ...] [--out data/lines_winback]
"""
import argparse
import csv
import json
import re
from collections import Counter, defaultdict
from difflib import SequenceMatcher
from pathlib import Path

import numpy as np
import yaml
from kraken.containers import BaselineLine, Segmentation
from kraken.lib.segmentation import extract_polygons
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
CFG = {k: ROOT / v for k, v in yaml.safe_load(open(ROOT / "config/paths.yaml")).items()}
T = yaml.safe_load(open(ROOT / "config/thresholds.yaml"))
MIN_SIM, VOTE_TOL, AGREE, COVER, PAGE_FAIL, LINE_MIN = (T[k] for k in ("MIN_SIM", "VOTE_TOL", "AGREE", "COVER",
                                                                       "PAGE_FAIL", "LINE_MIN"))
WIDTH_BAND = tuple(T["WIDTH_BAND"])
SHIFT = 0           # testing only: give each line the text of the ZL line SHIFT places later
UNSURE = re.compile(r"\?|\[|@\d+;|\{")


def zl_lines(path=None):
    """ZL loci by page: (line number, locus type, raw text); path: another IVTFF file read the same way."""
    out = defaultdict(list)
    for row in open(path or CFG["zl"], encoding="latin-1"):
        m = re.match(r"<f(\w+)\.(\d+),(.)(\w+)>\s*(.*)", row.rstrip("\n"))
        if m:
            out[m.group(1)].append((int(m.group(2)), m.group(4), m.group(5)))
    return out


def clean(text):
    """ZL text without tags, comments or braces, alternatives as their first reading, spaces as '.'."""
    t = re.sub(r"<[^>]*>", "", text.replace("<->", "."))
    t = re.sub(r"\[([^:\]]*):[^\]]*\]", r"\1", t)
    return t.replace("{", "").replace("}", "").replace(",", ".")


def match(box_texts, zl_texts):
    """Pairs (box index, ZL index, similarity) in page order, maximising total similarity above MIN_SIM."""
    nb, nz = len(box_texts), len(zl_texts)
    sim = [[SequenceMatcher(None, b, z).ratio() for z in zl_texts] for b in box_texts]
    gain = lambda i, j: sim[i][j] - MIN_SIM if sim[i][j] >= MIN_SIM else -np.inf
    best = np.zeros((nb + 1, nz + 1))
    for i in range(1, nb + 1):
        for j in range(1, nz + 1):
            best[i, j] = max(best[i - 1, j], best[i, j - 1], best[i - 1, j - 1] + gain(i - 1, j - 1))
    pairs, i, j = [], nb, nz
    while i and j:
        if np.isclose(best[i, j], best[i - 1, j - 1] + gain(i - 1, j - 1)):
            pairs.append((i - 1, j - 1, sim[i - 1][j - 1]))
            i, j = i - 1, j - 1
        elif best[i, j] == best[i - 1, j]:
            i -= 1
        else:
            j -= 1
    return pairs[::-1]


def base_y(line, x):
    """Height of a Kraken baseline at x, or None if x is outside it."""
    pts = np.array(sorted(line["baseline"]), float)
    return None if x < pts[0, 0] or x > pts[-1, 0] else float(np.interp(x, pts[:, 0], pts[:, 1]))


def vote_owner(words, segs, spacing, offs):
    """Word index -> Kraken line index for each box within VOTE_TOL of its offset baseline, and Kraken line
    index -> box line index where at least AGREE of a Kraken line's votes name the same box line."""
    votes = {}
    for wi, (_, x, y) in enumerate(words):
        d = [(abs(y - offs[wi] - b), si) for si, b in ((si, base_y(s, x)) for si, s in enumerate(segs))
             if b is not None]
        if d and min(d)[0] <= VOTE_TOL * spacing:
            votes[wi] = min(d)[1]
    owner = {}
    for si in set(votes.values()):
        c = Counter(words[wi][0] for wi, s in votes.items() if s == si)
        li, n = c.most_common(1)[0]
        if n / sum(c.values()) >= AGREE:
            owner[si] = li
    return votes, owner


def line_offset(words, segs, spacing):
    """Each box's offset as a line a + b*(x-x̄) across the page (the drift grows toward one side on some pages, not
    with height), fitted to the boxes a first pass (one constant offset, the median box-to-nearest-baseline
    distance) already assigns to their own box line's Kraken line, outliers beyond VOTE_TOL spacings of that fit
    dropped before a second fit. Falls back to the constant alone when fewer than LINE_MIN boxes agree with the
    first pass."""
    near = []
    for _, x, y in words:
        d = [y - b for b in (base_y(s, x) for s in segs) if b is not None]
        if d:
            near.append(min(d, key=abs))
    const = float(np.median(near)) if near else 0.0
    votes, owner = vote_owner(words, segs, spacing, [const] * len(words))
    pts = []
    for wi, si in votes.items():
        li, x, y = words[wi]
        if owner.get(si) == li:
            pts.append((x, y - base_y(segs[si], x)))
    if len(pts) < LINE_MIN:
        return lambda x: const
    xs, rs = np.array(pts).T
    xm = xs.mean()
    A = np.column_stack([np.ones_like(xs), xs - xm])
    coef = np.linalg.lstsq(A, rs, rcond=None)[0]
    keep = np.abs(rs - A @ coef) <= VOTE_TOL * spacing
    if LINE_MIN <= keep.sum() < len(pts):
        coef = np.linalg.lstsq(A[keep], rs[keep], rcond=None)[0]
    a, b = coef
    return lambda x: a + b * (x - xm)


def assign(lines, segs, spacing):
    """Kraken lines of each box line, its words' coverage, and the page's mean offset."""
    words = [(li, (w[2] + w[4]) / 2, (w[3] + w[5]) / 2) for li, ln in enumerate(lines) for w in ln]
    offset = line_offset(words, segs, spacing)
    offs = [offset(x) for _, x, _ in words]
    votes, owner = vote_owner(words, segs, spacing, offs)
    out = {}
    for li in range(len(lines)):
        mine = [si for si, o in owner.items() if o == li]
        idx = [wi for wi, w in enumerate(words) if w[0] == li]
        cover = sum(votes.get(wi) in mine for wi in idx) / len(idx)
        out[li] = (mine, cover)
    return out, float(np.mean(offs)) if offs else 0.0


def overlap(segs):
    """True if any two Kraken lines share more than a fifth of the shorter one's width."""
    spans = sorted((min(p[0] for p in s["baseline"]), max(p[0] for p in s["baseline"])) for s in segs)
    return any(min(a1, b1) - b0 > 0.2 * min(a1 - a0, b1 - b0) for (a0, a1), (b0, b1) in zip(spans, spans[1:]))


def parchment_outside(im):
    """Set the pixels outside Kraken's line outline (0 here, as the page was raised to at least 1) to parchment grey."""
    out = im.copy()
    out[im == 0] = np.percentile(im[im > 0], 90)
    return out


def join(images, gap):
    """Join line images left to right, centred vertically, with parchment-grey gaps."""
    h = max(im.shape[0] for im in images)
    fill = int(np.percentile(np.concatenate([im.ravel() for im in images]), 90))
    parts = []
    for k, im in enumerate(images):
        top = (h - im.shape[0]) // 2
        canvas = np.full((h, im.shape[1]), fill, np.uint8)
        canvas[top:top + im.shape[0]] = im
        parts += ([np.full((h, gap), fill, np.uint8)] if k else []) + [canvas]
    return np.hstack(parts)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pages", nargs="*")
    ap.add_argument("--out", type=Path, default=CFG["lines"])
    args = ap.parse_args()

    words = defaultdict(lambda: defaultdict(list))
    image_of = {}
    for r in csv.DictReader(open(CFG["word_boxes"])):
        if r["kept"] == "1" and (not args.pages or r["page"] in args.pages):
            words[r["page"]][int(r["line"])].append(
                (int(r["index"]), r["text"], *(float(r[k]) for k in ("x0", "y0", "x1", "y1"))))
            image_of[r["page"]] = r["iipimage"]
    labels = {r["page"]: r for r in csv.DictReader(open(CFG["page_labels"]))}
    zl = zl_lines()
    out = args.out
    (out / "img").mkdir(parents=True, exist_ok=True)

    rows, dropped = [], []
    print("page  box_lines  zl_matched  kept  offset/spacing")
    for page in sorted(words):
        seg_file = CFG["seg"] / f"{page}.json"
        if not seg_file.exists():
            continue
        segs = json.load(open(seg_file))["lines"]
        lines = [sorted(words[page][k]) for k in sorted(words[page])]
        cen = [np.mean([(w[3] + w[5]) / 2 for w in ln]) for ln in lines]
        spacing = float(np.median(np.diff(sorted(cen))))
        para = [z for z in zl.get(page, []) if z[1].startswith("P")]
        pairs = match([".".join(w[1] for w in ln) for ln in lines], [clean(z[2]) for z in para])
        owned, offset = assign(lines, segs, spacing)
        gray = Image.open(CFG["images"] / image_of[page]).convert("L").point(lambda v: max(v, 1))
        found = []
        matched = {bi for bi, _, _ in pairs}
        dropped += [{"page": page, "box_line": bi, "zl_line": "", "reason": "no ZL paragraph line"}
                    for bi in range(len(lines)) if bi not in matched]
        for bi, zi, sim in pairs:
            mine, cover = owned[bi]
            why = ("no Kraken line" if not mine else "words off its Kraken lines" if cover < COVER
                   else "Kraken lines overlap" if overlap([segs[si] for si in mine]) else "")
            if why:
                dropped.append({"page": page, "box_line": bi, "zl_line": para[zi][0], "reason": why})
                continue
            mine.sort(key=lambda si: min(p[0] for p in segs[si]["baseline"]))
            seg = Segmentation(type="baselines", imagename=str(CFG["images"] / image_of[page]),
                               text_direction="horizontal-lr", script_detection=False,
                               lines=[BaselineLine(id=str(si), baseline=segs[si]["baseline"],
                                                   boundary=segs[si]["boundary"]) for si in mine])
            parts = {ln.id: parchment_outside(np.array(im.convert("L"))) for im, ln in extract_polygons(gray, seg)}
            img = join([parts[str(si)] for si in mine], int(spacing / 2))
            num, loc, raw = para[min(zi + SHIFT, len(para) - 1)]
            name = f"{page}_{num:02d}.png"
            lab = labels.get(page, {})
            found.append((img, {"file": name, "page": page, "zl_line": num, "locus": loc, "box_line": bi,
                         "n_words": len(lines[bi]), "similarity": round(sim, 3),
                         "label": clean(raw).replace(".", " ").strip(),
                         "train": int(not UNSURE.search(re.sub(r"<[^>]*>", "", raw))),
                         "davis_hand": lab.get("davis_hand", ""), "language": lab.get("language", ""),
                         "n_segments": len(mine), "cover": round(cover, 3),
                         "height": img.shape[0], "width": img.shape[1], "spacing": round(spacing, 1)}))
        per = [r["width"] / max(1, len(r["label"].replace(" ", ""))) for _, r in found]
        ratio = np.array(per) / np.median(per) if per else np.array([])
        inside = (ratio >= WIDTH_BAND[0]) & (ratio <= WIDTH_BAND[1])
        page_ok = bool(len(found)) and np.mean(~inside) <= PAGE_FAIL
        kept = 0
        for (img, r), q, ok in zip(found, ratio, inside):
            if not (page_ok and ok):
                dropped.append({"page": page, "box_line": r["box_line"], "zl_line": r["zl_line"],
                                "reason": "page fails width test" if not page_ok else "line fails width test"})
            else:
                r["width_ratio"] = round(float(q), 3)
                Image.fromarray(img).save(out / "img" / r["file"])
                rows.append(r)
                kept += 1
        print(f"{page:6}{len(lines):9}{len(pairs):12}{kept:7}{offset / spacing:12.2f}"
              + ("" if page_ok or not found else f"   dropped: {np.mean(~inside):.0%} of lines outside the width band"))
    with open(out / "lines.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=rows[0].keys())
        w.writeheader()
        w.writerows(rows)
    with open(out / "dropped.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=dropped[0].keys())
        w.writeheader()
        w.writerows(dropped)
    print(f"{len(rows)} lines kept; train=1: {sum(r['train'] for r in rows)}")
    for why, n in Counter(d["reason"] for d in dropped).most_common():
        print(f"dropped, {why}: {n}")


if __name__ == "__main__":
    main()
