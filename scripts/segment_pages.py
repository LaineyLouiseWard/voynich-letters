"""Find the text lines on each page with Kraken's default baseline segmenter, from the ink alone.

Pages: those with placed word boxes (kept pages of voynich-glyph-embeddings' aligned.csv). A page already segmented is
skipped. Most of Kraken's time goes on tracing line outlines on the CPU, about 13 seconds a page whether the network
runs on the GPU or not, so WORKERS Kraken processes run side by side, each loading the model once for its share of
pages. Kraken reports success even when a page fails (six processes sharing the 8 GB GPU lost 55 of 166 pages), so
pages without output are retried one at a time and any still missing are listed.

Outputs: data/seg/<page>.json (Kraken's baselines and line outlines in page-image pixels).
Usage: python scripts/segment_pages.py [--pages 103r 10r ...]
"""
import argparse
import csv
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
CFG = {k: ROOT / v for k, v in yaml.safe_load(open(ROOT / "config/paths.yaml")).items()}
KRAKEN = Path(sys.executable).parent / "kraken"
WORKERS = 4


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pages", nargs="*")
    args = ap.parse_args()
    image_of = {r["page"]: r["iipimage"] for r in csv.DictReader(open(CFG["word_boxes"]))
                if r["kept"] == "1" and (not args.pages or r["page"] in args.pages)}
    out = CFG["seg"]
    out.mkdir(parents=True, exist_ok=True)
    todo = [p for p in sorted(image_of) if not (out / f"{p}.json").exists()]

    def run(pages):
        pairs = [a for p in pages for a in ("-i", str(CFG["images"] / image_of[p]), str(out / f"{p}.json"))]
        subprocess.run([str(KRAKEN), "-d", "cuda:0", *pairs, "segment", "-bl"], check=True, capture_output=True)
        return len(pages)

    with ThreadPoolExecutor(WORKERS) as pool:
        list(pool.map(run, [todo[k::WORKERS] for k in range(WORKERS)]))
    for p in [p for p in todo if not (out / f"{p}.json").exists()]:
        run([p])
    missing = [p for p in todo if not (out / f"{p}.json").exists()]
    print(f"{len(todo) - len(missing)} of {len(todo)} pages segmented" + (f"; failed: {missing}" if missing else ""))


if __name__ == "__main__":
    main()
