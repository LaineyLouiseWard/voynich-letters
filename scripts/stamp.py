"""What a result was made with: the md5 of a file, and the git commit of the code (marked when scripts/, config/,
cluster/ or the Makefile differ from it, so a result from uncommitted code shows as such).
Usage: python scripts/stamp.py FILE ...   (prints each file's md5, then the commit)
"""
import hashlib
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def md5(path):
    h = hashlib.md5()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def commit():
    def git(*a):
        return subprocess.run(["git", *a], cwd=ROOT, capture_output=True, text=True)
    try:
        head = git("rev-parse", "--short", "HEAD")
    except FileNotFoundError:  # no git program (Sonic's compute nodes)
        head = subprocess.CompletedProcess([], 1)
    if head.returncode:  # a copy without git history (Sonic): the commit cluster/sync.sh recorded
        synced = ROOT / ".synced_commit"
        return synced.read_text().strip() if synced.exists() else "unknown"
    changed = git("status", "--porcelain", "--", "scripts", "config", "cluster", "Makefile").stdout.strip()
    return head.stdout.strip() + ("+changes" if changed else "")


if __name__ == "__main__":
    for name in sys.argv[1:]:
        print(f"md5 {md5(name)} {name}")
    print(f"commit {commit()}")
