#!/bin/bash
# Run on the laptop. Copies the committed code and configs (stale files removed) and the dataset (links resolved to real
# files) to Sonic scratch, and records the commit in .synced_commit for the results' stamp (Sonic's copy has no git).
# Stops if code differs from the commit. HOST is sonic on campus, sonic-home off campus.
set -euo pipefail
HOST=${1:-sonic}
cd "$(dirname "$0")/.."
if [ -n "$(git status --porcelain -- scripts config cluster Makefile)" ]; then
    echo "uncommitted changes in scripts, config, cluster or Makefile: commit first"; exit 1
fi
DATASET=$(sed -n 's/^dataset: *//p' config/paths.yaml)
rsync -az --delete --exclude data/ --exclude external/ --exclude .git/ --exclude env/ --exclude logs/ \
    ./ "$HOST:scratch/voynich-letters/"
git rev-parse --short HEAD | ssh "$HOST" "cat > scratch/voynich-letters/.synced_commit"
ssh "$HOST" mkdir -p "scratch/voynich-letters/$DATASET"
rsync -azL --delete "$DATASET/" "$HOST:scratch/voynich-letters/$DATASET/"
# the scribe dataset: its images/ stays a link to the dataset's images, which are real files there
SCRIBES=$(sed -n 's/^scribe_dataset: *\([^ ]*\).*/\1/p' config/paths.yaml)
ssh "$HOST" mkdir -p "scratch/voynich-letters/$SCRIBES"
rsync -az --delete "$SCRIBES/" "$HOST:scratch/voynich-letters/$SCRIBES/"
