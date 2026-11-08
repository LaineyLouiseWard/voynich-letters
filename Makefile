# One command per laptop step, in order; run from the voynich-letters conda env. Training runs on Sonic (README).
.NOTPARALLEL:
# Every recipe runs the env's own python by its full path: a bare `python` can be another env's.
ENV := voynich-letters
ifneq ($(notdir $(CONDA_PREFIX)),$(ENV))
$(error activate the $(ENV) conda env before make (conda activate $(ENV)); CONDA_PREFIX is '$(CONDA_PREFIX)')
endif
PYTHON := $(CONDA_PREFIX)/bin/python

.PHONY: lines dataset model takahashi reading letters figures
TAKAHASHI := data/raw/IT2a-n.txt
lines:                                  ## Kraken lines for the 166 pages, matched to ZL
	$(PYTHON) scripts/segment_pages.py
	$(PYTHON) scripts/make_lines.py
dataset: lines                          ## the letter model's dataset, split by parchment sheet
	$(PYTHON) scripts/make_dataset.py
model:                                  ## the letter model's code (patched, compiled for the laptop's RTX 5060) and detector
	PY=$(PYTHON) TORCH_CUDA_ARCH_LIST=12.0 bash scripts/setup_model.sh
takahashi: $(TAKAHASHI)                 ## Takahashi's EVA transcription (voynich.nu, README of 24 Mar 2026)
$(TAKAHASHI):
	mkdir -p $(@D) && curl -fsS -o $@.part https://www.voynich.nu/data/IT2a-n.txt
	echo "0e7b3a6a3053c7d35a52feed8ff103de  $@.part" | md5sum -c --quiet && mv $@.part $@
RUN ?= reference
reading: $(TAKAHASHI) data/runs/$(RUN)/predict/made.json   ## ZL, Takahashi and RUN's model compared pairwise on the test lines
	$(PYTHON) scripts/test_reading.py
	$(PYTHON) scripts/reading.py takahashi zl
	$(PYTHON) scripts/reading.py $(RUN) zl
	$(PYTHON) scripts/reading.py $(RUN) takahashi
data/runs/$(RUN)/predict/made.json:     ## RUN's model reads every line of the dataset
	$(PYTHON) scripts/letter_model.py predict $(RUN)
letters: data/runs/$(RUN)/predict/made.json   ## every ZL letter placed on the page scan by RUN's model, into data/letters/
	$(PYTHON) scripts/test_read_letters.py
	$(PYTHON) scripts/read_letters.py $(RUN)
figures:                                ## the README's figures, from RUN's letter pictures and data/letters/
	$(PYTHON) scripts/figures.py $(RUN)
