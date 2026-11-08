"""Train the letter model (morphology4metrology, config/model.yaml) on the line dataset, and read the dataset with it.

  python scripts/letter_model.py train RUN STEP [extra reconstruction.py arguments]
  python scripts/letter_model.py predict RUN [extra predict.py arguments]
  python scripts/letter_model.py finetune RUN NAME [extra reconstruction.py arguments]

predict: RUN's step-1 model reads every line of the dataset with the authors' prediction script, one line at a time
(no padding), into data/runs/RUN/predict/: <line>.json per line (its reading, and each symbol's box, letter and
confidence), the letter pictures (sprite_final/, grid.jpg) and made.json.

finetune: the authors' step 2 per script (config/model.yaml step2), run on the scribe dataset (config/paths.yaml
scribe_dataset, "script" = Davis's scribe): RUN's step-1 letter pictures re-tuned on each scribe's training lines in
turn (config/model.yaml scribes), into data/runs/RUN/NAME/<scribe>/, with made.json in data/runs/RUN/NAME/. The scribe
dataset gets the dataset's letter list (charset*.json), so each scribe's pictures keep the step-1 model's letter order.

STEP 0 starts from the authors' pretrained detector; STEP 1 from RUN's step 0. Output: data/runs/RUN/stepN/ (the
code's own files) and made.json there: the full argument list, the code commit, the md5 of the patch, the detector's
sha256 and this repo's commit. Extra arguments go last, so they override config/model.yaml: for test runs only
(e.g. --max_e 0 for one pass). A step that already has a made.json is not rerun.
"""
import hashlib
import json
import shutil
import subprocess
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from stamp import commit, md5  # noqa: E402

PATHS = {k: ROOT / v for k, v in yaml.safe_load(open(ROOT / "config/paths.yaml")).items()}
MODEL = yaml.safe_load(open(ROOT / "config/model.yaml"))
CODE = PATHS["model_code"]


def as_args(settings):
    out = []
    for key, value in settings.items():
        if value is True:
            out.append(f"--{key}")
        elif value is not False:
            out += [f"--{key}", str(value)]
    return out


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def code_commit():
    """Read from .git/HEAD itself: Sonic's compute nodes have no git program."""
    head = CODE / ".git/HEAD"
    if not head.exists():
        return "unknown"
    ref = head.read_text().strip()
    if ref.startswith("ref: "):
        ref_file = CODE / ".git" / ref[5:]
        return ref_file.read_text().strip() if ref_file.exists() else "unknown"
    return ref


def use_dataset():
    """The code reads its datasets root from datasets/config.json and writes to logs_reconstruction/<tag>/."""
    dataset = PATHS["dataset"]
    if dataset.name != MODEL["common"]["data_folder"]:
        sys.exit(f"config/paths.yaml dataset ends in {dataset.name}, config/model.yaml data_folder is "
                 f"{MODEL['common']['data_folder']}")
    json.dump({"datasets_path": str(dataset.parent)}, open(CODE / "datasets/config.json", "w"))
    logs = CODE / "logs_reconstruction"
    PATHS["runs"].mkdir(parents=True, exist_ok=True)
    if not logs.is_symlink():
        logs.symlink_to(PATHS["runs"])


def predict(run, extra):
    model, out = PATHS["runs"] / run / "step1", PATHS["runs"] / run / "predict"
    if (out / "made.json").exists():
        sys.exit(f"{out} already has a made.json: delete the folder to read again")
    if code_commit() != MODEL["commit"]:
        sys.exit(f"{CODE} is at {code_commit()}, not {MODEL['commit']}: run scripts/setup_model.sh")
    if not (CODE / "predict.py").read_text().startswith('"""\nPrediction script'):
        sys.exit(f"{CODE / 'predict.py'} is not the prediction script: run scripts/setup_model.sh")
    for f in ("model.pth", "reconstructor_unfrozen.pth", "made.json"):
        if not (model / f).exists():
            sys.exit(f"missing {model / f}: RUN's step 1 has not finished")
    use_dataset()
    keep = ("dataset_file", "data_folder", "model_config_path", "num_fine_classes", "line_resize_h_ref",
            "line_resize_max_width")
    args = as_args({**{k: MODEL["common"][k] for k in keep}, "single_model": True, "full_dataset": True,
                    "path_models": model, "output_dir": out, "batch_size": 1, "generate_grid": True}) + extra
    made = {"args": args, "code": MODEL["code"], "code_commit": code_commit(), "predict_commit": MODEL["predict_commit"],
            "predict_md5": md5(CODE / "predict.py"), "patch_md5": md5(ROOT / "scripts/morphology4metrology.patch"),
            "model_md5": md5(model / "model.pth"), "repo_commit": commit()}
    print(json.dumps(made, indent=1), flush=True)
    done = subprocess.run([sys.executable, "-u", "predict.py", *args], cwd=CODE)
    if done.returncode:
        sys.exit(done.returncode)
    lines = json.load(open(PATHS["dataset"] / "annotation.json"))
    missing = [k for k in lines if not (out / f"{Path(k).stem}.json").exists()]
    if missing:
        sys.exit(f"{len(missing)} of {len(lines)} lines were not read, e.g. {missing[:3]}")
    json.dump(made, open(out / "made.json", "w"), indent=1)


def finetune(run, name, extra):
    model, out = PATHS["runs"] / run / "step1", PATHS["runs"] / run / name
    if (out / "made.json").exists():
        sys.exit(f"{out} already has a made.json: choose a new NAME")
    if code_commit() != MODEL["commit"]:
        sys.exit(f"{CODE} is at {code_commit()}, not {MODEL['commit']}: run scripts/setup_model.sh")
    for f in ("model.pth", "reconstructor_unfrozen.pth", "made.json"):
        if not (model / f).exists():
            sys.exit(f"missing {model / f}: RUN's step 1 has not finished")
    use_dataset()
    scribes = PATHS["scribe_dataset"]
    if scribes.parent != PATHS["dataset"].parent or not (scribes / "annotation.json").exists():
        sys.exit(f"{scribes} must hold annotation.json beside {PATHS['dataset']}: run scripts/make_dataset.py")
    for f in ("charset.json", "charset_without_accent.json", "charset_accent.json"):
        ref, mine = PATHS["dataset"] / f, scribes / f
        if not ref.exists():
            sys.exit(f"missing {ref}: the dataset's letter list, written when the model first reads the dataset")
        if not mine.exists():
            shutil.copyfile(ref, mine)
        if mine.read_bytes() != ref.read_bytes():
            sys.exit(f"{mine} differs from {ref}: the scribe pictures would not match the step-1 letters")
    args = as_args({**MODEL["common"], "data_folder": scribes.name, **MODEL["step2"],
                    "model_checkpoint_path": model / "model.pth",
                    "prototypes_only_path": model / "reconstructor_unfrozen.pth", "step": 2, "tag": f"{run}/{name}"})
    args += ["--script", *MODEL["scribes"]] + extra
    made = {"args": args, "code": MODEL["code"], "code_commit": code_commit(),
            "patch_md5": md5(ROOT / "scripts/morphology4metrology.patch"), "model_md5": md5(model / "model.pth"),
            "repo_commit": commit()}
    print(json.dumps(made, indent=1), flush=True)
    done = subprocess.run([sys.executable, "-u", "reconstruction.py", *args], cwd=CODE)
    if done.returncode:
        sys.exit(done.returncode)
    missing = [f"{s}/{f}" for s in MODEL["scribes"] for f in ("model.pth", "reconstructor.pth")
               if not (out / s / f).exists()]
    if missing:
        sys.exit(f"step 2 did not write {missing} in {out}")
    json.dump(made, open(out / "made.json", "w"), indent=1)


def train(run, step, extra):
    out = PATHS["runs"] / run / f"step{step}"
    if (out / "made.json").exists():
        sys.exit(f"{out} already has a made.json: choose a new RUN name")
    if code_commit() != MODEL["commit"]:
        sys.exit(f"{CODE} is at {code_commit()}, not {MODEL['commit']}: run scripts/setup_model.sh")
    checkpoint = PATHS["checkpoint"]
    if step == 0:
        start = {"model_checkpoint_path": checkpoint}
    else:
        before = PATHS["runs"] / run / "step0"
        start = {"model_checkpoint_path": before / "model.pth", "reconstructor_path": before / "reconstructor.pth"}
    for p in start.values():
        if not p.exists():
            sys.exit(f"missing {p}")
    use_dataset()
    args =as_args({**MODEL["common"], **MODEL[f"step{step}"], **start, "step": step, "tag": f"{run}/step{step}"}) + extra
    out.mkdir(parents=True, exist_ok=True)
    made = {"args": args, "code": MODEL["code"], "code_commit": code_commit(),
            "patch_md5": md5(ROOT / "scripts/morphology4metrology.patch"),
            "checkpoint_sha256": sha256(checkpoint), "repo_commit": commit()}
    if made["checkpoint_sha256"] != MODEL["checkpoint"]["sha256"]:
        sys.exit(f"{checkpoint} sha256 {made['checkpoint_sha256']} is not config/model.yaml's")
    print(json.dumps(made, indent=1), flush=True)
    done = subprocess.run([sys.executable, "-u", "reconstruction.py", *args], cwd=CODE)
    if done.returncode:
        sys.exit(done.returncode)
    expected = ["model.pth", "reconstructor.pth" if step == 0 else "reconstructor_unfrozen.pth", f"metrics_step_{step}.txt"]
    missing = [f for f in expected if not (out / f).exists()]
    if missing:
        sys.exit(f"step {step} did not write {missing} in {out}: it did not run as step {step}")
    json.dump(made, open(out / "made.json", "w"), indent=1)


if __name__ == "__main__":
    if len(sys.argv) >= 3 and sys.argv[1] == "predict":
        predict(sys.argv[2], sys.argv[3:])
    elif len(sys.argv) >= 4 and sys.argv[1] == "finetune":
        finetune(sys.argv[2], sys.argv[3], sys.argv[4:])
    elif len(sys.argv) >= 4 and sys.argv[1] == "train" and sys.argv[3] in ("0", "1"):
        train(sys.argv[2], int(sys.argv[3]), sys.argv[4:])
    else:
        sys.exit(__doc__)
