"""Phase 1 of the training-validation campaign: smoke-sweep every wired model/dataset pair.

Purpose is the plumbing, not the score. Per run we record: does it complete, does the
loss stay finite (and fall, when >1 epoch), are the metrics non-degenerate, and did the
evaluator produce anything at all. This is where hardcoded paths and the "mAR/mIoU pinned
at 0.0000" class of defect surface -- cheaply, before any multi-day run.

Same code path as the train page, minus the job queue: resolve datafiles from Postgres,
resolve the shipped model/train configs exactly as the UI's find_default_configs does,
parse the datafiles, then call the backend's train().

Usage:
    .venv/bin/python scripts/smoke_sweep.py                     # all runs, smoke split
    .venv/bin/python scripts/smoke_sweep.py --only RECORD       # substring filter on the run id
    .venv/bin/python scripts/smoke_sweep.py --split original    # official splits, full data
    .venv/bin/python scripts/smoke_sweep.py --build-only        # build dataloaders, don't train
    .venv/bin/python scripts/smoke_sweep.py --epochs 2          # override num_epochs

Requires Postgres up (datafile registry) and the dataset volumes mounted.
"""
import argparse
import copy
import csv
import json
import math
import os
import sys
import time
import traceback

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)
# Mirror detection_vis_backend/run.sh:13-16 -- the dataset parsers read these directly and
# fail deep in Path() with an unhelpful TypeError if any is unset (TMP_ROOTDIR is the
# feature cache used by the CRUW/RADIal parsers).
os.environ.setdefault("MODEL_ROOTDIR", "/home/kangle/dataset/trained_models")
os.environ.setdefault("DATASET_ROOTDIR", "/home/kangle/dataset")
os.environ.setdefault("TMP_ROOTDIR", "/home/kangle/dataset/tmp")
os.environ.setdefault("LOGURU_LEVEL", "INFO")
os.environ.setdefault("DATABASE_URL", "postgresql://postgres:postgres@localhost/postgres")

import psycopg2
import psycopg2.extras

NETWORKS_DIR = os.path.join(REPO_ROOT, "detection_vis_backend", "networks")
TRAIN_DIR = os.path.join(REPO_ROOT, "detection_vis_backend", "train")

# --------------------------------------------------------------------------------------
# The run matrix.
#
# 16 wired (model, dataset) pairs, but 22 runs: the RECORD family on CARRADA is
# single-view, so RA and RD are separate models with separate configs and separate
# published numbers (RECORD_model_config.json keys them under CARRADA/{RA,RD}).
# --------------------------------------------------------------------------------------
RODNET_VARIANTS = ["RODNet_CDC", "RODNet_CDCv2", "RODNet_HG",
                   "RODNet_HGv2", "RODNet_HGwI", "RODNet_HGwIv2"]

RUNS = (
    [{"model": "FFTRadNet", "dataset": "RADIal", "features": ["RD"]}]
    + [{"model": m, "dataset": "CRUW", "features": ["RA"]} for m in RODNET_VARIANTS]
    + [{"model": m, "dataset": "CRUW", "features": ["RA"]}
       for m in ("RECORD", "RECORDNoLstm", "RECORDNoLstmMulti")]
    + [{"model": m, "dataset": "CARRADA", "features": [v]}
       for m in ("RECORD", "RECORDNoLstm", "RECORDNoLstmMulti") for v in ("RA", "RD")]
    + [
        {"model": "MVRECORD", "dataset": "CARRADA", "features": ["RD", "RA", "AD"]},
        {"model": "RADDet", "dataset": "RADDetDataset", "features": ["RAD"]},
        {"model": "DAROD", "dataset": "CARRADA", "features": ["RD"]},
        {"model": "DAROD", "dataset": "RADDetDataset", "features": ["RD"]},
        {"model": "RAMP_CNN", "dataset": "UWCR", "features": ["RD", "RA", "AD"]},
        {"model": "RadarCrossAttention", "dataset": "CARRADA", "features": ["RD", "RA", "AD"]},
        {"model": "PointPillar", "dataset": "ViewOfDelft", "features": ["radarPC"]},
        {"model": "RadarPillar", "dataset": "ViewOfDelft", "features": ["radarPC"]},
    ]
)

# Datasets whose split is built into train.py (splitmode "original"). CRUW and UWCR have
# none, so they always go through an explicit sequence split.
HAS_ORIGINAL_SPLIT = {"RADIal", "CARRADA", "RADDetDataset", "ViewOfDelft"}

# CARRADA's official train/val sequences, copied from train.py's original-split branch so
# the smoke subset is drawn from the right side of the split (no train/val leakage).
CARRADA_TRAIN = ["2019-09-16-12-52-12", "2019-09-16-12-55-51", "2019-09-16-12-58-42"]
CARRADA_VAL = ["2019-09-16-13-06-41", "2019-09-16-13-23-22"]

RECORD_FAMILY = ("RECORD", "RECORDNoLstm", "RECORDNoLstmMulti", "MVRECORD")
RODNET_FAMILY = tuple(RODNET_VARIANTS) + ("RadarFormer_hrformer2d",)


def run_id(run):
    """Stable identifier: model/dataset or model/dataset/feature for the single-view pairs."""
    base = f"{run['model']}/{run['dataset']}"
    if run["dataset"] == "CARRADA" and run["model"] in ("RECORD", "RECORDNoLstm",
                                                        "RECORDNoLstmMulti"):
        return f"{base}/{run['features'][0]}"
    return base


# --------------------------------------------------------------------------------------
# Config resolution -- mirrors detection_vis_ui/pages/3_Train_model.py:find_default_configs.
# Reimplemented rather than imported: that module is a Streamlit page and executes UI code
# at import time. Keep the two in step.
# --------------------------------------------------------------------------------------
def resolve_configs(model, dataset, features):
    if model in RECORD_FAMILY:
        with open(os.path.join(NETWORKS_DIR, "RECORD_model_config.json")) as f:
            data = json.load(f)
        if model == "MVRECORD":
            model_cfg = data[model]
        else:
            if len(features) > 1:
                raise ValueError(f"{model} takes a single input feature, got {features}")
            node = data[model].get(dataset, data[model])
            model_cfg = node.get(features[0], node) if isinstance(node, dict) else node
    elif model == "DAROD":
        with open(os.path.join(NETWORKS_DIR, "DAROD_model_config.json")) as f:
            config = json.load(f)
        if dataset not in config:
            raise ValueError(f"DAROD model config has no entry for {dataset}")
        model_cfg = config[dataset]
    elif model in RODNET_FAMILY:
        with open(os.path.join(NETWORKS_DIR, "RODNet_model_config.json")) as f:
            model_cfg = json.load(f)[model]
    else:
        with open(os.path.join(NETWORKS_DIR, f"{model}_model_config.json")) as f:
            model_cfg = json.load(f)

    if model in RODNET_FAMILY:
        with open(os.path.join(TRAIN_DIR, "RODNet_train_config.json")) as f:
            train_cfg = json.load(f)
    else:
        with open(os.path.join(TRAIN_DIR, f"{model}_train_config.json")) as f:
            data = json.load(f)
        if "default" in data:
            train_cfg = copy.deepcopy(data["default"])
            if dataset in data:
                node = data[dataset]
                train_cfg.update(node.get(features[0], node) if features[0] in node else node)
        else:
            train_cfg = data

    model_cfg = copy.deepcopy(model_cfg)
    model_cfg["class"] = model
    return model_cfg, copy.deepcopy(train_cfg)


# --------------------------------------------------------------------------------------
# Datafiles
# --------------------------------------------------------------------------------------
def fetch_datafiles():
    """Every registered datafile as a plain dict, grouped by parser."""
    conn = psycopg2.connect(host="127.0.0.1", port=5432, user="postgres",
                            password="postgres", dbname="postgres")
    try:
        cur = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
        cur.execute("SELECT * FROM files ORDER BY id;")
        rows = [dict(r) for r in cur.fetchall()]
    finally:
        conn.close()
    by_parse = {}
    for r in rows:
        by_parse.setdefault(r["parse"], []).append(r)
    return by_parse


def build_split(dataset, files, mode):
    """Return (datafiles, split_patch) for this dataset.

    smoke    -- a few files / a small random fraction. Fast, exercises the whole stack.
    original -- the official split where train.py provides one; the protocol split
                otherwise (CRUW 36/4 per the RECORD paper; UWCR 18/4).
    """
    by_name = {f["name"]: f for f in files}

    if dataset == "CARRADA":
        if mode == "original":
            return files, {"splitmode": "original"}
        train = [by_name[n] for n in CARRADA_TRAIN if n in by_name][:2]
        val = [by_name[n] for n in CARRADA_VAL if n in by_name][:1]
        chosen = train + val
        return chosen, {"splitmode": "sequence",
                        "split_sequence": {"train": train, "val": val, "test": []}}

    if dataset in HAS_ORIGINAL_SPLIT:
        # RADIal / RADDetDataset / ViewOfDelft are single-file registrations, so a
        # sequence split is impossible (it asserts >1 file) -- and a *random* split is a
        # trap: random_split hands the remainder to test, train() then evaluates that
        # test loader (train.py:529, skipped only when datasets['test'] is None, which
        # random mode never produces). So [0.02, 0.02, 0.96] means training on 165 frames
        # and then evaluating on 7,922. There is no ratio that keeps all three small.
        # Use the official split for these instead: it is bounded, it is the split we
        # want in Phase 2 anyway, and ViewOfDelft's leaves test empty (test = None).
        return files[:1], {"splitmode": "original"}

    # CRUW and UWCR: no built-in split, always explicit sequence.
    if mode == "original":
        n_val = 4
        train, val = files[:-n_val], files[-n_val:]
    else:
        train, val = files[:2], files[2:3]
    chosen = train + val
    return chosen, {"splitmode": "sequence",
                    "split_sequence": {"train": train, "val": val, "test": []}}


# --------------------------------------------------------------------------------------
# Result inspection
# --------------------------------------------------------------------------------------
def read_val_eval(exp_name):
    path = os.path.join(os.environ["MODEL_ROOTDIR"], exp_name, "val_eval.csv")
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return [dict(r) for r in csv.DictReader(f)]


def _floats(rows, col):
    out = []
    for r in rows:
        try:
            out.append(float(r[col]))
        except (KeyError, TypeError, ValueError):
            pass
    return out


# Which of val_eval.csv's metric columns each evaluator actually computes. Every evaluator
# returns all three keys, but most hardcode the inapplicable ones to 0 -- e.g.
# RODNet_evaluation returns 'mIoU': 0 (evaluate.py:800) because a detection model scored by
# OLS has no IoU, and RadarPillar_evaluation returns 'mAR': 0.0, 'mIoU': 0.0
# (evaluate.py:1930). A structurally-absent metric is indistinguishable from a collapsed one
# in the CSV, so the zero-check must be told which is which or it reports false degeneracy.
LIVE_METRICS = {
    ("FFTRadNet", "RADIal"): ("mAP", "mAR", "mIoU"),
    ("RECORD", "CARRADA"): ("mAP", "mAR", "mIoU"),
    ("RECORDNoLstm", "CARRADA"): ("mAP", "mAR", "mIoU"),
    ("RECORDNoLstmMulti", "CARRADA"): ("mAP", "mAR", "mIoU"),
    ("MVRECORD", "CARRADA"): ("mAP", "mAR", "mIoU"),
    ("RADDet", "RADDetDataset"): ("mAP",),
    ("DAROD", "CARRADA"): ("mAP", "mAR"),
    ("DAROD", "RADDetDataset"): ("mAP", "mAR"),
    ("RAMP_CNN", "UWCR"): ("mAP", "mAR"),
    ("RadarCrossAttention", "CARRADA"): ("mAP",),
    ("PointPillar", "ViewOfDelft"): ("mAP",),
    ("RadarPillar", "ViewOfDelft"): ("mAP",),
}
for _m in RODNET_VARIANTS:
    LIVE_METRICS[_m, "CRUW"] = ("mAP", "mAR")
for _m in ("RECORD", "RECORDNoLstm", "RECORDNoLstmMulti"):
    LIVE_METRICS[_m, "CRUW"] = ("mAP", "mAR")


def check_health(rows, model=None, dataset=None):
    """Degeneracy checks read straight off val_eval.csv -- no new instrumentation."""
    if not rows:
        return {"verdict": "no-eval", "notes": ["val_eval.csv missing or empty"]}

    notes, verdict = [], "ok"
    losses = _floats(rows, "loss")
    if not losses:
        notes.append("no loss column")
        verdict = "degenerate"
    else:
        if not all(math.isfinite(v) for v in losses):
            notes.append("loss went non-finite (nan/inf)")
            verdict = "degenerate"
        if len(losses) > 1:
            notes.append(f"loss {losses[0]:.4g} -> {losses[-1]:.4g}"
                         f" ({'down' if losses[-1] < losses[0] else 'NOT down'})")
            if losses[-1] >= losses[0] and verdict == "ok":
                verdict = "suspect"
        else:
            notes.append(f"loss {losses[0]:.4g} (single epoch, no trend)")

    live = LIVE_METRICS.get((model, dataset))
    if live is None:
        notes.append(f"no metric map for ({model}, {dataset}) -- checking all columns")
        live = ("mAP", "mAR", "mIoU")

    pinned, present = [], []
    for col in live:
        vals = _floats(rows, col)
        if not vals:
            continue
        present.append(col)
        if all(v == 0.0 for v in vals):
            pinned.append(col)
    if not present:
        notes.append(f"evaluator produced none of its metrics ({', '.join(live)})")
        verdict = "degenerate"
    elif pinned:
        notes.append(f"pinned at exactly 0 across all epochs: {', '.join(pinned)}")
        verdict = "degenerate"
    else:
        vals = {c: _floats(rows, c)[-1] for c in present}
        notes.append("metrics non-zero: "
                     + ", ".join(f"{c}={v:.4g}" for c, v in vals.items()))
    return {"verdict": verdict, "notes": notes}


# --------------------------------------------------------------------------------------
# Driver
# --------------------------------------------------------------------------------------
def execute(run, by_parse, args):
    from detection_vis_backend.jobs.tasks import _parse_datafiles
    from detection_vis_backend.train.train import train, CreateDataLoaders

    rid = run_id(run)
    model, dataset, features = run["model"], run["dataset"], run["features"]
    result = {"id": rid, "model": model, "dataset": dataset, "features": features}

    files = by_parse.get(dataset, [])
    if not files:
        return {**result, "status": "error", "error": f"no datafiles registered for {dataset}"}

    model_cfg, train_cfg = resolve_configs(model, dataset, features)
    datafiles, split_patch = build_split(dataset, files, args.split)
    train_cfg.setdefault("dataloader", {}).update(split_patch)
    if args.epochs is not None:
        train_cfg["num_epochs"] = args.epochs
    if args.seed is not None:
        train_cfg["seed"] = args.seed
    result["num_epochs"] = train_cfg["num_epochs"]
    result["seed"] = train_cfg["seed"]
    result["splitmode"] = split_patch["splitmode"]
    result["n_datafiles"] = len(datafiles)

    t0 = time.time()
    try:
        _parse_datafiles(datafiles)
        if args.build_only:
            tr, va, te = CreateDataLoaders(datafiles, features, model_cfg, train_cfg,
                                           os.environ["MODEL_ROOTDIR"])
            result.update(status="built",
                          n_train=len(tr.dataset), n_val=len(va.dataset),
                          n_test=len(te.dataset) if te else 0)
        else:
            exp_name = train(datafiles, features, model_cfg, train_cfg)
            rows = read_val_eval(exp_name)
            result.update(status="ok", exp_name=exp_name,
                          epochs_recorded=len(rows) if rows else 0,
                          **check_health(rows, model, dataset))
    except Exception as e:
        result.update(status="error", error=f"{type(e).__name__}: {e}",
                      traceback=traceback.format_exc()[-4000:])
    result["seconds"] = round(time.time() - t0, 1)
    return result


def write_report(results, path, args):
    lines = [
        "# Phase 1 - smoke sweep",
        "",
        f"`--split {args.split}`"
        + (f", `--epochs {args.epochs}`" if args.epochs is not None else ", config-default epochs")
        + (f", `--seed {args.seed}`" if args.seed is not None else "")
        + (", **build-only** (dataloaders built, no training)" if args.build_only else ""),
        "",
        "Purpose is the plumbing, not the score.",
        "",
    ]
    skipped = [run_id(r) for r in RUNS if r not in [x for x in RUNS if run_id(x) in {q['id'] for q in results}]]
    if args.only or args.exclude:
        lines += [f"**Not covered by this sweep** ({len(skipped)}): "
                  + ", ".join(f"`{s}`" for s in skipped) + ".", ""]
    lines += [
        "| Run | Split | Files | Epochs | Status | Verdict | Time (s) | Notes |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for r in results:
        note = "; ".join(r.get("notes", [])) or r.get("error", "")
        if r.get("status") == "built":
            note = f"train={r.get('n_train')} val={r.get('n_val')} test={r.get('n_test')}"
        lines.append(
            f"| `{r['id']}` | {r.get('splitmode', '-')} | {r.get('n_datafiles', '-')} "
            f"| {r.get('num_epochs', '-')} | {r['status']} | {r.get('verdict', '-')} "
            f"| {r.get('seconds', '-')} | {note.replace('|', '/')} |"
        )
    counts = {}
    for r in results:
        key = r.get("verdict") if r["status"] not in ("error",) else "error"
        counts[key] = counts.get(key, 0) + 1
    lines += ["", "## Totals", ""]
    lines += [f"- **{k}**: {v}" for k, v in sorted(counts.items())]

    failing = [r for r in results if r["status"] == "error"]
    if failing:
        lines += ["", "## Failures", ""]
        for r in failing:
            lines += [f"### `{r['id']}`", "", "```", r.get("traceback", r.get("error", "")).strip(), "```", ""]

    with open(path, "w") as f:
        f.write("\n".join(lines) + "\n")


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--only", help="substring filter on the run id, e.g. RECORD or CARRADA")
    p.add_argument("--exclude", action="append", default=[], metavar="SUBSTR",
                   help="skip runs whose id contains SUBSTR; repeatable")
    p.add_argument("--split", choices=["smoke", "original"], default="smoke")
    p.add_argument("--epochs", type=int, help="override num_epochs (default: config value)")
    # Repeat a pair across seeds to measure run-to-run spread. With --split original the
    # split is fixed by sequence/frame lists, so the seed only moves weight init, batch
    # order and augmentation draws -- which is exactly the variation we want to quantify.
    p.add_argument("--seed", type=int, help="override train_config seed (default: config value)")
    p.add_argument("--build-only", action="store_true",
                   help="build dataloaders and stop -- validates split resolution cheaply")
    p.add_argument("--out", default=os.path.join(REPO_ROOT, "docs", "phase1-smoke-sweep.md"))
    p.add_argument("--list", action="store_true", help="print the run matrix and exit")
    p.add_argument("--recompute", action="store_true",
                   help="re-score an existing results .json from each run's saved "
                        "val_eval.csv and rewrite the report; runs nothing")
    args = p.parse_args()

    if args.recompute:
        json_path = os.path.splitext(args.out)[0] + ".json"
        with open(json_path) as f:
            results = json.load(f)
        for r in results:
            if r.get("status") != "ok" or not r.get("exp_name"):
                continue
            rows = read_val_eval(r["exp_name"])
            before = r.get("verdict")
            r.update(epochs_recorded=len(rows) if rows else 0,
                     **check_health(rows, r["model"], r["dataset"]))
            if before != r["verdict"]:
                print(f"{r['id']:<30} {before} -> {r['verdict']}")
        write_report(results, args.out, args)
        with open(json_path, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\nRescored {len(results)} runs -> {args.out}")
        return 0

    runs = [r for r in RUNS if not args.only or args.only.lower() in run_id(r).lower()]
    runs = [r for r in runs
            if not any(x.lower() in run_id(r).lower() for x in args.exclude)]
    if args.list:
        for r in runs:
            print(f"{run_id(r):<44} features={r['features']}")
        print(f"\n{len(runs)} runs")
        return 0
    if not runs:
        print(f"No runs match --only {args.only!r}", file=sys.stderr)
        return 2

    # Phase 1 (--split smoke) is about plumbing, so one epoch is the point. Phase 2
    # (--split original) is about the score, so honour each config's published schedule
    # instead of silently capping it at 1 -- pass --epochs to override either way.
    if args.epochs is None and not args.build_only and args.split == "smoke":
        args.epochs = 1

    by_parse = fetch_datafiles()
    results, json_path = [], os.path.splitext(args.out)[0] + ".json"
    os.makedirs(os.path.dirname(args.out), exist_ok=True)

    for i, run in enumerate(runs, 1):
        rid = run_id(run)
        print(f"\n{'=' * 78}\n[{i}/{len(runs)}] {rid}\n{'=' * 78}", flush=True)
        res = execute(run, by_parse, args)
        results.append(res)
        print(f"-> {res['status']} {res.get('verdict', '')} in {res['seconds']}s", flush=True)
        if res["status"] == "error":
            print(res.get("traceback", ""), file=sys.stderr, flush=True)
        # Rewrite after every run: a crash or Ctrl-C mid-sweep still leaves a usable report.
        write_report(results, args.out, args)
        with open(json_path, "w") as f:
            json.dump(results, f, indent=2)

    print(f"\nReport: {args.out}\nRaw:    {json_path}")
    return 0 if all(r["status"] != "error" for r in results) else 1


if __name__ == "__main__":
    sys.exit(main())
