#!/usr/bin/env python
"""Pre-build the CARRADA angle-Doppler map cache used by RadarCrossAttention.

Each AD map is an inverse FFT over the range axis of a 16 MB RAD cube -- measured 254 ms
per frame against 10 ms to load the cube -- and was recomputed identically every epoch,
for train and val alike (~5,100 frames/epoch, ~29 worker-hours over 80 epochs). This does
that pass once, in parallel, over every frame the dataset will read in every sequence;
afterwards CARRADA._get_ad_cached serves the 64 KB map straight from disk.

    python scripts/build_carrada_ad_cache.py [--workers 8]

Idempotent: existing entries are skipped, so it doubles as a repair pass. The cache
directory is content-addressed on the producing code (see CARRADA.__init__),
so this must be re-run after editing get_AD_map/get_RAD -- which is the point.
"""
import argparse
import os
import sys
import time
from multiprocessing import Pool

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)
sys.path.insert(0, os.path.join(REPO_ROOT, "scripts"))

os.environ.setdefault("MODEL_ROOTDIR", "/home/kangle/dataset/trained_models")
os.environ.setdefault("DATASET_ROOTDIR", "/home/kangle/dataset")
os.environ.setdefault("TMP_ROOTDIR", "/home/kangle/dataset/tmp")
os.environ.setdefault("LOGURU_LEVEL", "ERROR")
os.environ.setdefault("DATABASE_URL", "postgresql://postgres:postgres@localhost/postgres")

FEATURES = ["RD", "RA", "AD"]
_instances = {}


def _source(f, model_cfg, train_cfg):
    """One RadarCrossAttention sample adapter per (worker, file): it knows which frames the
    windows read; its parser owns the AD cache."""
    if f["id"] not in _instances:
        from detection_vis_backend.datasets.dataset import DatasetFactory
        from detection_vis_backend.datasets.adapters import make_sample_adapter
        factory = DatasetFactory()
        factory.get_shared_instance(f["parse"], f["id"],
                                    parse_args=(f["id"], f["path"], f["name"], f["config"]))
        parser = factory.get_instance(f["parse"], f["id"])
        _instances[f["id"]] = make_sample_adapter("RadarCrossAttention", parser, FEATURES,
                                                  model_cfg, train_cfg)
    return _instances[f["id"]]


def _frames(source):
    """Every frame any window will touch -- for win_size 1 that is exactly the annotated
    frames; larger windows add their forward offsets -- and the cache dir they land in."""
    frames = sorted({idx + i for idx in source.indexmapping for i in range(source.win_frames)})
    return frames, source.parser.ad_cache_dir


def _one(job):
    f, model_cfg, train_cfg, frame = job
    try:
        _source(f, model_cfg, train_cfg).parser._get_ad_cached(frame)
        return True
    except Exception as e:                       # keep going; report at the end
        return f"{f['name']}/{frame}: {type(e).__name__}: {e}"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--workers", type=int, default=8)
    args = p.parse_args()

    from smoke_sweep import resolve_configs, fetch_datafiles
    from detection_vis_backend.jobs.tasks import _parse_datafiles

    model_cfg, train_cfg = resolve_configs("RadarCrossAttention", "CARRADA", FEATURES, "_reference")
    datafiles = fetch_datafiles()["CARRADA"]
    _parse_datafiles(datafiles)

    jobs = []
    cache_dir = None
    for f in datafiles:
        frames, cache_dir = _frames(_source(f, model_cfg, train_cfg))
        jobs.extend((f, model_cfg, train_cfg, fr) for fr in frames)
    n = len(jobs)
    print(f"{n} frames over {len(datafiles)} sequences -> {cache_dir}  ({args.workers} workers)", flush=True)

    t0 = time.time()
    errors = []
    done = 0
    with Pool(args.workers) as pool:
        for r in pool.imap_unordered(_one, jobs, chunksize=16):
            done += 1
            if r is not True:
                errors.append(r)
            if done % 500 == 0 or done == n:
                el = time.time() - t0
                eta = el / done * (n - done)
                print(f"  {done}/{n}  elapsed {el/60:.1f}m  eta {eta/60:.1f}m  errors {len(errors)}", flush=True)

    cached = size = 0
    for root, _, files in os.walk(cache_dir):
        for x in files:
            if x.endswith(".npy"):
                cached += 1
                size += os.path.getsize(os.path.join(root, x))
    print(f"\ndone in {(time.time()-t0)/60:.1f}m: {cached}/{n} cached, {size/2**20:.0f} MiB")
    if errors:
        print(f"{len(errors)} errors, first 5:")
        for e in errors[:5]:
            print("  ", e)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
