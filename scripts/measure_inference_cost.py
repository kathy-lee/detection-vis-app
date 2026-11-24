#!/usr/bin/env python
"""Measure what a trained model costs at inference: FLOPs, latency per pipeline stage and
peak GPU memory, for the cost table of docs/training-validation-baselines.md.

    python scripts/measure_inference_cost.py <run> [<run> ...]
    python scripts/measure_inference_cost.py <run> --statistic p95 --from-saved

<run> is a folder under MODEL_ROOTDIR (or a path to one) holding train_info.txt and the
run's checkpoints. Nothing is trained: the network is rebuilt from the run's own
model_config, its checkpoint is loaded, and frames of the run's own test split go through
it one at a time. Every run gets a JSON record under --out with the per-frame times, so a
different statistic can be read from the same measurement later (--from-saved).

Requires Postgres up (datafile registry) and the dataset volumes mounted, like training.
"""
# --------------------------------------------------------------------------------------
# Why each quantity is measured the way it is
#
# WHAT IS MEASURED
#   The papers behind these models report compute (GFLOPs or GMACs) and inference time far
#   more often than anything else beyond accuracy and parameter count; none reports a
#   measured peak memory, which is added because it decides what fits on a smaller device
#   and parameter count hides it (activations, not weights, dominate several models).
#   All three depend on the architecture and the input, hardly on the weights, so the
#   existing checkpoints are reused. They are still loaded, and real frames are used,
#   because post-processing cost depends on how many candidates a trained model produces.
#
# FLOPS
#   - Counted with torch.utils.flop_counter.FlopCounterMode. It ships with PyTorch, and it
#     counts at the operator level as the model really runs, so functional calls
#     (torch.matmul, einsum, attention arithmetic) are counted. thop hooks nn.Module types
#     and misses those, undercounting the transformers; fvcore traces the model, which the
#     dict outputs and data-dependent control flow of several networks here make fragile.
#   - Convention: one fused multiply-add is one FLOP, as fvcore counts, so the number
#     equals the MAC count. FlopCounterMode itself counts the multiply and the add
#     separately (2 per multiply-add in every forward formula), so its total is halved.
#   - Batch size 1, the model's native input size, one real frame.
#   - Only convolutions and matrix products are counted. Elementwise activations,
#     normalisation, pooling and tensor reshapes are not, as with fvcore's defaults.
#   - Mamba's selective scan is a custom CUDA kernel that no profiler sees; its FLOPs are
#     added from a formula (inference_cost.mamba_scan_flops) and reported separately in
#     the record. Mamba2's scan has no such formula here and is listed as uncounted.
#   - Published figures are only roughly comparable: each paper used its own tool and
#     convention (T-RODNet thop; RECORD and SSMRadNet report GMACs; others do not say).
#
# LATENCY
#   - Three stages, timed separately, because they depend on different things: the loader
#     and the disk, the network and the GPU, the CPU and the number of candidates.
#       preprocessing    fetch one sample, batch it, move the input to the GPU
#       inference        the forward pass
#       post-processing  output copy to the CPU, decoding, non-maximum suppression
#   - Every stage is timed by wall clock (time.perf_counter) with torch.cuda.synchronize()
#     at its boundaries. CUDA events were the alternative for the GPU stages; they cannot
#     time CPU work, post-processing is not purely CPU for every model, and the wall clock
#     needs the synchronize anyway or the wait for an unfinished forward pass is counted
#     as post-processing. One clock also makes the stages add up to the time per frame.
#   - Batch size 1, FP32, gradients off, eval mode: single-frame processing, at the
#     precision every run here is evaluated in.
#   - Frames come from the run's own test split, in order (its val split where it has no
#     test split), so post-processing sees realistic candidate counts.
#   - The first 50 frames are warm-up and are not part of any statistic.
#   - One statistic is reported, chosen with --statistic (mean, median or p95). Median is
#     the default: a stray slowdown from other load on the machine moves the mean and the
#     95th percentile but hardly the median. The per-frame times are saved, so the others
#     can be read later without measuring again.
#   - Preprocessing is fetched in this process, without DataLoader workers: a worker
#     process would hide the fetch from the clock. It includes reading the frame from disk
#     and whatever the loader computes per frame; some loaders read a disk cache built on
#     an earlier pass, so the number is the cost with warm caches.
#   - Post-processing repeats what the model's evaluation function does to a frame before
#     scoring it, with the same helpers and thresholds, and stops where scoring starts.
#   - DAROD decodes its boxes and runs their NMS inside forward(). Hooks time that module
#     (net.decoder) and its time is moved to post-processing, so the columns mean the same
#     for every model. Its proposal NMS stays in inference: the network needs its result.
#   - RadarCrossAttention's post-processing runs on the GPU, driven by a Python loop over
#     the peaks; only the prediction half of the evaluation's peak association is timed,
#     not the ground-truth peaks or the matching against them.
#   - For the models that take a window of frames (RODNet's family, T-RODNet, RAMP-CNN) a
#     sample is one window, and its post-processing decodes the frames the window advances
#     by. The segmentation models (CARRADA) only take an argmax.
#
# PEAK MEMORY
#   - torch.cuda.reset_peak_memory_stats() before the forward pass and
#     torch.cuda.max_memory_allocated() after it. Without the reset the peak would cover
#     everything since the process started.
#   - The forward pass only, to match the inference column; for DAROD that includes its
#     decoder, which cannot be separated from the pass.
#   - The reset starts from what is already allocated, so the figure is the weights, the
#     input and the activations alive at the busiest moment. PyTorch's cache of freed
#     blocks and the CUDA context (which nvidia-smi adds) are not in it.
#   - Reported as the largest peak over the timed frames.
#
# ONE PROCESS PER MODEL
#   Each run is measured in a fresh Python process, so nothing a previous model left on
#   the GPU or in the allocator reaches the next one's memory or timing.
# --------------------------------------------------------------------------------------
import argparse
import json
import os
import subprocess
import sys
import tempfile
from datetime import datetime

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)
from detection_vis_backend.paths import load_env  # noqa: E402

load_env()
os.environ.setdefault("LOGURU_LEVEL", "WARNING")
os.environ.setdefault("DATABASE_URL", "postgresql://postgres:postgres@localhost/postgres")

WARMUP_FRAMES = 50


def run_dir_of(run):
    return run if os.path.isdir(run) else os.path.join(os.environ["MODEL_ROOTDIR"], run)


def record_path(run, out_dir):
    return os.path.join(out_dir, os.path.basename(os.path.normpath(run_dir_of(run))) + ".json")


def measure(run, args):
    """Measure one run in this process and write its record."""
    import torch
    from detection_vis_backend.jobs.tasks import _parse_datafiles
    from detection_vis_backend.networks.network import NetworkFactory
    from detection_vis_backend.train import inference_cost as ic
    from detection_vis_backend.train.inference import relocate_datafile
    from detection_vis_backend.train.train import CreateDataLoaders

    run_dir = run_dir_of(run)
    with open(os.path.join(run_dir, "train_info.txt")) as f:
        parameters = json.load(f)
    # train_info.txt freezes the dataset paths of the day the model was trained.
    datafiles = [relocate_datafile(d, d["id"]) for d in parameters["datafiles"]]
    features, train_config = parameters["features"], parameters["train_config"]
    model_config = dict(parameters["model_config"])
    model_type, dataset_type = model_config["class"], datafiles[0]["parse"]

    _parse_datafiles(datafiles)
    with tempfile.TemporaryDirectory() as split_files:
        _, val_loader, test_loader = CreateDataLoaders(datafiles, features, model_config, train_config, split_files)
    split = "val" if (test_loader is None or args.split == "val") else "test"
    loader = ic.single_frame_loader(test_loader if split == "test" else val_loader)

    device = torch.device(args.device)
    net = NetworkFactory().get_instance(model_type, {k: v for k, v in model_config.items() if k != "class"})
    checkpoint = ic.find_checkpoint(run_dir, args.epoch)
    # Checkpoints also carry the loss criterion's buffers, which a network built without
    # init_lossfunc does not have; a shape mismatch on a real weight still raises.
    incompatible = net.load_state_dict(torch.load(checkpoint, map_location=device)["net_state_dict"], strict=False)
    if incompatible.missing_keys:
        raise RuntimeError(f"{checkpoint} lacks network weights: {incompatible.missing_keys}")
    net.to(device).float().eval()

    pipeline = ic.make_pipeline(model_type, dataset_type, features, train_config, split, net)
    flops = ic.count_flops(net, pipeline.inputs(next(iter(loader)), device), pipeline.forward)
    pipeline.take_postprocess_seconds_in_forward()   # the counting pass is not a timed frame
    raw = ic.measure_run(net, loader, pipeline, device, max_frames=WARMUP_FRAMES + args.max_frames)

    record = {
        "run": os.path.basename(os.path.normpath(run_dir)),
        "model": model_type, "dataset": dataset_type, "features": features,
        "checkpoint": os.path.basename(checkpoint), "split": split,
        "batch_size": 1, "precision": "FP32", "warmup": WARMUP_FRAMES,
        "device": torch.cuda.get_device_name(device) if device.type == "cuda" else "cpu",
        "torch": torch.__version__, "measured_at": datetime.now().isoformat(timespec="seconds"),
        "parameters": sum(p.numel() for p in net.parameters()),
        "flops": flops, "raw": raw,
    }
    os.makedirs(args.out, exist_ok=True)
    with open(record_path(run, args.out), "w") as f:
        json.dump(record, f)


def report(runs, args):
    from detection_vis_backend.train import inference_cost as ic
    header = ("run", "split", "frames", "GFLOPs", "preprocess ms", "inference ms", "post-process ms", "peak MiB")
    rows = []
    for run in runs:
        path = record_path(run, args.out)
        if not os.path.exists(path):
            rows.append((os.path.basename(os.path.normpath(run)), "no record") + ("",) * 6)
            continue
        with open(path) as f:
            record = json.load(f)
        row = ic.summarize_run(record, args.statistic)
        peak = "" if row["peak_memory_mib"] is None else f"{row['peak_memory_mib']:.0f}"
        rows.append((record["run"], record["split"], str(row["frames"]), f"{row['gflops']:.2f}",
                     f"{row['preprocess_ms']:.2f}", f"{row['inference_ms']:.2f}",
                     f"{row['postprocess_ms']:.2f}", peak))
    widths = [max(len(line[i]) for line in [header] + rows) for i in range(len(header))]
    print(f"\n{args.statistic} per frame, batch size 1, FP32; the first {WARMUP_FRAMES} frames are warm-up")
    for line in [header] + rows:
        print("  ".join(cell.ljust(width) for cell, width in zip(line, widths)))


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("runs", nargs="+", help="run folders under MODEL_ROOTDIR, or paths to them")
    parser.add_argument("--statistic", choices=("mean", "median", "p95"), default="median",
                        help="which statistic of the per-frame times the table reports")
    parser.add_argument("--max-frames", type=int, default=1000, help="timed frames per run, after the warm-up")
    parser.add_argument("--epoch", type=int, default=None, help="checkpoint epoch (default: the last one)")
    parser.add_argument("--split", choices=("test", "val"), default="test",
                        help="split the frames come from; val is used anyway where a run has no test split")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--out", default=os.path.join(os.environ["TMP_ROOTDIR"], "inference_cost"),
                        help="folder of the per-run records")
    parser.add_argument("--from-saved", action="store_true", help="report from the saved records without measuring")
    parser.add_argument("--in-this-process", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()

    if args.in_this_process:
        measure(args.runs[0], args)
        return
    if not args.from_saved:
        for run in args.runs:
            command = [sys.executable, os.path.abspath(__file__), run, "--in-this-process",
                       "--max-frames", str(args.max_frames), "--split", args.split,
                       "--device", args.device, "--out", args.out]
            if args.epoch is not None:
                command += ["--epoch", str(args.epoch)]
            # A failed measurement must not be reported from an earlier one's record.
            if os.path.exists(record_path(run, args.out)):
                os.remove(record_path(run, args.out))
            print(f"measuring {run} ...", flush=True)
            if subprocess.run(command).returncode != 0:
                print(f"  {run} failed; see the error above", flush=True)
    report(args.runs, args)


if __name__ == "__main__":
    main()
