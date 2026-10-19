#!/usr/bin/env python
"""Score the TransRAD authors' shipped checkpoint on the official RADDet test split
through OUR adapter, collate, network, decoder, NMS and evaluator, and compare with the
numbers their own Test.py produced with the same weights (Test_results/AP_Results.txt).

    python scripts/transrad_reference_check.py [--tol 0.01] [--batch 12]

Exit 0 when mAP@0.3 and mAP@0.5 are within --tol of the authors' 0.6016 / 0.3777.
Their run scored 2028 of the 2032 frames (batch 12, drop_last); this scores all 2032.
"""
import argparse
import os
import sys

import torch

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)
sys.path.insert(0, os.path.join(REPO_ROOT, "scripts"))

os.environ.setdefault("MODEL_ROOTDIR", "/home/kangle/dataset/trained_models")
os.environ.setdefault("DATASET_ROOTDIR", "/home/kangle/dataset")
os.environ.setdefault("TMP_ROOTDIR", "/home/kangle/dataset/tmp")
os.environ.setdefault("LOGURU_LEVEL", "ERROR")
os.environ.setdefault("DATABASE_URL", "postgresql://postgres:postgres@localhost/postgres")

AUTHORS = {0.1: 0.730876, 0.3: 0.601569, 0.4: 0.495822, 0.5: 0.377664, 0.6: 0.249594, 0.7: 0.129277}
CHECKPOINT = "/home/kangle/dataset/transrad_reference/TransRAD/model_data/best_epoch_weights.pth"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--tol", type=float, default=0.01)
    p.add_argument("--batch", type=int, default=12)
    p.add_argument("--workers", type=int, default=4)
    args = p.parse_args()

    from smoke_sweep import resolve_configs, fetch_datafiles
    from detection_vis_backend.jobs.tasks import _parse_datafiles
    from detection_vis_backend.train.train import CreateDataLoaders
    from detection_vis_backend.train.evaluate import TransRAD_evaluation
    from detection_vis_backend.networks.network import NetworkFactory

    model_cfg, train_cfg = resolve_configs("TransRAD", "RADDetDataset", ["RAD"])
    train_cfg["dataloader"]["splitmode"] = "original"
    train_cfg["dataloader"]["test"] = {"batch_size": args.batch, "num_workers": args.workers}
    files = fetch_datafiles()["RADDetDataset"][:1]
    _parse_datafiles(files)
    _, _, test_loader = CreateDataLoaders(files, ["RAD"], model_cfg, train_cfg, os.environ["TMP_ROOTDIR"])
    print(f"test frames: {len(test_loader.dataset)}", flush=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    params = {k: v for k, v in model_cfg.items() if k != "class"}
    net = NetworkFactory().get_instance("TransRAD", params)
    net.load_state_dict(torch.load(CHECKPOINT, map_location="cpu"), strict=True)
    net.to(device).eval()
    net.init_lossfunc(train_cfg, device)

    res = TransRAD_evaluation(net, test_loader, train_cfg, ["RAD"], None, "test", device)
    ours = {0.3: res["mAP"]}
    ours.update({float(k.split("@")[1]): v for k, v in res.items() if k.startswith("mAP@")})
    print(f"\n{'thr':>5} {'authors':>9} {'ours':>9} {'diff':>8}")
    for thr in sorted(AUTHORS):
        if thr in ours:
            print(f"{thr:>5} {AUTHORS[thr]:>9.4f} {ours[thr]:>9.4f} {ours[thr]-AUTHORS[thr]:>+8.4f}")
    for name in ("person", "bicycle", "car", "motorcycle", "bus", "truck"):
        print(f"  AP@0.3 {name:<11} {res.get('AP@0.3_' + name, float('nan')):.4f}")
    bad = [thr for thr in (0.3, 0.5) if abs(ours[thr] - AUTHORS[thr]) > args.tol]
    print("gate:", "OK" if not bad else f"FAIL at {bad} (tol {args.tol})")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
