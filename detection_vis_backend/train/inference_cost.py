"""Inference cost of a trained model: FLOPs, latency per pipeline stage and peak GPU
memory. scripts/measure_inference_cost.py drives this over the runs of the baselines cost
table and records there why each quantity is measured the way it is.

A frame goes through three stages, each timed on its own:

  preprocessing    fetch one sample from the dataset, batch it, move the input to the device
  inference        the network's forward pass
  post-processing  everything between the raw output and the detections (or the class
                   mask): the copy to the CPU, decoding, non-maximum suppression

Each model's post-processing repeats what its evaluation function does to a frame before
scoring it, with the same helpers and thresholds, and stops where scoring starts.
"""
import json
import os
import re
import time

import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.utils.flop_counter import FlopCounterMode

from detection_vis_backend.datasets.utils import confmap2ra
from detection_vis_backend.networks.transrad import decode_predictions as transrad_decode, nms_class_then_location as transrad_nms
from detection_vis_backend.train import evaluate
from detection_vis_backend.train.evaluate import ConfmapStack, eval_func_dict
from detection_vis_backend.train.utils import (association, create_default, decode, distribute, nms, peaks_detect,
                                               post_process_single_frame, process_predictions_FFT, update_peaks,
                                               yoloheadToPredictions)

STATISTICS = {
    "mean": np.mean,
    "median": np.median,
    "p95": lambda values: np.percentile(values, 95),
}


def summarize(seconds, statistic, warmup):
    """One number in milliseconds for a stage's per-frame times, without the warm-up frames."""
    if statistic not in STATISTICS:
        raise ValueError(f"unknown statistic {statistic!r}: expected one of {', '.join(STATISTICS)}")
    timed = seconds[warmup:]
    if len(timed) == 0:
        raise ValueError(f"{len(seconds)} frames measured, none left after the {warmup} warm-up frames")
    return float(STATISTICS[statistic](np.asarray(timed, dtype=np.float64) * 1e3))


def summarize_run(record, statistic):
    """The cost-table row of one measured run: GFLOPs, the chosen statistic of each stage in
    milliseconds, and the largest forward-pass peak over the timed frames in MiB."""
    raw, warmup = record["raw"], record["warmup"]
    peaks = raw["peak_memory_bytes"][warmup:]
    return {
        "gflops": record["flops"]["total"] / 1e9,
        "preprocess_ms": summarize(raw["preprocess_s"], statistic, warmup),
        "inference_ms": summarize(raw["inference_s"], statistic, warmup),
        "postprocess_ms": summarize(raw["postprocess_s"], statistic, warmup),
        "peak_memory_mib": max(peaks) / 2**20 if peaks else None,
        "frames": len(raw["inference_s"]) - warmup,
    }


def find_checkpoint(run_dir, epoch=None):
    """Path of a run's checkpoint: the given epoch's, or the last epoch's."""
    by_epoch = {}
    for name in os.listdir(run_dir):
        match = re.search(r"_epoch(\d+)_.*\.pth$", name)
        if match:
            by_epoch[int(match.group(1))] = name
    if not by_epoch:
        raise FileNotFoundError(f"no checkpoint in {run_dir}")
    if epoch is None:
        epoch = max(by_epoch)
    if epoch not in by_epoch:
        raise FileNotFoundError(f"no checkpoint of epoch {epoch} in {run_dir}")
    return os.path.join(run_dir, by_epoch[epoch])


def mamba_scan_flops(n_sequences, length, d_inner, d_state):
    """FLOPs of Mamba's selective scan, which runs in a custom CUDA kernel no profiler sees.

    VMamba's count for the same kernel (flops_selective_scan_fn): 9 per token, channel and
    state for the recurrence h = exp(dt*A)*h + dt*B*x and the readout y = C*h, plus one per
    token and channel each for the skip term D*x and for the gate y*silu(z), both of which
    mamba_ssm's Mamba passes to the kernel."""
    return n_sequences * length * d_inner * (9 * d_state + 2)


def count_flops(net, inputs, forward=None):
    """FLOPs of one forward pass, one fused multiply-add counted as one FLOP (fvcore's
    convention). torch's FlopCounterMode counts the multiply and the add separately -- 2
    per multiply-add in every forward formula it has (mm, addmm, bmm, baddbmm,
    convolution) -- so its total is halved, which is exact. Only convolutions and matrix
    products are profiled; elementwise activations, normalisation, pooling and tensor
    reshapes are not. Mamba's scan is added from mamba_scan_flops; Mamba2's scan has no
    such count and is listed under 'uncounted'."""
    forward = forward or (lambda net, inputs: net(inputs))
    scan = {"flops": 0}
    uncounted, hooks = [], []
    try:
        from mamba_ssm import Mamba, Mamba2
    except ImportError:
        Mamba = Mamba2 = ()

    def add_scan(module, args, output):
        n_sequences, length = args[0].shape[:2]
        scan["flops"] += mamba_scan_flops(n_sequences, length, module.d_inner, module.d_state)

    for name, module in net.named_modules():
        if Mamba and isinstance(module, Mamba):
            hooks.append(module.register_forward_hook(add_scan))
        elif Mamba2 and isinstance(module, Mamba2):
            uncounted.append(f"{name or type(module).__name__}: Mamba2 scan")
    was_training = net.training
    net.eval()
    try:
        with torch.no_grad(), FlopCounterMode(display=False) as counter:
            forward(net, inputs)
    finally:
        for hook in hooks:
            hook.remove()
        net.train(was_training)
    profiled = int(counter.get_total_flops()) // 2
    return {"total": profiled + scan["flops"], "profiled": profiled, "mamba_scan": scan["flops"], "uncounted": uncounted}


def single_frame_loader(loader):
    """The loader's samples one at a time, in order, fetched in this process: a worker
    process would hide the fetch from the preprocessing clock."""
    return DataLoader(loader.dataset, batch_size=1, shuffle=False, num_workers=0, collate_fn=loader.collate_fn)


def _clock(device):
    """Wall-clock reading once the device has finished its queued work: CUDA calls return
    before they complete, so an unsynchronised reading would hand the remainder of one
    stage to the next."""
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    return time.perf_counter()


def measure_run(net, loader, pipeline, device, max_frames=None):
    """Per-frame seconds of each stage over the loader, and the forward pass's peak
    allocated GPU memory in bytes (CUDA only). Warm-up frames are measured like the rest
    and dropped by summarize()."""
    device = torch.device(device)
    on_cuda = device.type == "cuda"
    raw = {"preprocess_s": [], "inference_s": [], "postprocess_s": [], "peak_memory_bytes": []}
    net.eval()
    frames = iter(loader)
    n_frames = len(loader) if max_frames is None else min(len(loader), max_frames)
    with torch.no_grad():
        for _ in range(n_frames):
            start = _clock(device)
            batch = next(frames)
            inputs = pipeline.inputs(batch, device)
            fetched = _clock(device)

            if on_cuda:
                torch.cuda.reset_peak_memory_stats(device)
            before_forward = _clock(device)
            outputs = pipeline.forward(net, inputs)
            after_forward = _clock(device)
            if on_cuda:
                raw["peak_memory_bytes"].append(torch.cuda.max_memory_allocated(device))

            before_post = _clock(device)
            pipeline.postprocess(outputs, batch)
            after_post = _clock(device)

            in_forward = pipeline.take_postprocess_seconds_in_forward()
            raw["preprocess_s"].append(fetched - start)
            raw["inference_s"].append(after_forward - before_forward - in_forward)
            raw["postprocess_s"].append(after_post - before_post + in_forward)
    raw["frames"] = n_frames
    return raw


class Pipeline:
    """What one model does with a frame around its forward pass."""

    def __init__(self, features=None, train_config=None, split="test", net=None):
        self.features, self.train_config, self.split, self.net = features, train_config, split, net

    def inputs(self, batch, device):
        raise NotImplementedError

    def forward(self, net, inputs):
        return net(inputs)

    def postprocess(self, outputs, batch):
        raise NotImplementedError

    def take_postprocess_seconds_in_forward(self):
        """Seconds of the last forward pass that were post-processing; 0 unless the
        network decodes its own output."""
        return 0.0


class DetectionMapPipeline(Pipeline):
    """FFTRadNet and SSMRadNet on RADIal: decode the detection map at the 0.05 candidate
    threshold, then confidence 0.2 and NMS (FFTRadNet_val_evaluation through
    Metrics.update and GetDetMetrics); threshold the freespace map at 0.5."""

    def inputs(self, batch, device):
        return batch["ADC" if "ADC" in batch else "RD"].to(device).float()

    def postprocess(self, outputs, batch):
        out_obj = outputs["Detection"].detach().cpu().numpy().copy()
        out_seg = torch.sigmoid(outputs["Segmentation"]).detach().cpu().numpy().copy()
        frames = []
        for pred_obj, pred_map in zip(out_obj, out_seg):
            objects = np.asarray(decode(pred_obj, 0.05))
            if len(objects) > 0:
                objects = process_predictions_FFT(objects, confidence_threshold=0.2)
            frames.append((objects, pred_map[0] >= 0.5))
        return frames


class ConfmapPipeline(Pipeline):
    """RODNet's family and T-RODNet on CRUW: a sample is a window of frames. Each frame's
    confidence map is averaged over the windows that cover it, and the frames the window
    advances by (the loader's stride) are peak-picked and suppressed by location (L-NMS),
    as in RODNet_evaluation. A new sequence starts from an empty average."""

    def __init__(self, features=None, train_config=None, split="test", net=None,
                 n_class=3, rng_grid=None, agl_grid=None):
        super().__init__(features, train_config, split, net)
        self.n_class, self.rng_grid, self.agl_grid = n_class, rng_grid, agl_grid
        self.stride = train_config["test_stride"] if split == "test" else train_config["train_stride"]
        self.head = None
        self.sequence = None

    def inputs(self, batch, device):
        return batch["RA"].to(device).float()

    def confmaps(self, outputs):
        if hasattr(self.net, "stacked_num"):
            return outputs[-1].cpu().detach().numpy()
        return outputs.cpu().detach().numpy()

    def postprocess(self, outputs, batch):
        confmap_pred = self.confmaps(outputs)   # (1, n_class, win, range, angle)
        shape = confmap_pred.shape[1:2] + confmap_pred.shape[3:]
        sequence = batch["seq_name"][0] if "seq_name" in batch else None
        if self.head is None or sequence != self.sequence:
            self.head = ConfmapStack(shape)
            self.sequence = sequence
        node = self.head
        for i in range(confmap_pred.shape[2]):
            if node.next is None and i != confmap_pred.shape[2] - 1:
                node.next = ConfmapStack(shape)
            node.append(confmap_pred[0, :, i, :, :])
            node = node.next
        decoded = []
        for _ in range(self.stride):
            decoded.append(post_process_single_frame(self.head.confmap, self.train_config, self.n_class,
                                                     self.rng_grid, self.agl_grid))
            self.head = self.head.next
            if self.head is None:
                self.head = ConfmapStack(shape)
        return decoded


class RampCnnPipeline(ConfmapPipeline):
    """RAMP-CNN on UWCR: the same windowed decoding over its fused confidence maps, always
    advancing by the train stride (RAMP_CNN_evaluation)."""

    def __init__(self, features=None, train_config=None, split="test", net=None,
                 n_class=3, rng_grid=None, agl_grid=None):
        super().__init__(features, train_config, "val", net, n_class, rng_grid, agl_grid)

    def inputs(self, batch, device):
        return {feature: batch[feature].to(device).float() for feature in ("RD", "RA", "AD")}

    def confmaps(self, outputs):
        return np.expand_dims(outputs["confmap_pred"][-1].cpu().detach().numpy(), axis=0)


class RecordCruwPipeline(Pipeline):
    """RECORD on CRUW: one confidence map per window, for its last frame, peak-picked and
    suppressed by location (RECORD_CRUW_evaluation)."""

    def __init__(self, features=None, train_config=None, split="test", net=None,
                 n_class=3, rng_grid=None, agl_grid=None):
        super().__init__(features, train_config, split, net)
        self.n_class, self.rng_grid, self.agl_grid = n_class, rng_grid, agl_grid

    def inputs(self, batch, device):
        return batch["RA"].to(device).float()

    def postprocess(self, outputs, batch):
        return post_process_single_frame(outputs[0].cpu(), self.train_config, self.n_class,
                                         self.rng_grid, self.agl_grid)


class SegmentationPipeline(Pipeline):
    """RECORD on CARRADA: the class of each cell is the argmax over the class channels,
    copied to the CPU (RECORD_CARRADA_evaluation)."""

    def inputs(self, batch, device):
        return batch[self.features[0]].to(device).float()

    def postprocess(self, outputs, batch):
        return torch.argmax(outputs, axis=1).cpu()


class TRodnetCarradaPipeline(SegmentationPipeline):
    """T-RODNet on CARRADA predicts every frame of its window; the last one is the frame
    it is scored on (T_RODNet_CARRADA_evaluation)."""

    def postprocess(self, outputs, batch):
        return torch.argmax(outputs[:, :, -1], axis=1).cpu()


class MvrecordPipeline(Pipeline):
    """MV-RECORD on CARRADA: one class mask per view, range-Doppler and range-angle
    (MVRECORD_CARRADA_evaluation)."""

    def inputs(self, batch, device):
        return {feature: batch[feature].to(device).float() for feature in self.features}

    def postprocess(self, outputs, batch):
        return torch.argmax(outputs["rd"], axis=1).cpu(), torch.argmax(outputs["ra"], axis=1).cpu()


class RaddetPipeline(Pipeline):
    """RADDet: YOLO-head rows above the confidence threshold, then 3D NMS (RADDet_evaluation)."""

    def inputs(self, batch, device):
        return batch["RAD"].to(device).float()

    def postprocess(self, outputs, batch):
        pred = outputs["pred"].detach().cpu().numpy()
        predictions = yoloheadToPredictions(pred[0], conf_threshold=self.train_config["confidence_threshold"])
        return nms(predictions, self.train_config["nms_iou3d_threshold"], self.net.input_size, sigma=0.3, method="nms")


class TransradPipeline(Pipeline):
    """TransRAD: decode the three scales, class NMS then location-aware NMS (TransRAD_evaluation)."""

    def inputs(self, batch, device):
        return batch["RAD"].to(device).float()

    def postprocess(self, outputs, batch):
        eval_size = self.train_config["eval_input_size"]
        rows = transrad_decode(outputs, self.net.n_class, eval_size)
        return transrad_nms(rows[0], eval_size, self.train_config["confidence_threshold"],
                            self.train_config["nms_iou3d_threshold"], self.train_config["la_nms_iou3d_threshold"])


class DarodPipeline(Pipeline):
    """DAROD decodes its boxes and runs their NMS inside forward(), in its `decoder`
    module, where every other detector does so after the forward pass. The decoder is
    timed through hooks and its time is moved from inference to post-processing, so the
    two columns mean the same thing for every model. The proposal NMS earlier in the
    forward pass stays in inference: the rest of the network needs its result."""

    def __init__(self, features=None, train_config=None, split="test", net=None):
        super().__init__(features, train_config, split, net)
        self._decoder_started = None
        self._decoder_seconds = 0.0
        net.decoder.register_forward_pre_hook(self._decoder_start)
        net.decoder.register_forward_hook(self._decoder_end)

    def _device(self):
        parameter = next(self.net.parameters(), None)
        return parameter.device if parameter is not None else torch.device("cpu")

    def _decoder_start(self, module, args):
        self._decoder_started = _clock(self._device())

    def _decoder_end(self, module, args, output):
        self._decoder_seconds += _clock(self._device()) - self._decoder_started

    def inputs(self, batch, device):
        return batch["RD"].to(device).float()

    def postprocess(self, outputs, batch):
        if outputs["decoder_output"] is None:
            return None
        return tuple(part.cpu().numpy() for part in outputs["decoder_output"])

    def take_postprocess_seconds_in_forward(self):
        seconds, self._decoder_seconds = self._decoder_seconds, 0.0
        return seconds


class CrossAttentionPipeline(Pipeline):
    """RadarCrossAttention: peaks of the class heat map above 0.1, shifted by the predicted
    centre offset, then 2 m NMS -- the prediction half of the evaluation's
    _detect_and_associate. It runs on the device, driven by a Python loop over the peaks."""

    def __init__(self, features=None, train_config=None, split="test", net=None):
        super().__init__(features, train_config, split, net)
        self._size = None

    def inputs(self, batch, device):
        return {feature: batch[feature].to(device).float() for feature in ("RD", "RA", "AD")}

    def postprocess(self, outputs, batch):
        pred_map = torch.sigmoid(outputs["pred_mask"])
        device = pred_map.device
        pred_c = (8 * (torch.sigmoid(outputs["pred_center"]) - 0.5)
                  if torch.is_tensor(outputs["pred_center"]) else 0)
        size = batch["gt_mask"].size()
        if size != self._size:
            mask, peak_cls = create_default(size, kernel_window=(3, 5))
            self._mask, self._peak_cls, self._size = mask.to(device), peak_cls.to(device), size
        pred_intent, pred_idx = peaks_detect(pred_map, self._mask, self._peak_cls, heat_thresh=0.1)
        pred_idx = distribute(pred_idx, device)
        if torch.is_tensor(pred_c):
            pred_idx = update_peaks(pred_c, pred_idx)
        return association(pred_intent, pred_idx, device)


class PillarPipeline(Pipeline):
    """PointPillars and RadarPillars on View-of-Delft: decode the anchor boxes, then rotated
    NMS per class (RadarPillar_evaluation, up to the conversion to the devkit's format)."""

    def inputs(self, batch, device):
        return {key: (value.to(device) if torch.is_tensor(value) else value) for key, value in batch.items()}

    def postprocess(self, outputs, batch):
        from detection_vis_backend.networks.radarpillar import postprocess
        return postprocess(outputs, self.net.net.anchors, dict(self.net.model_cfg), self.train_config)


PIPELINE_FOR_EVALUATION = {
    evaluate.FFTRadNet_evaluation: DetectionMapPipeline,
    evaluate.RODNet_evaluation: ConfmapPipeline,
    evaluate.RECORD_CRUW_evaluation: RecordCruwPipeline,
    evaluate.RECORD_CARRADA_evaluation: SegmentationPipeline,
    evaluate.T_RODNet_CARRADA_evaluation: TRodnetCarradaPipeline,
    evaluate.MVRECORD_CARRADA_evaluation: MvrecordPipeline,
    evaluate.RADDet_evaluation: RaddetPipeline,
    evaluate.TransRAD_evaluation: TransradPipeline,
    evaluate.DAROD_evaluation: DarodPipeline,
    evaluate.RAMP_CNN_evaluation: RampCnnPipeline,
    evaluate.RadarCrossAttention_evaluation: CrossAttentionPipeline,
    evaluate.RadarPillar_evaluation: PillarPipeline,
}


def _range_angle_grids(dataset_type):
    """(rng_grid, agl_grid) of the confidence-map models, from the radar configuration
    their evaluation functions read."""
    root = os.environ.get("DATASET_ROOTDIR", "/home/kangle/dataset")
    if dataset_type == "CRUW":
        with open(os.path.join(root, "CRUW", "sensor_config_rod2021.json")) as f:
            radar_cfg = json.load(f)["radar_cfg"]
    else:
        with open(os.path.join(root, "UWCR", "Automotive", "radar_config.json")) as f:
            radar_cfg = json.load(f)
    return confmap2ra(radar_cfg, name="range"), confmap2ra(radar_cfg, name="angle")


def make_pipeline(model_type, dataset_type, features, train_config, split, net):
    """The pipeline of a wired (model, dataset) pair, chosen by its evaluation function."""
    evaluation = eval_func_dict.get((model_type, dataset_type))
    if evaluation is None:
        raise ValueError(f"{model_type} on {dataset_type} has no evaluation, so nothing to measure")
    pipeline = PIPELINE_FOR_EVALUATION[evaluation]
    if issubclass(pipeline, (ConfmapPipeline, RecordCruwPipeline)):
        rng_grid, agl_grid = _range_angle_grids(dataset_type)
        n_class = net.n_class if dataset_type == "UWCR" else 3
        return pipeline(features, train_config, split, net, n_class=n_class, rng_grid=rng_grid, agl_grid=agl_grid)
    return pipeline(features, train_config, split, net)
