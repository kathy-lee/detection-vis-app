import numpy as np
import pandas as pd
import pkbar
import torch
import time
import os
import math
import json
import cv2
import polarTransform
import torch.nn as nn

from loguru import logger

from detection_vis_backend.datasets.dataset import DatasetFactory
from detection_vis_backend.datasets.adapters import make_sample_adapter, MIGRATED
from detection_vis_backend.networks.network import NetworkFactory
from detection_vis_backend.train.utils import post_process_single_frame, get_class_name, worldToImage, decode, process_predictions_FFT, yoloheadToPredictions, nms, create_default, peaks_detect, distribute, update_peaks, association, pol2cord, orent, radarpillar_collate
from detection_vis_backend.networks.radarpillar import postprocess
from detection_vis_backend.networks.transrad import decode_predictions as transrad_decode, nms_class_then_location as transrad_nms

def radarpillar_display_geometry(res, dataset_inst, sample_id):
    """Pure output formatter for RadarPillar (like carrada_seg_readout below):
    postprocessed boxes -> display geometry, computed backend-side because the
    frontend has no calib access.

    image_boxes rows = 8 image-projected corners + cls + score (18 values), via
    the frame's calib (intrinsic @ (R0_ext @ V2C_ext) -- the same matrices the
    dataset's label transform uses); boxes whose center is behind the camera are
    skipped from image_boxes only. bev_boxes rows = 4 BEV corners (bottom face,
    radar frame) + cls + score (10 values).
    """
    from detection_vis_backend.networks.radarpillar import box_corners_3d
    intrinsic, R0_ext, V2C_ext = dataset_inst._calib_radar(sample_id)
    cam_from_radar = R0_ext @ V2C_ext
    image_boxes, bev_boxes = [], []
    for box, label, score in zip(res["boxes"], res["labels"], res["scores"]):
        corners = box_corners_3d(box)                              # (8,3), bottom 4 first
        bev = corners[:4, :2].flatten().tolist()
        bev_boxes.append([float(v) for v in bev] + [int(label), float(score)])
        hom = np.concatenate([corners, np.ones((8, 1))], axis=1)   # (8,4)
        cam = (cam_from_radar @ hom.T).T                           # (8,4)
        if cam[:, 2].min() <= 0.1:                                 # behind the camera
            continue
        img = (intrinsic @ cam.T).T                                # (8,3)
        img = img[:, :2] / img[:, 2:3]
        image_boxes.append([float(v) for v in img.flatten()] + [int(label), float(score)])
    return {"image_boxes": image_boxes, "bev_boxes": bev_boxes, "viz_idx": sample_id}


def carrada_seg_readout(pred_logits, gt_mask, viz_idx):
    """Segmentation readout for CARRADA (RECORD / MVRECORD).

    CARRADA is a semantic-segmentation task (dense masks, wce_w10sdice, mIoU) -- not detection.
    This mirrors evaluate.RECORD_CARRADA_evaluation / MVRECORD_CARRADA_evaluation, which do
    argmax over the class channels for both prediction and ground truth. We return per-pixel
    class-id maps (0=background,1=pedestrian,2=cyclist,3=car) plus viz_idx, the frame the mask
    belongs to (the last frame of the input window), so the UI can overlay them on the matching
    radar image.

    :param pred_logits: model output for one sample, shape (n_class, H, W)
    :param gt_mask: ground-truth mask, one-hot (n_class, H, W) or already label ids (H, W)
    :param viz_idx: frame index the masks correspond to
    :return: {"pred_mask": [[...]], "gt_mask": [[...]], "viz_idx": int}
    """
    pred_mask = np.argmax(pred_logits, axis=0)
    gt = np.asarray(gt_mask)
    gt_labels = np.argmax(gt, axis=0) if gt.ndim == 3 else gt

    # --- TEMP DIAGNOSTIC (remove after debugging the all-background predictions) ---
    # Does the model have ANY foreground signal, especially where objects actually are?
    flat = pred_logits.reshape(pred_logits.shape[0], -1)
    logger.info(f"[seg-diag] per-class mean logit: {np.round(flat.mean(axis=1), 3).tolist()}, "
                f"max logit: {np.round(flat.max(axis=1), 3).tolist()}")
    fg = gt_labels > 0
    if fg.any():
        fg_pred_vals, fg_pred_cnts = np.unique(pred_mask[fg], return_counts=True)
        bg_logit = pred_logits[0][fg]
        best_fg_logit = pred_logits[1:][:, fg].max(axis=0)
        logger.info(f"[seg-diag] at {int(fg.sum())} gt-foreground px -> pred dist "
                    f"{dict(zip(fg_pred_vals.tolist(), fg_pred_cnts.tolist()))}; "
                    f"mean bg logit={bg_logit.mean():.3f}, mean best-fg logit={best_fg_logit.mean():.3f}, "
                    f"mean margin (bg-fg)={(bg_logit - best_fg_logit).mean():.3f}")
    # --- END TEMP DIAGNOSTIC ---

    return {
        "pred_mask": pred_mask.astype(int).tolist(),
        "gt_mask": np.asarray(gt_labels).astype(int).tolist(),
        "viz_idx": int(viz_idx),
    }


def summarize_pred_objs(pred_objs):
    """Compact, log-friendly view of a prediction result.

    Segmentation results carry dense H*W masks; logging them verbatim floods the backend log
    with tens of thousands of numbers per view. For those, log the mask shape and per-class
    pixel counts instead. Detection results (short lists of objects) are logged as-is.
    """
    if not isinstance(pred_objs, dict):
        return pred_objs
    summary = {}
    for key, val in pred_objs.items():
        if isinstance(val, dict) and "pred_mask" in val:
            pred = np.asarray(val["pred_mask"])
            gt = np.asarray(val["gt_mask"])
            summary[key] = {
                "shape": list(pred.shape),
                "viz_idx": val.get("viz_idx"),
                "pred_class_counts": {int(c): int(n) for c, n in zip(*np.unique(pred, return_counts=True))},
                "gt_class_counts": {int(c): int(n) for c, n in zip(*np.unique(gt, return_counts=True))},
            }
        else:
            summary[key] = val
    return summary


def relocate_datafile(file_info, file_id):
    """Refresh a train_info.txt datafile record's filesystem locations from the DB.

    train_info.txt is a SNAPSHOT written when the model was trained, so the absolute
    paths in it are frozen. When the dataset directory moved, every model trained
    before the move kept pointing at the old location and /predict died on a bare
    ENOENT with nothing naming the cause -- 22 datafile references across 16 models
    (July 2026, `/home/kangle/Downloads/wd_disk_radar_data/...`).

    The files table is the live source of truth and already held the new paths, so
    resolve by id and take `path`/`config` from there. Only those two: `name` and
    `parse` describe WHAT was trained on, and the snapshot stays authoritative for
    that -- a renamed or re-parsed row must not silently change what a stored model
    is evaluated against.

    Falls back to the snapshot, unchanged, if the row is gone or the DB is
    unreachable, so inference still works against a deleted registration.
    """
    try:
        from data import crud
        from data.database import get_session_local
        db = get_session_local()
        try:
            row = crud.get_datafile(db, file_id)
        finally:
            db.close()
    except Exception as e:
        logger.warning(f"Could not reach the files table to re-resolve datafile {file_id} "
                       f"({e}); using the paths frozen in train_info.txt")
        return file_info

    if row is None:
        logger.warning(f"Datafile {file_id} is no longer registered; using the paths "
                       f"frozen in train_info.txt")
        return file_info

    fresh = dict(file_info)
    for key in ("path", "config"):
        live = getattr(row, key, None)
        if live is not None and live != fresh.get(key):
            # Loud: an id whose path changed is normally a moved dataset, but it would
            # look identical if the id had been reused for different data.
            logger.warning(f"Datafile {file_id} {key} moved since training: "
                           f"{fresh.get(key)!r} -> {live!r}; using the registered value")
            fresh[key] = live
    return fresh


def infer(model, checkpoint_id, sample_id, file_id, split_type):
    model_rootdir = os.getenv('MODEL_ROOTDIR')
    parameter_path = os.path.join(model_rootdir, model, "train_info.txt")
    with open(parameter_path, 'r') as f:
        parameters = json.load(f)
    if not parameters:
        raise ValueError("Parameters are empty")

    logger.info(f"Inference with model {model} (checkpoint {checkpoint_id}) on no.{sample_id}th {split_type} sample:")
    logger.info(parameters)

    # Get input data and groundtruth label info
    dataset_factory = DatasetFactory()
    file_info = next((d for d in parameters["datafiles"] if d.get("id") == file_id), None)
    if file_info is None:
        raise ValueError(f"Model {model} was not trained on datafile {file_id}")
    file_info = relocate_datafile(file_info, file_id)
    logger.info(file_info)
    dataset_type = file_info["parse"]
    # dataset_inst = dataset_factory.get_instance(dataset_type, file_id)
    # dataset_inst.parse(file_id, file_info["path"], file_info["name"], file_info["config"])
    # Mixed case like train.py's CARRADA branch: parse on the shared instance (file state
    # must persist, and parse-once skips the redundant 2-20s re-parse on every /predict),
    # then configure a request copy so this model's config cannot leak into other requests.
    dataset_factory.get_shared_instance(
        dataset_type, file_id,
        parse_args=(file_id, file_info["path"], file_info["name"], file_info["config"]))
    dataset_inst = dataset_factory.get_instance(dataset_type, file_id)
    if dataset_type in MIGRATED:
        # Sample building lives in the adapter; the parser needs no per-request config.
        sample_source = make_sample_adapter(
            parameters["model_config"]["class"], dataset_inst, parameters["features"],
            parameters["model_config"], parameters["train_config"], splittype=split_type)
    else:
        dataset_inst.prepare_for_train(parameters["features"], parameters["train_config"], parameters["model_config"], split_type)
        sample_source = dataset_inst
    # Augmentation is a training-only concern, but prepare_for_train enables it unconditionally
    # (CARRADA sets self.transformations regardless of split type), so it leaks into
    # inference. That breaks inference two ways:
    #   - the input gets randomly flipped relative to the un-augmented feature image the UI
    #     draws gt on, so masks/boxes land in the wrong place;
    #   - RADDet's flips used to return negative-stride views that torch.tensor() rejected.
    # Disable augmentation for every dataset and adapter that supports it. Classes that
    # never flip simply ignore the attribute.
    sample_source.transformations = None
    data = sample_source[sample_id]
    gt_labels = dataset_inst.get_label(parameters["features"], sample_id)
    logger.info("Get data sample.")
    # The model predicts for the LAST frame of the input window: __getitem__[sample_id] loads
    # frame (sample_id + win_frames - 1), while /feature indexes frames directly. Align the
    # display frame to the mask's frame so the CARRADA segmentation overlay lines up.
    win_frames = getattr(dataset_inst, "win_frames", 1)
    viz_idx = sample_id + win_frames - 1

    # Initialize the model
    model_config = parameters["model_config"]
    network_factory = NetworkFactory()
    model_type = model_config['class']
    model_config.pop('class', None)
    net = network_factory.get_instance(model_type, model_config)

    # set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load the model checkpoint
    model_rootdir = os.getenv('MODEL_ROOTDIR')
    model_path = os.path.join(model_rootdir, model)
    checkpoint = [file for file in os.listdir(model_path) if f"epoch{checkpoint_id:02}" in file][0]
    dict = torch.load(os.path.join(model_path, checkpoint), map_location=device)
    # Checkpoints save net.state_dict() after init_lossfunc() attached the loss criterion
    # as a submodule (e.g. MVRECORD rd/ra_criterion, RECORD criterion when the loss is a
    # weighted CE), so they carry criterion.* buffers. Inference never calls init_lossfunc,
    # so those keys are "unexpected". strict=False ignores them; shape mismatches on real
    # network keys still raise (verified), so this does not mask config/shape errors.
    # net.load_state_dict(dict['net_state_dict'])
    incompatible = net.load_state_dict(dict['net_state_dict'], strict=False)
    if incompatible.missing_keys:
        logger.warning(f"Missing network keys when loading checkpoint: {incompatible.missing_keys}")
    if incompatible.unexpected_keys:
        logger.debug(f"Ignored non-network keys in checkpoint: {incompatible.unexpected_keys}")
    net.to(device)
    logger.info("Loaded the model.")
    
    # Prediction
    net.eval()
    with torch.set_grad_enabled(False):
        if model_type == "FFTRadNet":
            # input_data: [radar_FFT, segmap,out_label,box_labels,image]
            input = torch.tensor(data["RD"]).unsqueeze(0).to(device).float()
            output = net(input)
            pred_obj = output['Detection'].detach().cpu().numpy().copy()[0]
            pred_obj = decode(pred_obj,0.05)
            pred_obj = np.asarray(pred_obj)
            # process prediction: polar to cartesian, NMS...
            if(len(pred_obj)>0):
                pred_obj = process_predictions_FFT(pred_obj,confidence_threshold=0.2)
            pred_objs = []
            for box in pred_obj:
                box = box[1:]
                u1,v1 = worldToImage(-box[2],box[1],0)
                u2,v2 = worldToImage(-box[0],box[1],1.6)
                u1 = int(u1/2)
                v1 = int(v1/2)
                u2 = int(u2/2)
                v2 = int(v2/2)
                pred_objs.append([u1, v1, u2, v2])
            #feature_show_pred = "image"
            pred_objs = {"image": pred_objs}
        elif model_type in ("RODNet_CDC", "RODNet_CDCv2", "RODNet_HG", "RODNet_HGv2", "RODNet_HGwI", "RODNet_HGwIv2", "RadarFormer_hrformer2d"):
            input = torch.tensor(data["RA"]).unsqueeze(0).to(device).float()
            logger.info(f"Input data: {input.shape}")
            output = net(input)
            logger.info(f"Network output prediction: {output.shape}")
            if hasattr(net, 'stacked_num'):
                confmap_pred = output[-1].cpu().detach().numpy()  # (1, 4, 32, 128, 128)
            else:
                confmap_pred = output.cpu().detach().numpy()
            pred_objs = post_process_single_frame(confmap_pred[0,:,0,:,:], parameters["train_config"], dataset_inst.n_class, dataset_inst.rng_grid, dataset_inst.agl_grid) #[B, win_size, max_dets, 4]
            logger.info("Post-process finished.")
            # filter invalid predictions
            mask = pred_objs[:, 0] != -1 
            pred_objs = pred_objs[mask]
            # limit conf value
            mask = pred_objs[:, -1] > 1  
            pred_objs[mask, -1] = 1
            # reorganize the items: [row_id, col_id, cls_id, conf_value]
            pred_objs = pred_objs[:, [1, 2, 0, 3]]
            #feature_show_pred = "RA"
            pred_objs = {"RA": pred_objs.tolist()}
        elif model_type in ("RECORD", "RECORDNoLstm", "RECORDNoLstmMulti") and dataset_type == "CRUW":
            input = torch.tensor(data["RA"]).unsqueeze(0).to(device).float()
            output = net(input)
            confmap_pred = output[0].detach().cpu().numpy()
            pred_objs = post_process_single_frame(confmap_pred, parameters["train_config"], dataset_inst.n_class, dataset_inst.rng_grid, dataset_inst.agl_grid) #[B, win_size, max_dets, 4]
            # filter invalid predictions
            mask = pred_objs[:, 0] != -1 
            pred_objs = pred_objs[mask]
            # limit conf value
            mask = pred_objs[:, -1] > 1  
            pred_objs[mask, -1] = 1
            # reorganize the items: [row_id, col_id, cls_id, conf_value]
            pred_objs = pred_objs[:, [1, 2, 0, 3]]
            #feature_show_pred = "RA"
            pred_objs = {"RA": pred_objs.tolist()}
        elif model_type == "RECORD" and dataset_type == "CARRADA":
            feature = parameters["features"][0]
            input = torch.tensor(data[feature]).unsqueeze(0).to(device).float()
            output = net(input)
            # Segmentation readout: CARRADA-trained RECORD is a semantic-segmentation model
            # (dense masks, wce_w10sdice, mIoU). Argmax the class channels for pred and gt,
            # mirroring evaluate.RECORD_CARRADA_evaluation. The old detection post-processing
            # (peak-pick -> points) is kept below for reference but does not fit a seg model.
            # confmap_pred = output[0].detach().cpu().numpy()
            # if feature == "RA":
            #     pred_objs = post_process_single_frame(confmap_pred, parameters["train_config"], dataset_inst.n_class, dataset_inst.rng_grid, dataset_inst.agl_grid) #[B, win_size, max_dets, 4]
            # elif feature == "RD":
            #     pred_objs = post_process_single_frame(confmap_pred, parameters["train_config"], dataset_inst.n_class, dataset_inst.rng_grid, dataset_inst.dpl_grid) #[B, win_size, max_dets, 4]
            # else:
            #     raise ValueError("Feature type not supported in inference.")
            # # filter invalid predictions
            # mask = pred_objs[:, 0] != -1
            # pred_objs = pred_objs[mask]
            # # limit conf value
            # mask = pred_objs[:, -1] > 1
            # pred_objs[mask, -1] = 1
            # # reorganize the items: [row_id, col_id, cls_id, conf_value]
            # pred_objs = pred_objs[:, [1, 2, 0, 3]]
            # pred_objs = {feature: pred_objs.tolist()}
            pred_logits = output[0].detach().cpu().numpy()   # (n_class, H, W)
            pred_objs = {feature: carrada_seg_readout(pred_logits, data['gt_mask'], viz_idx)}
        elif model_type == "MVRECORD":
            # input = (torch.tensor(data['RD']).unsqueeze(0).to(device).float(), 
            #          torch.tensor(data['RA']).unsqueeze(0).to(device).float(), 
            #          torch.tensor(data['AD']).unsqueeze(0).to(device).float())
            input = {feature: torch.tensor(data[feature]).unsqueeze(0).to(device).float() for feature in parameters["features"]}
            output = net(input)
            # Segmentation readout: MV-RECORD is a semantic-segmentation model (two dense
            # decoders RD + RA, wce_w10sdice, mIoU). forward returns per-view maps of shape
            # (B, n_class, H, W); index [0] and argmax the class channels per view, mirroring
            # evaluate.MVRECORD_CARRADA_evaluation. Old detection post-processing (peak-pick ->
            # points) is kept below for reference but does not fit a seg model.
            # confmap_pred_ra = output['ra'][0].detach().cpu().numpy()
            # confmap_pred_rd = output['rd'][0].detach().cpu().numpy()
            # pred_objs_ra = post_process_single_frame(confmap_pred_ra, parameters["train_config"], dataset_inst.n_class, dataset_inst.rng_grid, dataset_inst.agl_grid) #[B, win_size, max_dets, 4]
            # pred_objs_rd = post_process_single_frame(confmap_pred_rd, parameters["train_config"], dataset_inst.n_class, dataset_inst.rng_grid, dataset_inst.dpl_grid) #[B, win_size, max_dets, 4]
            # pred_objs = {"RA": pred_objs_ra, "RD": pred_objs_rd}
            # # filter invalid predictions
            # mask = pred_objs["RA"][:, 0] != -1
            # pred_objs["RA"] = pred_objs["RA"][mask]
            # # limit conf value
            # mask = pred_objs["RA"][:, -1] > 1
            # pred_objs["RA"][mask, -1] = 1
            # # reorganize the items: [row_id, col_id, cls_id, conf_value]
            # pred_objs["RA"] = pred_objs["RA"][:, [1, 2, 0, 3]]
            # # filter invalid predictions
            # mask = pred_objs["RD"][:, 0] != -1
            # pred_objs["RD"] = pred_objs["RD"][mask]
            # # limit conf value
            # mask = pred_objs["RD"][:, -1] > 1
            # pred_objs["RD"][mask, -1] = 1
            # # reorganize the items: [row_id, col_id, cls_id, conf_value]
            # pred_objs["RD"] = pred_objs["RD"][:, [1, 2, 0, 3]]
            # pred_objs = {"RA": pred_objs_ra.tolist(), "RD": pred_objs_rd.tolist()}
            pred_logits_ra = output['ra'][0].detach().cpu().numpy()   # (n_class, H, W)
            pred_logits_rd = output['rd'][0].detach().cpu().numpy()   # (n_class, H, W)
            pred_objs = {
                "RA": carrada_seg_readout(pred_logits_ra, data['ra_mask'], viz_idx),
                "RD": carrada_seg_readout(pred_logits_rd, data['rd_mask'], viz_idx),
            }
        elif model_type == "RADDet":
            input = torch.tensor(data['RAD']).unsqueeze(0).to(device).float()
            output = net(input)
            train_config = parameters["train_config"]
            pred = output['pred'].detach().cpu()
            predicitons = yoloheadToPredictions(pred[0], conf_threshold=train_config["confidence_threshold"])
            # input_size is an architecture property (RAD tensor dims), so it lives in the model
            # config, not the train config -- train_config has no 'input_size' key.
            pred_objs = nms(predicitons, train_config["nms_iou3d_threshold"], model_config["input_size"], sigma=0.3, method="nms")
            rd_objs = []
            ra_objs = []
            for i in range(len(pred_objs)):
                bbox3d = pred_objs[i, :6]   # [x=range, y=azimuth, z=doppler, w, h, d] center+size
                cls = int(pred_objs[i, 7])
                # Old output was center+size and range-first, but the frontend box renderer expects
                # CORNER boxes [x1, y1, x2, y2] with x=horizontal (doppler/azimuth), y=vertical
                # (range). Convert exactly like RADDet.get_label so predictions render like the gt.
                # rd_objs.append([bbox3d[0], bbox3d[2], bbox3d[3], bbox3d[5], cls])
                # ra_objs.append([bbox3d[0], bbox3d[1], bbox3d[3], bbox3d[4], cls])
                # RD view: x-axis=doppler (bbox3d[2],[5]), y-axis=range (bbox3d[0],[3])
                yc, xc, h, w = bbox3d[0], bbox3d[2], bbox3d[3], bbox3d[5]
                rd_objs.append([int(xc - w / 2), int(yc - h / 2), int(xc + w / 2), int(yc + h / 2), cls])
                # RA view: x-axis=azimuth (bbox3d[1],[4]), y-axis=range (bbox3d[0],[3])
                yc, xc, h, w = bbox3d[0], bbox3d[1], bbox3d[3], bbox3d[4]
                ra_objs.append([int(xc - w / 2), int(yc - h / 2), int(xc + w / 2), int(yc + h / 2), cls])
            pred_objs = {"RD": rd_objs, "RA": ra_objs}
        elif model_type == "TransRAD":
            input = torch.tensor(data['RAD']).unsqueeze(0).to(device).float()
            output = net(input)
            train_config = parameters["train_config"]
            eval_size = train_config["eval_input_size"]
            rows = transrad_decode(output, net.n_class, eval_size)[0]
            pred_objs = transrad_nms(rows, eval_size, train_config["confidence_threshold"],
                                     train_config["nms_iou3d_threshold"], train_config["la_nms_iou3d_threshold"])
            rd_objs = []
            ra_objs = []
            for i in range(len(pred_objs)):
                bbox3d = pred_objs[i, :6]   # [x=range, y=azimuth, z=doppler, w, h, d], z in the 64-bin cube
                cls = int(pred_objs[i, 7])
                # Same corner-box conversion as the RADDet branch above, so the UI draws
                # TransRAD boxes exactly where it draws RADDet's and the gt from get_label.
                yc, xc, h, w = bbox3d[0], bbox3d[2], bbox3d[3], bbox3d[5]
                rd_objs.append([int(xc - w / 2), int(yc - h / 2), int(xc + w / 2), int(yc + h / 2), cls])
                yc, xc, h, w = bbox3d[0], bbox3d[1], bbox3d[3], bbox3d[4]
                ra_objs.append([int(xc - w / 2), int(yc - h / 2), int(xc + w / 2), int(yc + h / 2), cls])
            pred_objs = {"RD": rd_objs, "RA": ra_objs}
        elif model_type == "DAROD":
            input = torch.tensor(data['RD']).unsqueeze(0).to(device).float()
            output = net(input)
            pred_boxes, pred_labels, pred_scores = output["decoder_output"]
            pred_boxes = pred_boxes.detach().cpu().numpy()[0].astype(float)
            pred_labels = pred_labels.detach().cpu().numpy()[0].astype(float)
            pred_scores = pred_scores.detach().cpu().numpy()[0].astype(float)
            pred_labels = pred_labels - 1
            pred_objs = [list(pred_boxes[i]) + [pred_labels[i], pred_scores[i]] for i in range(len(pred_boxes))]
            pred_objs = {"RD": pred_objs}
        elif model_type in ("RadarPillar", "PointPillar"):
            input = radarpillar_collate([dataset_inst[sample_id]])
            input = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in input.items()}
            output = net(input)
            res = postprocess(output, net.net.anchors, model_config, parameters["train_config"])[0]
            pred_objs = radarpillar_display_geometry(res, dataset_inst, sample_id)
        elif model_type == "RAMP_CNN":
            #input = (data['RA'].to(device).float(), data['RD'].to(device).float(), data['AD'].to(device).float())
            input = {feature: torch.tensor(data[feature]).unsqueeze(0).to(device).float() for feature in parameters["features"]}
            output = net(input)
            confmap_pred = output['confmap_pred'].cpu().detach().numpy() 
            pred_objs = post_process_single_frame(confmap_pred[0,:,0,:,:], parameters["train_config"], dataset_inst.n_class, dataset_inst.rng_grid, dataset_inst.agl_grid) #[B, win_size, max_dets, 4]
            # filter invalid predictions
            mask = pred_objs[:, 0] != -1 
            pred_objs = pred_objs[mask]
            # limit conf value
            mask = pred_objs[:, -1] > 1  
            pred_objs[mask, -1] = 1
            # reorganize the items: [row_id, col_id, cls_id, conf_value]
            pred_objs = pred_objs[:, [1, 2, 0, 3]]
            #feature_show_pred = "RA"
            pred_objs = {"RA": pred_objs.tolist()}
        elif model_type == "RadarCrossAttention":
            #input = (data['ra_matrix'].to(device).float(), data['rd_matrix'].to(device).float(), data['ad_matrix'].to(device).float())  
            input = {feature: torch.tensor(data[feature]).unsqueeze(0).to(device).float() for feature in parameters["features"]}
            output = net(input)
            pred_map = torch.sigmoid(output["pred_mask"]).detach().cpu()
            pred_c = 8 * (torch.sigmoid(output["pred_center"]) - 0.5)
            pred_c = pred_c.detach().cpu().numpy()
            pred_o = output["pred_orent"].detach().cpu().numpy()
            mask, peak_cls = create_default(pred_map.shape, kernel_window=(3,5))
            pred_intent, pred_idx = peaks_detect(pred_map, mask, peak_cls, heat_thresh=0.6)
            pred_idx= distribute(pred_idx)
            pred_idx = update_peaks(pred_c, pred_idx)
            pred_idx, pred_intent= association(pred_intent, pred_idx, device)
            logger.debug(pred_idx)

            ra_objs = []
            for cnt, cord in enumerate(pred_idx):
                p_cls, p_row, p_col = int(cord[1]), cord[2], cord[3]
                orientation = orent(pred_o[p_row,::], int(p_row//4), int(p_col//4), pred=True)
                ra_objs.append([p_row, p_col, p_cls, pred_intent[cnt], orientation]) # [row_id, col_id, cls_id, confidence, heading]
                logger.debug(f"Class: {p_cls}, Confidence: {pred_intent[cnt]},\
                    Range: {50*p_row/256}, Angle(deg): {p_col*57.29*2/256 - 57.29},\
                    Heading: {orientation}")  
            pred_objs = {"RA": ra_objs}
        else:
            raise ValueError(f"Inference of the chosen model type {model_type} is not supported")
    
    logger.info(f"Prediction result: {summarize_pred_objs(pred_objs)}")
    return pred_objs


def display_inference_FFTRadNet(image, input, model_outputs, obj_labels, train_config=None):
    # Usage: #pred_image = display_inference_FFTRadNet(data[4], data[0], output, data[3])
    
    # Model outputs
    pred_obj = model_outputs['Detection'].detach().cpu().numpy().copy()[0]
    out_seg = torch.sigmoid(model_outputs['Segmentation']).detach().cpu().numpy().copy()[0,0]
    logger.debug(f"pred_obj before decode: {pred_obj.shape}")
    # Decode the output detection map
    pred_obj = decode(pred_obj,0.05)
    logger.debug(f"pred_obj after decode: {type(pred_obj)}, {len(pred_obj)}")
    pred_obj = np.asarray(pred_obj)

    # process prediction: polar to cartesian, NMS...
    if(len(pred_obj)>0):
        pred_obj = process_predictions_FFT(pred_obj,confidence_threshold=0.2)

    ## FFT
    FFT = np.abs(input[...,:16]+input[...,16:]*1j).mean(axis=2)
    PowerSpectrum = np.log10(FFT)
    # rescale
    PowerSpectrum = (PowerSpectrum -PowerSpectrum.min())/(PowerSpectrum.max()-PowerSpectrum.min())*255
    PowerSpectrum = cv2.cvtColor(PowerSpectrum.astype('uint8'),cv2.COLOR_GRAY2BGR)
    ## Image
    for box in pred_obj:
        box = box[1:]
        u1,v1 = worldToImage(-box[2],box[1],0)
        u2,v2 = worldToImage(-box[0],box[1],1.6)

        u1 = int(u1/2)
        v1 = int(v1/2)
        u2 = int(u2/2)
        v2 = int(v2/2)

        image = cv2.rectangle(image, (u1,v1), (u2,v2), (0, 0, 255), 3)
    
    for box in obj_labels:
        box = box[6:]
        box = [int(x/2) for x in box]
        image = cv2.rectangle(image, (box[0],box[1]), (box[2],box[3]), (255, 0, 0), 2)

    RA_cartesian,_=polarTransform.convertToCartesianImage(np.moveaxis(out_seg,0,1),useMultiThreading=True,
        initialAngle=0, finalAngle=np.pi,order=0,hasColor=False)
    # Make a crop on the angle axis
    RA_cartesian = RA_cartesian[:,256-100:256+100]
    
    RA_cartesian = np.asarray((RA_cartesian*255).astype('uint8'))
    RA_cartesian = cv2.cvtColor(RA_cartesian, cv2.COLOR_GRAY2BGR)
    RA_cartesian = cv2.resize(RA_cartesian,dsize=(400,512))
    RA_cartesian=cv2.flip(RA_cartesian,flipCode=-1)
    return np.hstack((PowerSpectrum,image[:512],RA_cartesian))
