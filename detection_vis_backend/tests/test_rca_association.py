"""The vectorized RadarCrossAttention evaluator helpers must reproduce the reference's loops.

distribute / update_peaks / association in train/utils.py replace the reference's
Python-loop implementations (crossattention_radar_detector, Utils/val_utils.py:153-206),
which are O(n^2) with a torch.cat per element and take hours per val batch on an
untrained model. The reference versions are copied here VERBATIM (only pol2cord/distance
are shared) and the two are compared on random peak sets: identical kept peaks, in the
same order, with the same intensities.
"""
import numpy as np
import pytest
import torch

from detection_vis_backend.train.utils import (association, distribute, update_peaks,
                                               pol2cord, distance)


# ---- verbatim reference implementations (Utils/val_utils.py) ---------------------------

def ref_distribute(index_t, device='cpu'):
    index = torch.tensor(()).to(device=device)
    for i in range(len(index_t[0])):
        index = torch.cat((index, torch.stack((index_t[0][i],index_t[1][i],index_t[2][i],index_t[3][i]),dim=0)))

    index = torch.reshape(index,(len(index_t[0]),4))
    return index


def ref_update_peaks(pred_cen,pred_idx):
    for cnt,cord in enumerate(pred_idx):
        r_id = int(cord[2])
        a_id = int(cord[3])
        bat = int(cord[0])
        cord[2] += int(pred_cen[bat,0,r_id,a_id])
        cord[3] += int(pred_cen[bat,1,r_id,a_id])
        pred_idx[cnt]= cord
    return pred_idx


def ref_association(intensity, index,device ="cpu"):
    idx_list = torch.argsort(intensity)
    idx_list = torch.flip(idx_list,dims=[-1])
    out_intent =torch.Tensor().to(device=device)
    out_idx =torch.Tensor().to(device=device)

    while len(idx_list)!=0:
        out_intent=torch.cat((out_intent,torch.Tensor([intensity[idx_list[0]]]).to(device=device)))
        out_idx = torch.cat((out_idx, index[idx_list[0]]))
        p1_row , p1_col = index[idx_list[0]][2],index[idx_list[0]][3]
        frame_1, cls_1 = index[idx_list[0]][0],index[idx_list[0]][1]
        x1,y1 = pol2cord(p1_row,p1_col)
        idx_list = idx_list[1:]

        count = 0
        while len(idx_list)!=0 and count != len(idx_list):
            frame_2, cls_2 = index[idx_list[count]][0],index[idx_list[count]][1]
            if frame_2==frame_1:
                p2_row , p2_col = index[idx_list[count]][2],index[idx_list[count]][3]
                x2,y2 = pol2cord(p2_row,p2_col)
                dist = distance(x1,y1,x2,y2)

                if dist <2:
                   idx_list = torch.cat([idx_list[:count], idx_list[count+1:]])
                else:count +=1
            else: count +=1
    out_idx = torch.reshape(out_idx,(len(out_idx)//4,4))
    return out_idx,out_intent


# ---- fixtures --------------------------------------------------------------------------

def _peaks(gen, n, frames=3, classes=3, lo=(1, 2), hi=(254, 253), ties=False):
    """A torch.where-style index tuple plus intensities, on the interior the (3,5)
    max-pool mask allows (rows 1..254, cols 2..253)."""
    b = torch.randint(0, frames, (n,), generator=gen)
    c = torch.randint(0, classes, (n,), generator=gen)
    r = torch.randint(lo[0], hi[0] + 1, (n,), generator=gen)
    a = torch.randint(lo[1], hi[1] + 1, (n,), generator=gen)
    inten = torch.rand(n, generator=gen)
    if ties:
        inten = (inten * 20).round() / 20        # many exact duplicates
    return (b, c, r, a), inten


@pytest.mark.parametrize("seed,n,ties", [(0, 50, False), (1, 400, False), (2, 400, True),
                                          (3, 1500, False), (4, 1, False)])
def test_association_matches_reference(seed, n, ties):
    gen = torch.Generator().manual_seed(seed)
    idx_t, inten = _peaks(gen, n, ties=ties)
    ref_idx, ref_int = ref_association(inten.clone(), ref_distribute(idx_t))
    new_idx, new_int = association(inten.clone(), distribute(idx_t))
    assert torch.equal(new_idx, ref_idx), "kept peaks or their order differ"
    assert torch.equal(new_int, ref_int), "kept intensities differ"
    assert len(new_idx) < n or n == 1     # sanity: the NMS actually suppressed something


def test_association_empty():
    idx, inten = association(torch.Tensor(), distribute((torch.tensor([], dtype=torch.long),) * 4))
    assert idx.shape == (0, 4) and inten.shape == (0,)


def test_distribute_matches_reference():
    gen = torch.Generator().manual_seed(7)
    idx_t, _ = _peaks(gen, 300)
    assert torch.equal(distribute(idx_t), ref_distribute(idx_t))
    empty = (torch.tensor([], dtype=torch.long),) * 4
    assert torch.equal(distribute(empty), ref_distribute(empty))


def test_update_peaks_matches_reference_where_it_is_defined():
    """Offsets in (-4, 4) like 8*(sigmoid-0.5); peaks kept >=4 bins from every edge so the
    reference neither raises (past 255) nor wraps (below 0)."""
    gen = torch.Generator().manual_seed(11)
    idx_t, _ = _peaks(gen, 500, lo=(4, 4), hi=(251, 251))
    pred_cen = (torch.rand(3, 2, 256, 256, generator=gen) - 0.5) * 8
    ref = ref_update_peaks(pred_cen, ref_distribute(idx_t))
    new = update_peaks(pred_cen, distribute(idx_t))
    assert torch.equal(new, ref)


def test_update_peaks_clamps_at_the_edges():
    """Where the reference would raise or wrap, ours stays on the map."""
    idx = torch.tensor([[0., 0., 254., 100.], [0., 0., 1., 3.]])
    pred_cen = torch.zeros(1, 2, 256, 256)
    pred_cen[0, 0, 254, 100] = 3.9      # -> 257 in the reference: IndexError downstream
    pred_cen[0, 0, 1, 3] = -3.9         # -> -2 in the reference: wraps to row 254
    pred_cen[0, 1, 1, 3] = -3.9         # -> col 0
    out = update_peaks(pred_cen, idx)
    assert out[0, 2] == 255 and out[1, 2] == 0 and out[1, 3] == 0
