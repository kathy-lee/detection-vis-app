"""Sample adapters: one torch Dataset per (model, dataset) sample-building behaviour.

Two roles:
  parser  -- the existing dataset class (e.g. RADDetDataset). Parses files once, then serves
             raw views and raw annotations per frame. Knows nothing about any model.
  adapter -- a SampleAdapter subclass. Wraps a parser and builds the samples one model
             consumes. This is what the DataLoader indexes.

Migrated datasets (every model that reads them goes through an adapter; the dataset class
has no model branches left):
  RADDetDataset -- adapters in raddet.py
  CRUW          -- adapters in cruw.py
  CARRADA       -- adapters in carrada.py
All other datasets still build samples in their own __getitem__ / prepare_for_train.
"""
from typing import Protocol, runtime_checkable

from torch.utils.data import Dataset


@runtime_checkable
class Parser(Protocol):
    """What every adapter may rely on. Checked by presence at construction (hasattr),
    not by inheritance: no dataset class inherits from this, and a test stub with
    these two methods and no parent is a Parser too. Per-dataset raw views
    (get_RAD, _get_collapsed, ...) are not part of it; an adapter that calls them
    declares which parser it is written for through its DATASET attribute."""

    def __len__(self) -> int: ...            # number of frames (windowing is the adapter's)

    def annotation(self, i: int): ...        # raw labels of frame i, in the dataset's format


class SampleAdapter(Dataset):
    """Base for one subclass per sample-building behaviour.

    MODELS lists every network class that reads DATASET this way (usually one; a future
    CRUW adapter would list all six RODNet variants). Subclasses implement __getitem__.
    """
    MODELS: tuple = ()
    DATASET: str = ""

    def __init__(self, parser, features, model_cfg, train_cfg, splittype=None):
        # features: the view list train() was asked for (e.g. ["RD"]). It is a separate
        # argument to CreateDataLoaders, not a key of either config.
        self.parser = parser
        self.features = features
        self.splittype = splittype
        # Augmentation policy is the adapter's; train-only. inference.py sets this to
        # None on the adapter it builds, exactly as it did on the dataset before.
        self.transformations = train_cfg.get("transformations")

    def __len__(self):
        return len(self.parser)


_REGISTRY = {}


def register(cls):
    """Class decorator: fan one adapter out to every model in cls.MODELS."""
    assert cls.MODELS and cls.DATASET, f"{cls.__name__} must set MODELS and DATASET"
    for model in cls.MODELS:
        key = (model, cls.DATASET)
        assert key not in _REGISTRY, f"{key} registered twice ({_REGISTRY[key].__name__}, {cls.__name__})"
        _REGISTRY[key] = cls
    return cls


def make_sample_adapter(model, parser, features, model_cfg, train_cfg, splittype=None):
    """The one construction point for training and inference alike."""
    if not isinstance(parser, Parser):
        missing = [m for m in ("__len__", "annotation") if not hasattr(parser, m)]
        raise TypeError(f"{type(parser).__name__} is not a Parser: missing {missing}")
    dataset = type(parser).__name__          # == files.parse, the name DatasetFactory used
    try:
        cls = _REGISTRY[(model, dataset)]
    except KeyError:
        raise ValueError(f"no sample adapter for model {model!r} on dataset {dataset!r}; "
                         f"registered: {sorted(_REGISTRY)}") from None
    return cls(parser, features, model_cfg, train_cfg, splittype)


# Per-dataset adapter modules register themselves on import. Imported last so they can
# import the names above.
from detection_vis_backend.datasets.adapters import raddet, cruw, carrada  # noqa: E402,F401

MIGRATED = frozenset(dataset for (_, dataset) in _REGISTRY)
