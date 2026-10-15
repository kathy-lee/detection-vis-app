"""Both on-disk numerical caches must be content-addressed by their producing code.

A stale numerical cache is the worst failure mode this codebase has available: the run
completes, every metric is computed, and the numbers are simply wrong, with nothing in
the output saying so. Both caches therefore put a hash of the code that PRODUCES the
arrays into the cache path, so editing a producer misses the cache instead of serving
values computed by the old code.

These are source-level assertions rather than behavioural ones on purpose -- they must
run anywhere, including where neither dataset is on disk, which is the same reason
test_dataset_state.py pins its ten classes unparsed.
"""
import ast
import os

import pytest

DATASET_PY = os.path.join(os.path.dirname(__file__), "..", "datasets", "dataset.py")

# (class, cache-dir attribute). Add a row here when a new on-disk numerical cache
# appears; a cache of file PATHS or of raw bytes off disk does not need one, only a
# cache of COMPUTED arrays.
CACHED_ARRAY_PRODUCERS = [
    ("UWCR", "frame_cache_dir"),
    ("RADDetDataset", "collapsed_cache_dir"),
    ("CARRADA", "ad_cache_dir"),
]


def _class_def(class_name):
    with open(DATASET_PY) as fh:
        tree = ast.parse(fh.read())
    for node in tree.body:
        if isinstance(node, ast.ClassDef) and node.name == class_name:
            return node
    raise AssertionError(f"class {class_name} not found in dataset.py")


def _assignment_to_self_attr(class_node, attr):
    """The Assign node writing self.<attr>, and the method containing it."""
    for func in ast.walk(class_node):
        if not isinstance(func, ast.FunctionDef):
            continue
        for node in ast.walk(func):
            if not isinstance(node, ast.Assign):
                continue
            for target in node.targets:
                if (isinstance(target, ast.Attribute) and target.attr == attr
                        and isinstance(target.value, ast.Name)
                        and target.value.id == "self"):
                    return node, func
    raise AssertionError(f"no assignment to self.{attr} in {class_node.name}")


def _calls(node):
    return [n for n in ast.walk(node) if isinstance(n, ast.Call)]


@pytest.mark.parametrize("class_name,attr", CACHED_ARRAY_PRODUCERS)
def test_cache_dir_is_content_addressed(class_name, attr):
    """The cache path must be built from a digest, not be a fixed string.

    Without this, the only thing standing between a preprocessing edit and a silently
    wrong metric is someone remembering to delete a directory by hand.
    """
    assign, _ = _assignment_to_self_attr(_class_def(class_name), attr)

    digests = [c for c in _calls(assign.value)
               if isinstance(c.func, ast.Attribute) and c.func.attr == "hexdigest"]
    assert digests, (
        f"{class_name}.{attr} is not content-addressed: no .hexdigest() in the path "
        f"expression, so editing a producer keeps serving the old cached arrays")


@pytest.mark.parametrize("class_name,attr", CACHED_ARRAY_PRODUCERS)
def test_cache_digest_covers_the_producing_code(class_name, attr):
    """The digested content must be the producers' source, not merely config values.

    Hashing the config alone would be a trap: it invalidates when a parameter changes
    but not when the arithmetic does, which is exactly the edit most likely to go
    unnoticed.
    """
    _, method = _assignment_to_self_attr(_class_def(class_name), attr)

    getsource = [c for c in _calls(method)
                 if isinstance(c.func, ast.Attribute) and c.func.attr == "getsource"
                 and isinstance(c.func.value, ast.Name) and c.func.value.id == "inspect"]
    assert getsource, (
        f"{class_name}.{attr}: {method.name} never calls inspect.getsource, so the "
        f"cache key cannot reflect changes to the code that produces the arrays")
