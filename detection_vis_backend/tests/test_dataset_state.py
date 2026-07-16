import ast
import copy
import os
import sys

import pytest

from detection_vis_backend.datasets.dataset import DatasetFactory

# Every class reachable through DatasetFactory.get_instance (it resolves via globals()
# in dataset.py, which imports ViewOfDelft and NuScenes at lines 30-31).
ALL_DATASET_CLASSES = [
    "RaDICaL", "RADIal", "RADIalRaw", "CRUW", "CARRADA",
    "RADDetDataset", "Astyx", "UWCR", "ViewOfDelft", "NuScenes",
]


@pytest.mark.parametrize("class_name", ALL_DATASET_CLASSES)
def test_copy_is_safe_for_every_dataset_class(class_name):
    """copy.copy must work on every dataset class and must NOT duplicate data.

    The state-ownership design hands each request a copy.copy of the cached instance.
    That is only viable if (a) copying does not raise, and (b) attribute values are
    shared by reference rather than duplicated -- CARRADA's parse output is 331 MB.

    Only 4 of these 10 classes were verifiable during design (the ones with datafiles
    in the dev DB). This pins all 10, unparsed, so it runs anywhere.
    """
    factory = DatasetFactory()
    # Build directly rather than via get_instance(), which this task has not changed yet.
    import detection_vis_backend.datasets.dataset as ds_mod
    inst = getattr(ds_mod, class_name)()

    dup = copy.copy(inst)

    assert dup is not inst, f"{class_name}: copy returned the same object"
    assert type(dup) is type(inst), f"{class_name}: copy changed the type"
    for key, value in inst.__dict__.items():
        assert getattr(dup, key) is value, (
            f"{class_name}.{key} was duplicated by copy.copy, not shared by reference"
        )


def test_require_shared_instance_rejects_a_copy():
    from detection_vis_backend.datasets.utils import require_shared_instance

    class Fake:
        pass

    shared = Fake()
    require_shared_instance(shared)  # no flag -> treated as shared -> must not raise

    request_copy = Fake()
    request_copy._request_scoped = True
    with pytest.raises(RuntimeError, match="shared instance"):
        require_shared_instance(request_copy)


def test_require_request_scope_rejects_the_shared_instance():
    from detection_vis_backend.datasets.utils import require_request_scope

    class Fake:
        pass

    request_copy = Fake()
    request_copy._request_scoped = True
    require_request_scope(request_copy)  # must not raise

    shared = Fake()
    with pytest.raises(RuntimeError, match="request copy"):
        require_request_scope(shared)


def _fresh_factory():
    """DatasetFactory is a singleton with a process-wide instance_dict; isolate tests."""
    factory = DatasetFactory()
    factory.instance_dict.clear()
    return factory


def test_get_instance_returns_a_request_scoped_copy():
    factory = _fresh_factory()
    a = factory.get_instance("CRUW", 9001)
    b = factory.get_instance("CRUW", 9001)

    assert a is not b, "each caller must get its own copy"
    assert a._request_scoped is True
    assert b._request_scoped is True
    shared = factory.get_shared_instance("CRUW", 9001)
    assert a is not shared and b is not shared
    assert getattr(shared, "_request_scoped", False) is False


def test_get_shared_instance_returns_the_same_object_every_time():
    factory = _fresh_factory()
    a = factory.get_shared_instance("CRUW", 9002)
    b = factory.get_shared_instance("CRUW", 9002)
    assert a is b


def test_configuring_a_copy_leaves_shared_untouched():
    """Bug #3, pinned (2026-07-15).

    prepare_for_train's RadarCrossAttention branch did `self.annos = json.load(...)`,
    replacing parse()'s frame-oriented annotations on the CACHED instance -- so one
    /predict made /feature raise "list indices must be integers or slices, not str"
    process-wide. On a copy that rebind cannot escape.
    """
    factory = _fresh_factory()
    shared = factory.get_shared_instance("CARRADA", 9003)
    shared.annos = {"seq": {"000066": {"obj": {"range_angle": {"label": 3}}}}}

    request_copy = factory.get_instance("CARRADA", 9003)
    request_copy.annos = {"seq": {"000066": {"cls": [3.0], "idx": [1.0]}}}  # the rebind

    assert shared.annos["seq"]["000066"] == {"obj": {"range_angle": {"label": 3}}}


def test_two_copies_dont_interfere():
    """The concurrent failure mode: two overlapping /predict calls for different
    models on one file would interleave inside prepare_for_train on the shared object.
    """
    factory = _fresh_factory()
    a = factory.get_instance("CARRADA", 9004)
    b = factory.get_instance("CARRADA", 9004)

    a.model_type = "MVRECORD"
    b.model_type = "RadarCrossAttention"

    assert a.model_type == "MVRECORD"
    assert b.model_type == "RadarCrossAttention"


def test_parse_runs_once_per_file_id():
    factory = _fresh_factory()
    calls = []

    class FakeParsed:
        def parse(self, *args):
            calls.append(args)

    factory.instance_dict[9005] = FakeParsed()
    factory.get_shared_instance("CRUW", 9005, parse_args=("a", "b"))
    factory.get_shared_instance("CRUW", 9005, parse_args=("a", "b"))
    factory.get_shared_instance("CRUW", 9005, parse_args=("a", "b"))

    assert calls == [("a", "b")], f"parse ran {len(calls)} times, expected once"


def test_shared_instance_creation_is_thread_safe():
    """With handlers in a threadpool, _shared() races: two threads can both
    see the file_id missing, both construct, and hand different objects to different
    callers -- one of which is then not the cached instance anyone else sees.
    """
    import threading
    import time
    import detection_vis_backend.datasets.dataset as ds_mod

    constructions = []

    class FakeSlowCtor:
        def __init__(self):
            constructions.append(1)
            time.sleep(0.05)  # hold the race window open

    ds_mod.FakeSlowCtor = FakeSlowCtor
    try:
        factory = _fresh_factory()
        barrier = threading.Barrier(8)
        results = []

        def grab():
            barrier.wait()
            results.append(factory.get_shared_instance("FakeSlowCtor", 9100))

        threads = [threading.Thread(target=grab) for _ in range(8)]
        for t in threads: t.start()
        for t in threads: t.join()

        assert len(constructions) == 1, f"constructed {len(constructions)} times, expected once"
        assert all(r is results[0] for r in results), "threads got different shared instances"
    finally:
        del ds_mod.FakeSlowCtor


def test_parse_once_is_thread_safe():
    """Concurrent get_shared_instance(parse_args=...) for the same file must parse once.
    Without a lock, every thread sees _parsed unset and re-runs the 2-20s parse.
    """
    import threading
    import time

    factory = _fresh_factory()
    parse_calls = []

    class FakeParsed:
        def parse(self, *args):
            parse_calls.append(args)
            time.sleep(0.05)  # hold the race window open

    factory.instance_dict[9101] = FakeParsed()
    barrier = threading.Barrier(8)

    def grab():
        barrier.wait()
        factory.get_shared_instance("FakeParsed", 9101, parse_args=("a",))

    threads = [threading.Thread(target=grab) for _ in range(8)]
    for t in threads: t.start()
    for t in threads: t.join()

    assert len(parse_calls) == 1, f"parse ran {len(parse_calls)} times, expected once"


PARSE_DEFS = {  # module path -> number of parse() defs expected
    "detection_vis_backend/datasets/dataset.py": 8,
    "detection_vis_backend/datasets/view_of_delft.py": 1,
    "detection_vis_backend/datasets/nuscenes.py": 1,
}
PREPARE_DEFS = {
    "detection_vis_backend/datasets/dataset.py": 6,
    "detection_vis_backend/datasets/view_of_delft.py": 1,
    "detection_vis_backend/datasets/nuscenes.py": 1,
}


def _defs_calling(path, method_name, guard_name):
    """Return (total_defs, defs_whose_first_statement_calls guard_name)."""
    tree = ast.parse(open(path).read())
    total, guarded = 0, 0
    for cls in [n for n in ast.walk(tree) if isinstance(n, ast.ClassDef)]:
        for fn in [f for f in cls.body if isinstance(f, ast.FunctionDef) and f.name == method_name]:
            total += 1
            for node in ast.walk(fn):
                if (isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
                        and node.func.id == guard_name):
                    guarded += 1
                    break
    return total, guarded


@pytest.mark.parametrize("path,expected", PARSE_DEFS.items())
def test_every_parse_calls_require_shared_instance(path, expected):
    total, guarded = _defs_calling(path, "parse", "require_shared_instance")
    assert total == expected, f"{path}: expected {expected} parse defs, found {total}"
    assert guarded == total, f"{path}: {total - guarded} parse def(s) missing the guard"


@pytest.mark.parametrize("path,expected", PREPARE_DEFS.items())
def test_every_prepare_calls_require_request_scope(path, expected):
    total, guarded = _defs_calling(path, "prepare_for_train", "require_request_scope")
    assert total == expected, f"{path}: expected {expected} prepare defs, found {total}"
    assert guarded == total, f"{path}: {total - guarded} prepare def(s) missing the guard"


def test_config_writes_only_rebind():
    """The shallow-copy design is safe ONLY because prepare_for_train rebinds
    attributes (self.x = ...) rather than mutating shared containers
    (self.x[k] = ..., self.x.append(...)). A copy shares its values by reference,
    so an in-place mutation would reach through to the cached instance.

    Verified true across all 8 implementations on 2026-07-15; this keeps it true.
    """
    offenders = []
    for path in PREPARE_DEFS:
        tree = ast.parse(open(path).read())
        for cls in [n for n in ast.walk(tree) if isinstance(n, ast.ClassDef)]:
            for fn in [f for f in cls.body
                       if isinstance(f, ast.FunctionDef) and f.name == "prepare_for_train"]:
                rebound = set()
                for node in ast.walk(fn):
                    if isinstance(node, ast.Assign):
                        for t in node.targets:
                            if (isinstance(t, ast.Attribute) and isinstance(t.value, ast.Name)
                                    and t.value.id == "self"):
                                rebound.add(t.attr)
                            if isinstance(t, ast.Subscript):
                                v = t.value
                                if (isinstance(v, ast.Attribute) and isinstance(v.value, ast.Name)
                                        and v.value.id == "self" and v.attr not in rebound):
                                    offenders.append(f"{path}:{node.lineno} self.{v.attr}[...] =")
                    if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
                        m = node.func
                        if m.attr in ("append", "extend", "update", "add", "insert",
                                      "pop", "clear", "sort"):
                            if (isinstance(m.value, ast.Attribute)
                                    and isinstance(m.value.value, ast.Name)
                                    and m.value.value.id == "self"
                                    and m.value.attr not in rebound):
                                offenders.append(
                                    f"{path}:{node.lineno} self.{m.value.attr}.{m.attr}()")
    assert not offenders, (
        "prepare_for_train must only rebind attributes, never mutate a container it "
        "shares with the cached instance:\n  " + "\n  ".join(offenders))


def test_prepare_on_shared_raises():
    """The three inference bugs of 2026-07-15 all started with prepare_for_train
    writing config onto the cached instance. It must now refuse.
    """
    factory = _fresh_factory()
    shared = factory.get_shared_instance("CARRADA", 9006)
    with pytest.raises(RuntimeError, match="request copy"):
        shared.prepare_for_train([], {}, {})  # guard fires before any config is read


def test_parse_on_copy_raises():
    """Parsing a copy would silently succeed, then vanish with the copy, leaving
    /feature reading an unparsed shared instance (frame_sync == 0).
    """
    factory = _fresh_factory()
    request_copy = factory.get_instance("CARRADA", 9007)
    with pytest.raises(RuntimeError, match="shared instance"):
        request_copy.parse(9007, "/nonexistent", "x", {})  # guard fires before any I/O


CALL_SITES_NEEDING_SHARED = {
    "detection_vis_backend/app.py": 3,        # /parse, /sync_set, /sync_unset
    "detection_vis_backend/jobs/tasks.py": 1,  # _parse_datafiles
    "detection_vis_backend/train/train.py": 1,  # the CARRADA original-split branch
    # infer() parses AND configures (found by the guard during end-to-end verification;
    # the design-time call-site table had missed its direct parse() call).
    "detection_vis_backend/train/inference.py": 1,
}


@pytest.mark.parametrize("path,expected", CALL_SITES_NEEDING_SHARED.items())
def test_persisting_call_sites_use_get_shared_instance(path, expected):
    """Callers whose writes must outlive the request must ask for the shared
    instance explicitly. Everyone else keeps get_instance() and gets a copy.
    """
    tree = ast.parse(open(path).read())
    found = sum(
        1 for n in ast.walk(tree)
        if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)
        and n.func.attr == "get_shared_instance"
    )
    assert found == expected, f"{path}: expected {expected} get_shared_instance calls, found {found}"
