"""Where this machine keeps the datasets, the trained models and temporary results.

Each root is an environment variable: DATASET_ROOTDIR, MODEL_ROOTDIR, TMP_ROOTDIR. A
variable already set in the environment wins; otherwise it is read from the .env file at
the repository root. That file is not committed (it holds one machine's paths), so the
paths are written in exactly one place. A root set in neither raises an error naming the
variable, instead of the code guessing a location.

Standard library only: scripts that run in their own uv environment, without the
project's packages, import this module too.
"""
import os
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
ENV_FILE = REPO_ROOT / ".env"


def read_env_file(path=None) -> dict:
    """KEY=VALUE lines of a .env file; blank lines, comments and an `export ` prefix are
    allowed, and one pair of surrounding quotes is removed. A missing file reads as {}."""
    path = Path(path or ENV_FILE)
    if not path.is_file():
        return {}
    values = {}
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        if line.startswith("export "):
            line = line[len("export "):]
        key, value = (part.strip() for part in line.split("=", 1))
        if len(value) >= 2 and value[0] == value[-1] and value[0] in "'\"":
            value = value[1:-1]
        values[key] = value
    return values


def load_env(path=None) -> None:
    """Put the .env values into os.environ, without overriding variables already set."""
    for key, value in read_env_file(path).items():
        os.environ.setdefault(key, value)


def _root(name: str) -> str:
    load_env(ENV_FILE)
    value = os.environ.get(name)
    if not value:
        raise RuntimeError(f"{name} is not set: add it to {ENV_FILE} or export it in the environment")
    return value


def dataset_root() -> str:
    return _root("DATASET_ROOTDIR")


def model_root() -> str:
    return _root("MODEL_ROOTDIR")


def tmp_root() -> str:
    return _root("TMP_ROOTDIR")


def dataset_path(*parts: str) -> str:
    """A path under the dataset root. An absolute first part is returned as it is, so
    paths stored before they were made relative to the root keep working."""
    return os.path.join(dataset_root(), *parts)


def relative_to_dataset_root(path: str) -> str:
    """path without the dataset root in front; a path outside the root is returned as it is."""
    root = dataset_root().rstrip("/") + "/"
    return path[len(root):] if path.startswith(root) else path
