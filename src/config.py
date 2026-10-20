"""Load configs/experiment.yaml and hash it for the provenance trail.

The hash covers only the sections that change the *contents* of
intermediate_csv/ and windows/, so editing a task-level setting does not
invalidate windows that are already on disk.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG = REPO_ROOT / "configs" / "experiment.yaml"

# Sections whose values determine what gets written to disk. Anything
# outside this list is a downstream modelling choice and does not affect
# the derived artifacts.
_PROVENANCE_SECTIONS = ("config_version", "ingest", "cleaning", "windowing", "splits")


class Config(dict):
    """dict with attribute access and a stable provenance hash."""

    def __getattr__(self, name: str) -> Any:
        try:
            value = self[name]
        except KeyError as exc:
            raise AttributeError(name) from exc
        return Config(value) if isinstance(value, dict) else value

    def path(self, key: str) -> Path:
        """Resolve a configured relative path against the repository root."""
        return REPO_ROOT / self["paths"][key]

    @property
    def config_hash(self) -> str:
        payload = {k: self.get(k) for k in _PROVENANCE_SECTIONS}
        blob = json.dumps(payload, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(blob.encode()).hexdigest()[:12]


def load(path: str | Path = DEFAULT_CONFIG) -> Config:
    with open(path) as fh:
        return Config(yaml.safe_load(fh))


if __name__ == "__main__":
    cfg = load()
    print(f"config_version : {cfg.config_version}")
    print(f"config_hash    : {cfg.config_hash}")
    print(f"repo root      : {REPO_ROOT}")
    for key in cfg["paths"]:
        p = cfg.path(key)
        print(f"  {key:14s} {'OK ' if p.exists() else 'MISSING'} {p}")
