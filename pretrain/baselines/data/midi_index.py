"""MIDI discovery + CSV filtering, shared by every arm behind ``baselines/train.py``.

This module is genuinely model-agnostic (file discovery and a CSV keep/drop
filter), unlike the per-codec cache builders, so it is safe to author here
without guessing at any other arm's tokenization contract. See
``baselines/README.md`` -> "Expected data layout" for the exact ``data.filter``
schema this implements.
"""

from __future__ import annotations

import csv
import os
from dataclasses import dataclass, field
from typing import List, Optional, Sequence

_MIDI_EXTS = (".mid", ".midi")
_FALSY = {"", "0", "false", "no", "nan", "none"}


@dataclass
class FilterConfig:
    """``data.filter`` from a recipe YAML, plus ``--filter-csv`` as ``path``.

    ``path=None`` means "no CSV filter" -- every discovered MIDI file is kept.
    """

    path: Optional[str] = None
    id_column: Optional[str] = None
    match: str = "stem"           # id | stem | basename | relpath | relpath_stem
    keep_columns: List[str] = field(default_factory=list)
    drop_columns: List[str] = field(default_factory=list)
    require_listed: bool = True


def _is_true(raw: str) -> bool:
    return raw.strip().lower() not in _FALSY


def _key_for(rel_path: str, mode: str) -> str:
    """Reduce a MIDI path (relative to --midi-dir) to the id ``mode`` compares."""
    base = os.path.basename(rel_path)
    if mode == "basename":
        return base
    if mode == "relpath":
        return rel_path
    if mode == "relpath_stem":
        root, _ = os.path.splitext(rel_path)
        return root
    if mode == "id":
        # first "." -> everything before it, e.g. "100000.mxl.mid" -> "100000"
        return base.split(".", 1)[0]
    # "stem" (default): strip exactly one trailing MIDI extension
    for ext in _MIDI_EXTS:
        if base.lower().endswith(ext):
            return base[: -len(ext)]
    return os.path.splitext(base)[0]


def _discover(midi_dir: str) -> List[str]:
    """Every ``.mid``/``.midi`` file under ``midi_dir``, recursively."""
    out = []
    for root, _dirs, files in os.walk(midi_dir):
        for name in files:
            if name.lower().endswith(_MIDI_EXTS):
                out.append(os.path.relpath(os.path.join(root, name), midi_dir))
    out.sort()
    return out


def _load_csv_keys(cfg: FilterConfig) -> dict:
    """path -> {key: (keep_ok: bool, drop_ok: bool)} decided per row.

    Returns a dict from lookup key to (keep, drop) booleans; a file whose key is
    absent from the dict is "not listed" (dropped iff ``require_listed``).
    """
    with open(cfg.path, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames or []
        id_col = cfg.id_column or (fieldnames[0] if fieldnames else None)
        if id_col is None:
            raise ValueError("filter CSV {} has no header row".format(cfg.path))
        rows = list(reader)

    print("[filter] csv columns: {}".format(fieldnames), flush=True)
    print("[filter] {} data rows, id column {!r}".format(len(rows), id_col), flush=True)

    decisions = {}
    for row in rows:
        raw_id = (row.get(id_col) or "").strip()
        if not raw_id:
            continue
        keep_ok = all(_is_true(row.get(c, "")) for c in cfg.keep_columns)
        drop_ok = not any(_is_true(row.get(c, "")) for c in cfg.drop_columns)
        key = _key_for(raw_id, cfg.match)
        decisions[key] = (keep_ok, drop_ok)
    print("[filter] {} distinct lookup keys".format(len(decisions)), flush=True)
    return decisions


def select_files(midi_dir: str, cfg: FilterConfig,
                 limit: Optional[int] = None) -> List[str]:
    """Discover MIDI files under ``midi_dir``, apply ``cfg``, return ABSOLUTE paths."""
    rel_files = _discover(midi_dir)
    print("[filter] discovered {} MIDI files under {}".format(len(rel_files), midi_dir),
          flush=True)

    if cfg.path:
        decisions = _load_csv_keys(cfg)
        kept, dropped_unlisted, dropped_keep, dropped_drop = [], 0, 0, 0
        for rel in rel_files:
            key = _key_for(rel, cfg.match)
            hit = decisions.get(key)
            if hit is None:
                dropped_unlisted += 1
                if not cfg.require_listed:
                    kept.append(rel)
                continue
            keep_ok, drop_ok = hit
            if not keep_ok:
                dropped_keep += 1
                continue
            if not drop_ok:
                dropped_drop += 1
                continue
            kept.append(rel)
        print("[filter] kept {} / dropped: {} unlisted, {} keep_columns, {} drop_columns"
              .format(len(kept), dropped_unlisted, dropped_keep, dropped_drop), flush=True)
        rel_files = kept

    if limit:
        rel_files = rel_files[:limit]
    return [os.path.join(midi_dir, rel) for rel in rel_files]
