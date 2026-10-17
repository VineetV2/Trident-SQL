"""Read and write the base stages' dataset snapshots as plain JSON.

Layout (common/dataset/utils.py save_dataset / load_dataset):
  <save_dir>/<stage>.snapshot                    manifest {format: "structured_dataset_snapshot", version: 1,
                                                 snapshot_root: "<stage>.snapshot.data", dataset_config: {...},
                                                 num_items, dataset/item class names}
  <save_dir>/<stage>.snapshot.data/items.jsonl   one {input, pipeline_artifacts} per item; pipeline_artifacts =
                                                 {value_retrieval, schema_linking, sql_generation, sql_revision,
                                                 sql_selection, metrics}
  <save_dir>/<stage>.artifacts/                  incremental checkpoint of a stage that has not finished
                                                 (common/services/artifact_store.py: Path(save_path).stem + ".artifacts",
                                                 so "sql_revision.artifacts", NOT "sql_revision.snapshot.artifacts");
                                                 removed when the stage materializes its snapshot
A stage loads its own snapshot when it exists (and then skips finished items), else the previous stage's, and applies
its checkpoint on top. Nothing here imports the base stages (no pydantic): our thesis env reads the files directly.
API kept small and stable (stages 4, 6 and 8 import it).
"""
from __future__ import annotations

import json
import os
import shutil
import uuid
from typing import Any, Iterator, List, Optional

SNAPSHOT_FORMAT = "structured_dataset_snapshot"
STAGES = ("value_retrieval", "schema_linking", "sql_generation", "sql_revision", "sql_selection")


# ---------------------------------------------------------------- paths
def stage_snapshot(run_dir: str, stage: str) -> str:
    """<run_dir>/<stage>.snapshot (the default save_path for a stage inside the run folder)."""
    return os.path.join(os.path.abspath(run_dir), f"{stage}.snapshot")


def artifacts_dir(snapshot_path: str) -> str:
    """The stage's incremental checkpoint dir: <dir>/<stem>.artifacts (ArtifactStore._root)."""
    base = os.path.basename(snapshot_path)
    stem = base[: -len(".snapshot")] if base.endswith(".snapshot") else os.path.splitext(base)[0]
    return os.path.join(os.path.dirname(os.path.abspath(snapshot_path)), f"{stem}.artifacts")


def read_manifest(snapshot_path: str) -> dict:
    with open(snapshot_path, encoding="utf-8") as f:
        manifest = json.load(f)
    if manifest.get("format") != SNAPSHOT_FORMAT:
        raise ValueError(f"{snapshot_path}: not a structured stage snapshot (format={manifest.get('format')!r})")
    return manifest


def items_file(snapshot_path: str, manifest: Optional[dict] = None) -> str:
    """<dir>/<snapshot_root>/items.jsonl; snapshot_root defaults to <name>.data."""
    if manifest is None and os.path.exists(snapshot_path):
        manifest = read_manifest(snapshot_path)
    root = (manifest or {}).get("snapshot_root") or os.path.basename(snapshot_path) + ".data"
    return os.path.join(os.path.dirname(os.path.abspath(snapshot_path)), root, "items.jsonl")


def has_checkpoint(snapshot_path: str) -> bool:
    """True when the stage left a non-empty incremental checkpoint (it started but has not materialized)."""
    path = os.path.join(artifacts_dir(snapshot_path), f"{_stage_name(snapshot_path)}.jsonl")
    return os.path.isfile(path) and os.path.getsize(path) > 0


def stage_started(run_dir: str, stage: str) -> bool:
    """The stage's snapshot manifest exists or it left a checkpoint (either one makes the stage resume from it)."""
    snap = stage_snapshot(run_dir, stage)
    return os.path.exists(snap) or has_checkpoint(snap)


def _stage_name(snapshot_path: str) -> str:
    base = os.path.basename(snapshot_path)
    return base[: -len(".snapshot")] if base.endswith(".snapshot") else os.path.splitext(base)[0]


# ---------------------------------------------------------------- items
def iter_items(snapshot_path: str) -> Iterator[dict]:
    with open(items_file(snapshot_path), encoding="utf-8") as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


def read_items(snapshot_path: str) -> List[dict]:
    """All items; raises when the count disagrees with the manifest's num_items (a truncated write)."""
    manifest = read_manifest(snapshot_path)
    items = list(iter_items(snapshot_path))
    if manifest.get("num_items") is not None and manifest["num_items"] != len(items):
        raise ValueError(f"{snapshot_path}: manifest says {manifest['num_items']} items, items.jsonl has {len(items)}")
    return items


def qid(item: dict) -> int:
    return int(item["input"]["question_id"])


def artifact(item: dict, stage: str, name: Optional[str] = None) -> Any:
    """pipeline_artifacts[stage] (a dict, {} when absent), or one field of it when `name` is given."""
    art = (item.get("pipeline_artifacts") or {}).get(stage) or {}
    return art if name is None else art.get(name)


def with_artifact(item: dict, stage: str, fields: dict) -> dict:
    """A copy of `item` whose pipeline_artifacts[stage] carries `fields` on top of the old ones (item unchanged)."""
    arts = dict(item.get("pipeline_artifacts") or {})
    arts[stage] = {**(arts.get(stage) or {}), **fields}
    return {**item, "pipeline_artifacts": arts}


# ---------------------------------------------------------------- writing
def atomic_lines(path: str, rows: List[dict]) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    tmp = f"{path}.{uuid.uuid4().hex}.tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    os.replace(tmp, path)


def atomic_json(path: str, obj: Any, indent: Optional[int] = 1) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    tmp = f"{path}.{uuid.uuid4().hex}.tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=indent, ensure_ascii=False)
    os.replace(tmp, path)


def write_snapshot(out_path: str, manifest: dict, items: List[dict]) -> None:
    """A NEW snapshot: items.jsonl first, manifest last (both atomic), so a manifest on disk means complete items."""
    out_path = os.path.abspath(out_path)
    root = os.path.basename(out_path) + ".data"
    atomic_lines(os.path.join(os.path.dirname(out_path), root, "items.jsonl"), items)
    atomic_json(out_path, {**manifest, "snapshot_root": root, "num_items": len(items)}, indent=2)


def rewrite_items(snapshot_path: str, items: List[dict]) -> None:
    """Replace an existing snapshot's items in place (atomic). The item set must stay the same (same question ids,
    same order), so the manifest stays valid and the next stage sees every item."""
    old = [qid(it) for it in iter_items(snapshot_path)]
    new = [qid(it) for it in items]
    if old != new:
        raise ValueError(f"{snapshot_path}: rewrite would change the item set ({len(old)} -> {len(new)} items "
                         "or a different order); refusing")
    atomic_lines(items_file(snapshot_path), items)


def backup_items(snapshot_path: str, suffix: str) -> Optional[str]:
    """One-time copy items.jsonl -> items.jsonl<suffix>; an existing backup is never overwritten. Returns its path."""
    src = items_file(snapshot_path)
    dst = src + suffix
    if os.path.exists(dst):
        return dst
    tmp = f"{dst}.{uuid.uuid4().hex}.tmp"
    shutil.copyfile(src, tmp)
    os.replace(tmp, dst)
    return dst


def move_aside(run_dir: str, stage: str, tag: str) -> List[str]:
    """Rename the stage's snapshot manifest, .data dir and .artifacts checkpoint to <name>.<tag> (nothing deleted),
    so the stage reruns from the previous one. Returns the new paths."""
    snap = stage_snapshot(run_dir, stage)
    moved = []
    for path in (snap, snap + ".data", artifacts_dir(snap)):
        if os.path.exists(path):
            dst = f"{path}.{tag}"
            os.replace(path, dst)
            moved.append(dst)
    return moved
