"""THESIS-PATCH (selection): gated additions to SQL selection (used by sql_selection.py).

The vote is always Cross-Writer Agreement (writer_vote_shares); every gate below unset = the default behaviour.
"""
import json
import math
import os
import threading
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from common.logger import logger


# Gates (environment variables):
#   THESIS_SELECTION_INPUT=<snapshot>     read candidates from this snapshot instead of sql_revision_config.save_path
#   THESIS_SELECTION_OUTPUT=<snapshot>    write (and resume) this snapshot instead of sql_selection_config.save_path
#   THESIS_SELECTION_WRITERS=dc,icl,...   keep only candidates whose generator label is listed (needs labels)
#   THESIS_SELECTION_PER_WRITER=N         keep only the first N candidates of each writer, generation order (needs labels)
#   THESIS_SELECTION_LOG=<jsonl>          one line per item: path taken, top-k with shares, final SQL
#   THESIS_SELECTION_TIE_KEY=order        equal shares ordered by first appearance (no execution re-timing)

_LOG_LOCK = threading.Lock()


def env_on(name: str) -> bool:
    return (os.environ.get(name) or "").strip().lower() in ("1", "on", "true")


def selection_writers() -> Optional[List[str]]:
    raw = os.environ.get("THESIS_SELECTION_WRITERS")
    if raw is None or not raw.strip():
        return None
    writers = [writer.strip() for writer in raw.split(",") if writer.strip()]
    if not writers:
        raise ValueError(f"THESIS_SELECTION_WRITERS has no writer names: {raw!r}")
    return writers


def selection_per_writer() -> Optional[int]:
    raw = os.environ.get("THESIS_SELECTION_PER_WRITER")
    if raw is None or not raw.strip():
        return None
    try:
        per_writer = int(raw.strip())
    except ValueError:
        raise ValueError(f"THESIS_SELECTION_PER_WRITER must be a positive integer, got {raw!r}")
    if per_writer < 1:
        raise ValueError(f"THESIS_SELECTION_PER_WRITER must be a positive integer, got {raw!r}")
    return per_writer


def tie_key_mode() -> str:
    """'time' (default: faster first, ties re-timed) or 'order' (first appearance, no re-timing)."""
    mode = (os.environ.get("THESIS_SELECTION_TIE_KEY") or "time").strip().lower()
    if mode not in ("time", "order"):
        raise ValueError(f"THESIS_SELECTION_TIE_KEY must be 'time' or 'order', got {mode!r}")
    return mode


def selection_candidates(data_item) -> Tuple[Optional[List[str]], Optional[List[Optional[str]]]]:
    """THESIS-PATCH (selection): the candidate list selection reads (vote, shortcut, pairwise judge and the no-valid
    fallback all derive from it) and the generator label of each kept candidate. Gates unset: returns
    sql_candidates_after_revision itself and its labels. Labels come from sql_candidate_generators, which sql_revision keeps
    aligned (same length and order). Never mutates the data item."""
    candidates = data_item.sql_candidates_after_revision
    if candidates is None:
        return candidates, None
    labels = getattr(data_item, "sql_candidate_generators", None)
    item_id = getattr(data_item, "question_id", None)
    if labels is None or len(labels) != len(candidates):
        raise ValueError(
            f"Item {item_id}: Cross-Writer Agreement needs "
            f"sql_candidate_generators aligned with sql_candidates_after_revision "
            f"(labels={None if labels is None else len(labels)}, candidates={len(candidates)}). "
            "Labels are written only while candidates are generated with THESIS_LABELS=1 (pipeline.py sets it), and a "
            "rerun does not add them to items already generated (completeness ignores labels). Rerun the generation "
            "stage from scratch: delete the run's sql_generation snapshot and every later output (sql_revision*, "
            "variants/, their checkpoints) with done_generation and the later done_* markers, then run again."
        )
    writers = selection_writers()
    per_writer = selection_per_writer()
    kept_candidates: List[str] = []
    kept_labels: List[Optional[str]] = []
    writer_counts: Counter = Counter()
    for sql_candidate, label in zip(candidates, labels):
        if writers is not None and label not in writers:
            continue
        if per_writer is not None:
            if writer_counts[label] >= per_writer:
                continue
            writer_counts[label] += 1
        kept_candidates.append(sql_candidate)
        kept_labels.append(label)
    if any(label is None for label in kept_labels):
        raise ValueError(f"Item {item_id}: Cross-Writer Agreement needs a generator label for every candidate")
    return kept_candidates, kept_labels


def writer_vote_shares(
    valid_sql_candidates: List[Tuple[str, Any]],
    valid_labels: List[Optional[str]],
) -> Dict[Any, float]:
    """THESIS-PATCH (writer vote): share of a result = number of DIFFERENT writers behind it / number of writers
    present among the valid candidates, + 1e-3 * sample share to order equal writer counts."""
    writers_by_result: Dict[Any, set] = {}
    sample_counts: Counter = Counter()
    for (_, result_hash), label in zip(valid_sql_candidates, valid_labels):
        writers_by_result.setdefault(result_hash, set()).add(label)
        sample_counts[result_hash] += 1
    writers_present = len(set(valid_labels))
    return {
        result_hash: len(writers) / writers_present + 1e-3 * sample_counts[result_hash] / len(valid_sql_candidates)
        for result_hash, writers in writers_by_result.items()
    }


def log_selection(data_item, path: str, top_k, pair_votes=None, ranking_scores=None) -> None:
    """THESIS-PATCH (selection log): THESIS_SELECTION_LOG=<jsonl> appends one line per selected item."""
    log_path = os.environ.get("THESIS_SELECTION_LOG")
    if not log_path:
        return
    record = {
        "question_id": getattr(data_item, "question_id", None),
        "path": path,
        "tie_key": tie_key_mode(),
        "writers": selection_writers(),
        "per_writer": selection_per_writer(),
        "top_k": [
            {
                "sql": candidate[0],
                "share": float(candidate[2]),
                "execution_time": None if candidate[3] is None or not math.isfinite(candidate[3]) else float(candidate[3]),
            }
            for candidate in (top_k or [])
        ],
        "pair_votes": pair_votes or [],
        "ranking_scores": [float(score) for score in ranking_scores] if ranking_scores is not None else None,
        "final_selected_sql": getattr(data_item, "final_selected_sql", None),
    }
    with _LOG_LOCK:
        Path(log_path).parent.mkdir(parents=True, exist_ok=True)
        with open(log_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


def selection_io(stage_config, default_input_save_path: str):
    """THESIS-PATCH (selection io): apply THESIS_SELECTION_INPUT / THESIS_SELECTION_OUTPUT. The output override goes
    through stage_config.save_path, so the artifact checkpoint, the resume snapshot and the final save all follow it.
    The default resume rule: an existing output snapshot is loaded instead of the input, so INPUT requires OUTPUT, and a
    sidecar <output>.thesis_input.json records the input an output belongs to (a mismatch raises)."""
    input_override = (os.environ.get("THESIS_SELECTION_INPUT") or "").strip() or None
    output_override = (os.environ.get("THESIS_SELECTION_OUTPUT") or "").strip() or None
    if input_override is None and output_override is None:
        return stage_config, default_input_save_path
    if input_override is not None and output_override is None:
        raise ValueError("THESIS_SELECTION_INPUT requires THESIS_SELECTION_OUTPUT (otherwise selection may resume from "
                         f"another run's snapshot at {stage_config.save_path})")
    input_save_path = input_override or default_input_save_path
    if output_override is not None:
        stage_config = stage_config.model_copy(update={"save_path": output_override})
    output_path = Path(stage_config.save_path)
    if output_path.resolve() == Path(input_save_path).resolve():
        raise ValueError(f"THESIS_SELECTION_OUTPUT must differ from the selection input snapshot ({input_save_path})")
    binding_path = output_path.with_name(f"{output_path.name}.thesis_input.json")
    binding = {"input_save_path": str(Path(input_save_path).resolve())}
    if binding_path.exists():
        recorded = json.loads(binding_path.read_text(encoding="utf-8"))
        if recorded.get("input_save_path") != binding["input_save_path"]:
            raise ValueError(f"Selection output {output_path} was started from {recorded.get('input_save_path')}, "
                             f"not {binding['input_save_path']}; use a new THESIS_SELECTION_OUTPUT")
    else:
        binding_path.parent.mkdir(parents=True, exist_ok=True)
        binding_path.write_text(json.dumps(binding, indent=2), encoding="utf-8")
    logger.info(f"THESIS selection io: input={input_save_path} output={stage_config.save_path}")
    return stage_config, input_save_path


def preflight(dataset, checkpoint_source: str) -> None:
    """THESIS-PATCH (selection): fail before any LLM call on a bad gate combination, missing labels, or an input
    snapshot that already carries selection results (its items would be skipped as complete)."""
    tie_key_mode()
    selection_writers()
    selection_per_writer()
    for data_item in dataset:
        if not data_item.is_stage_complete("sql_selection"):
            selection_candidates(data_item)
    if os.environ.get("THESIS_SELECTION_INPUT") and checkpoint_source == "base":
        if any(data_item.is_stage_complete("sql_selection") for data_item in dataset):
            raise ValueError("THESIS_SELECTION_INPUT already has selection results; point it at a revision-stage snapshot")
