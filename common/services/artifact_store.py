import atexit
import json
import os
import queue
import shutil
import threading
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, Iterator, Tuple

from common.dataset.artifacts import STAGE_ARTIFACT_FIELDS
from common.logger import logger


def _last_line_start(f, size: int, block: int = 1 << 16) -> int:
    """Byte offset just past the last b"\\n" in the binary file f (0 if there is none)."""
    end = size
    while end > 0:
        begin = max(0, end - block)
        f.seek(begin)
        newline = f.read(end - begin).rfind(b"\n")
        if newline != -1:
            return begin + newline + 1
        end = begin
    return 0


def _repair_unterminated_tail(path: Path) -> None:
    """Make the records file end with a newline before anything is appended to it.

    A run killed mid-write can leave a last line without its newline. If that line is a whole
    record (only the newline was lost) it is kept and the newline added. Otherwise it is a torn
    record, which the loader already skips, and it is cut off: appending after it would leave it as
    a bad line in the middle of the file, which the loader treats as corruption.
    """
    try:
        size = path.stat().st_size
    except FileNotFoundError:
        return
    if size == 0:
        return
    with open(path, "r+b") as f:
        f.seek(size - 1)
        if f.read(1) == b"\n":
            return
        start = _last_line_start(f, size)
        f.seek(start)
        try:
            json.loads(f.read().decode("utf-8"))
        except ValueError:  # JSONDecodeError or UnicodeDecodeError: a torn record
            f.truncate(start)
            logger.warning(f"Cut a partial last record ({size - start} bytes, from an interrupted write) off {path}")
        else:
            f.write(b"\n")
        f.flush()
        os.fsync(f.fileno())


class ArtifactStore:
    def __init__(self, save_path: str, stage_name: str, stage_fields: Iterable[str]):
        self._save_path = Path(save_path)
        self._stage_name = stage_name
        self._stage_fields = list(dict.fromkeys(stage_fields))
        self._root = self._save_path.with_name(f"{self._save_path.stem}.artifacts")
        self._meta_path = self._root / "meta.json"
        self._records_path = self._root / f"{stage_name}.jsonl"
        self._buffer: list[Dict[str, Any]] = []
        self._lock = threading.Lock()
        self._write_queue: queue.Queue[str | None] = queue.Queue()
        self._pending_lock = threading.Condition()
        self._pending_batches = 0
        self._writer_error: Exception | None = None
        self._closed = False
        self._writer_thread = threading.Thread(
            target=self._writer_loop,
            name=f"artifact-writer-{stage_name}",
            daemon=True,
        )
        self._writer_thread.start()
        # the writer is a daemon thread: without this, a batch queued by flush() is lost when the
        # process exits without close() (e.g. an exception after a checkpoint flush)
        atexit.register(self._close_at_exit)

    def has_checkpoint(self) -> bool:
        self.sync()
        return self._records_path.exists() and self._records_path.stat().st_size > 0

    def record_item(self, data_item: Any, extra_fields: Iterable[str] | None = None) -> None:
        self._raise_if_closed()
        entry = {
            "item_id": self._get_item_id(data_item),
            "stage_artifact": self._to_jsonable(data_item.get_stage_artifact(self._stage_name).model_dump()),
            "metrics": self._to_jsonable(data_item.get_metrics_record().model_dump()),
        }
        if extra_fields is not None:
            entry["fields"] = {
                field: self._to_jsonable(getattr(data_item, field))
                for field in dict.fromkeys(extra_fields)
                if hasattr(data_item, field)
            }
        with self._lock:
            self._buffer.append(entry)

    def flush(self) -> int:
        self._raise_if_closed()
        self._raise_writer_error()
        with self._lock:
            if not self._buffer:
                return 0
            entries = self._buffer
            self._buffer = []

        self._ensure_meta()

        payload = "".join(f"{json.dumps(entry, ensure_ascii=False)}\n" for entry in entries)
        with self._pending_lock:
            self._pending_batches += 1
        self._write_queue.put(payload)
        logger.info(f"[{self._stage_name}] Queued {len(entries)} checkpoint records for {self._records_path}")
        return len(entries)

    def apply_to_dataset(self, dataset: Any) -> int:
        self.sync()
        if not self.has_checkpoint():
            return 0

        latest_entries: Dict[str, Dict[str, Any]] = {}
        for entry in self._read_records():
            latest_entries[entry["item_id"]] = entry

        item_map = {self._get_item_id(item): item for item in dataset}
        applied = 0
        for item_id, entry in latest_entries.items():
            item = item_map.get(item_id)
            if item is None:
                logger.warning(f"[{self._stage_name}] Checkpoint item {item_id} not found in base dataset")
                continue
            if "stage_artifact" in entry:
                item.apply_stage_artifact(self._stage_name, entry["stage_artifact"])
            if "metrics" in entry:
                item.apply_metrics_record(entry["metrics"])
            fields = entry.get("fields", {})
            for field, value in fields.items():
                setattr(item, field, value)
            applied += 1

        logger.info(f"[{self._stage_name}] Restored {applied} items from incremental checkpoint")
        return applied

    def _read_records(self) -> Iterator[Dict[str, Any]]:
        """Records in file order. A last line that does not parse is a record torn by a killed run:
        it is skipped with a warning. A line that does not parse anywhere else is corruption: raise."""
        bad = None  # (line number, error) of an unparseable line, fatal once a later record shows up
        with open(self._records_path, "rb") as f:
            for line_number, raw in enumerate(f, start=1):
                if bad is not None and raw.strip():
                    raise ValueError(
                        f"[{self._stage_name}] Corrupt checkpoint {self._records_path}: "
                        f"line {bad[0]} is not a valid record ({bad[1]})"
                    ) from bad[1]
                try:
                    line = raw.decode("utf-8")
                    if not line.strip():
                        continue
                    entry = json.loads(line)
                except ValueError as exc:  # JSONDecodeError or UnicodeDecodeError
                    bad = (line_number, exc)
                    continue
                yield entry
        if bad is not None:
            logger.warning(
                f"[{self._stage_name}] Skipping partial last record at line {bad[0]} of "
                f"{self._records_path} (interrupted write): {bad[1]}"
            )

    def cleanup(self) -> None:
        self.close()
        if self._root.exists():
            shutil.rmtree(self._root)

    def sync(self) -> None:
        self._raise_writer_error()
        with self._pending_lock:
            while self._pending_batches > 0 and self._writer_error is None:
                self._pending_lock.wait()
        self._raise_writer_error()

    def close(self) -> None:
        if self._closed:
            return
        self.sync()
        self._write_queue.put(None)
        self._writer_thread.join()
        self._closed = True
        atexit.unregister(self._close_at_exit)
        self._raise_writer_error()

    def _close_at_exit(self) -> None:
        # daemon threads still run during atexit, so the queued batches can be drained; skip if the
        # writer is gone (e.g. in a forked child), where waiting for it would hang
        if not self._closed and self._writer_thread.is_alive():
            self.close()

    @staticmethod
    def _get_item_id(data_item: Any) -> str:
        if hasattr(data_item, "get_item_id") and callable(data_item.get_item_id):
            return str(data_item.get_item_id())
        return str(getattr(data_item, "question_id"))

    @classmethod
    def _to_jsonable(cls, value: Any) -> Any:
        if hasattr(value, "model_dump") and callable(value.model_dump):
            value = value.model_dump()

        if isinstance(value, Path):
            return str(value)
        if isinstance(value, dict):
            return {str(k): cls._to_jsonable(v) for k, v in value.items()}
        if isinstance(value, (list, tuple)):
            return [cls._to_jsonable(v) for v in value]
        if isinstance(value, set):
            return [cls._to_jsonable(v) for v in sorted(value, key=str)]
        return value

    def _ensure_meta(self) -> None:
        self._root.mkdir(parents=True, exist_ok=True)
        if not self._meta_path.exists():
            self._meta_path.write_text(
                json.dumps(
                    {
                        "format_version": 2,
                        "stage_name": self._stage_name,
                        "save_path": str(self._save_path),
                        "stage_fields": self._stage_fields,
                    },
                    ensure_ascii=False,
                    indent=2,
                ),
                encoding="utf-8",
            )

    def _writer_loop(self) -> None:
        while True:
            payload = self._write_queue.get()
            if payload is None:
                self._write_queue.task_done()
                break
            try:
                if self._writer_error is None:
                    _repair_unterminated_tail(self._records_path)
                    with open(self._records_path, "a", encoding="utf-8", buffering=1024 * 1024) as f:
                        f.write(payload)
                        f.flush()
                        os.fsync(f.fileno())
            except Exception as exc:
                self._writer_error = exc
            finally:
                with self._pending_lock:
                    self._pending_batches = max(0, self._pending_batches - 1)
                    if self._pending_batches == 0 or self._writer_error is not None:
                        self._pending_lock.notify_all()
                self._write_queue.task_done()

    def _raise_writer_error(self) -> None:
        if self._writer_error is not None:
            raise RuntimeError(
                f"ArtifactStore writer failed for stage {self._stage_name}: {self._writer_error}"
            ) from self._writer_error

    def _raise_if_closed(self) -> None:
        if self._closed:
            raise RuntimeError(f"ArtifactStore for stage {self._stage_name} is already closed")


def load_stage_dataset(
    *,
    load_dataset_fn: Callable[[str], Any],
    current_save_path: str,
    fallback_load_path: str,
    artifact_store: ArtifactStore,
    stage_name: str,
) -> Tuple[Any, str]:
    current_path = Path(current_save_path)
    if current_path.exists():
        logger.info(f"Loading {stage_name} snapshot from {current_path}")
        dataset = load_dataset_fn(str(current_path))
        checkpoint_source = "snapshot"
    else:
        fallback_path = Path(fallback_load_path)
        if not fallback_path.exists():
            raise FileNotFoundError(
                f"{stage_name} requires an input snapshot at {fallback_path}. "
                "Run the previous pipeline stage or preprocess the dataset first."
            )
        logger.info(f"Loading {stage_name} base dataset from {fallback_path}")
        dataset = load_dataset_fn(str(fallback_path))
        checkpoint_source = "base"

    if artifact_store.has_checkpoint():
        artifact_store.apply_to_dataset(dataset)
        checkpoint_source = f"{checkpoint_source}+artifact_checkpoint"

    return dataset, checkpoint_source
