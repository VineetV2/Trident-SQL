"""Infrastructure shim for the base stages with LOCAL embeddings: no change to their code, no change to any computation.

The base stages' local-embedding path builds chromadb's SentenceTransformerEmbeddingFunction from many threads at once (the
vector DB build makes one per worker thread; the few-shot retriever initialises lazily without a lock). chromadb keeps
the model in a class-level dict behind an unlocked check-then-construct, and concurrent construction under
transformers 4.56.1 / sentence-transformers 5.1.0 races: "Cannot copy out of meta tensor; no data!" (baseline job
1390056: all 11 dev databases; debug 1390042: 1 of 3 few-shot items). An embedding server (api_type
"openai") never loads the model in-process, but we embed locally (offline compute nodes).

This file makes that constructor hold one process-wide lock, so the model is built once and every thread reuses
chromadb's cached instance (one GPU copy instead of one per thread). Loaded through PYTHONPATH (sitecustomize); the
patch is applied only when chromadb's module is imported, so other processes are untouched.
"""
import importlib.abc
import importlib.util
import sys
import threading

_TARGET = "chromadb.utils.embedding_functions.sentence_transformer_embedding_function"
_LOCK = threading.RLock()


def _patch(module):
    cls = getattr(module, "SentenceTransformerEmbeddingFunction", None)
    if cls is None or getattr(cls, "_thesis_locked", False):
        return
    original = cls.__init__

    def locked_init(self, *args, **kwargs):
        with _LOCK:
            original(self, *args, **kwargs)

    cls.__init__ = locked_init
    cls._thesis_locked = True
    sys.stderr.write("[thesis shim] chromadb SentenceTransformerEmbeddingFunction construction serialized\n")


class _PatchOnImport(importlib.abc.MetaPathFinder):
    def find_spec(self, name, path, target=None):
        if name != _TARGET:
            return None
        sys.meta_path.remove(self)
        try:
            spec = importlib.util.find_spec(name)
        finally:
            sys.meta_path.insert(0, self)
        if spec is None or spec.loader is None:
            return spec
        exec_module = spec.loader.exec_module

        def exec_and_patch(module):
            exec_module(module)
            _patch(module)

        spec.loader.exec_module = exec_and_patch
        return spec


sys.meta_path.insert(0, _PatchOnImport())
