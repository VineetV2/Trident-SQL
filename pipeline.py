"""Our text-to-SQL pipeline on BIRD dev, end to end, in ONE Python environment.

Value finding, similar solved examples, column finding, two of the SQL writers (divide-and-conquer and few-shot),
repair and the judge are the base stages (0-5 and 7). Plan-and-Fragment generation, the query tree and Cross-Writer
Agreement are ours. Stages and settings are those of the reported dev run: every model call at high reasoning effort,
temperature 0.7 (query tree 1.0). The model and the base stages' settings are in configs/pipeline_config.toml; the
on/off switches of our own parts are set below (build_env, stage_query_tree, stage_selection).

Stages, in order (each leaves <run>/done_<stage>; a rerun skips finished stages, and the base stages also resume per
question inside). The folder holds the stage's code:
  dataset          stage0_dataset            read BIRD dev into the run
  value_index      stage1_value_finding      embed every text value of each database (skipped when stored)
  value_retrieval  stage1_value_finding      find the stored values that match each question (up to 3 passes)
  fewshot_index    stage2_similar_examples   index of solved training questions (reused when it exists)
  fewshot_prep     stage2_similar_examples   similar solved examples per question (up to 3 passes)
  column_finding   stage3_column_finding     the columns the SQL needs (up to 3 passes)
  generation       stage4_generation         all three writers at the same time: divide-and-conquer 4, few-shot 4
                                             and Plan-and-Fragment 4
  repair           stage5_repair             8 checks; a failed check gets the SQL rewritten (up to 3 passes)
  query_tree       stage6_query_tree         2 rules written from training-set mistakes
  selection        stage7_selection          Cross-Writer Agreement, the judge for close calls
  score            stage8_scoring            BIRD execution accuracy over the 1,534 questions

Each stage runs as `python -m <stage module>` in a subprocess of THIS interpreter, from this folder, so every stage
frees its GPU memory when it ends. Needs a running vLLM server for gpt-oss-120b (run_all.sh starts it). Ctrl-C or
SIGTERM stops the running stage's processes too and exits with 130 / 143; the same command resumes the run.

  python pipeline.py --run-dir <dir> --data-root <bird> --db-root <dev_databases> --prep <prep> --vector-store <dir>
      --fewshot-index <dir> --embed-model <dir> --base-url http://127.0.0.1:<port> [--max-samples N] [--preflight]
"""
from __future__ import annotations

import argparse
import datetime
import json
import os
import re
import shutil
import signal
import subprocess
import sys
import time
import tomllib

CODE = os.path.dirname(os.path.abspath(__file__))
CONFIG = os.path.join(CODE, "configs", "pipeline_config.toml")                        # model + base stages' settings
PYSHIM = os.path.join(CODE, "common", "pyshim")                                       # embedding-model load lock
TREE_RULES = os.path.join(CODE, "stage6_query_tree", "rules", "tree_rules_r8_supnest.json")
EMBED_DEVICE = "cuda:0"   # the stages see only the embedding GPU (serve_and_run_local.sh), so it is always cuda:0
STAGES = ("dataset", "value_index", "value_retrieval", "fewshot_index", "fewshot_prep", "column_finding",
          "generation", "repair", "query_tree", "selection", "score")
REPORTED = "final_qtree_wv"   # the reported variant: query tree + Cross-Writer Agreement, the default tie key


class StageFailed(RuntimeError):
    pass


def log(msg: str) -> None:
    print(f"##### {msg} {datetime.datetime.now().isoformat(timespec='seconds')}", flush=True)


# ------------------------------------------------------------------ config
# The model and the base stages' settings come from configs/pipeline_config.toml. At the start of a run it is turned
# into the files the stages read: the base stages' config (port_config.toml) and one model-settings file each for
# Plan-and-Fragment and the query tree.
# The written files depend only on the config and the command line (no time stamps), so a resumed run writes the same
# bytes; the query tree's resume check hashes its settings file.
def load_config(path: str = CONFIG) -> dict:
    with open(path, "rb") as f:
        return tomllib.load(f)


def _fill(table: dict, section: str, values: dict) -> dict:
    """`table` with `values` added; a key the config already sets is an error (pipeline.py owns it)."""
    clash = sorted(set(values) & set(table))
    if clash:
        raise SystemExit(f"!! {CONFIG}: [base.{section}] must not set {', '.join(clash)} (pipeline.py fills it in)")
    return {**table, **values}


def base_config(cfg: dict, a: argparse.Namespace, run_dir: str) -> dict:
    """The base stages' config for this run: the [base] tables with the model settings and this machine's paths."""
    base, model, smoke = cfg["base"], cfg["model"], cfg["smoke"]
    profile = base["llm_profiles"]["gpt_oss_120b"]
    if a.max_samples:   # smoke test only: first N questions, shorter outputs
        profile = {**profile, "max_model_len": smoke["base_max_model_len"]}
    profile = _fill(profile, "llm_profiles.gpt_oss_120b", {
        "model": model["name"], "base_url": a.base_url.rstrip("/") + "/v1",
        "max_tokens": smoke["max_output_tokens"] if a.max_samples else model["max_output_tokens"],
        "temperature": cfg["temperature"]["base_stages"], "reasoning_effort": model["reasoning_effort"]})
    dataset = _fill(base["dataset"], "dataset", {"root_path": a.data_root})
    if a.max_samples:
        dataset = {**dataset, "max_samples": a.max_samples}
    return {**base,
            "run": _fill(base["run"], "run", {"save_root": os.path.dirname(run_dir),
                                              "exp_name": os.path.basename(run_dir)}),
            "llm_profiles": {**base["llm_profiles"], "gpt_oss_120b": profile},
            "embedding": _fill(base["embedding"], "embedding", {"embedding_model_name_or_path": a.embed_model,
                                                                "embedding_device": EMBED_DEVICE}),
            "dataset": dataset,
            "vector_database": _fill(base["vector_database"], "vector_database", {"store_root_path": a.vector_store}),
            "few_shot_index": _fill(base["few_shot_index"], "few_shot_index", {
                "similarity_device": EMBED_DEVICE, "save_path": a.fewshot_index,
                "mask_cache_path": a.fewshot_index + ".mask_cache.jsonl"}),
            "value_retrieval": _fill(base["value_retrieval"], "value_retrieval", {"local_index_device": EMBED_DEVICE})}


def _toml_value(v) -> str:
    if isinstance(v, bool):
        return "true" if v else "false"
    if isinstance(v, (int, float)):
        return repr(v)
    if isinstance(v, str):
        return json.dumps(v)   # a JSON string is a valid TOML basic string
    raise SystemExit(f"!! config: cannot write {v!r} ({type(v).__name__}) to TOML")


def to_toml(data: dict, prefix: str = "") -> str:
    """TOML text of nested tables of plain values (all the base stages' config has)."""
    keys = [k for k, v in data.items() if not isinstance(v, dict)]
    out = [f"[{prefix}]"] if prefix and keys else []
    out += [f"{k} = {_toml_value(data[k])}" for k in keys]
    for k, v in data.items():
        if isinstance(v, dict):
            sub = to_toml(v, f"{prefix}.{k}" if prefix else k)
            out += ["", sub] if out else [sub]
    return "\n".join(out)


def model_settings_yaml(cfg: dict, stage: str, max_samples: int | None) -> str:
    """Model settings for one of our stages, in the format common/llm_client/settings.py reads."""
    model = cfg["model"]
    rows = [("model", model["name"]), ("hf_revision", model["hf_revision"]),
            ("reasoning_effort", model["reasoning_effort"]), ("temperature", cfg["temperature"][stage]),
            ("top_p", model["top_p"]),
            ("max_output_tokens", cfg["smoke"]["max_output_tokens"] if max_samples else model["max_output_tokens"]),
            ("request_timeout_s", model["request_timeout_s"])]
    head = f"# Written by pipeline.py from configs/pipeline_config.toml ([model], [temperature] {stage}). Do not edit.\n"
    return head + "".join(f"{k}: {json.dumps(v) if isinstance(v, str) else repr(v)}\n" for k, v in rows)


def write_settings(cfg: dict, a: argparse.Namespace, run_dir: str) -> dict:
    """Write the files the stages read into the run folder; their paths by name."""
    paths = {"base": os.path.join(run_dir, "port_config.toml"),
             "plan_and_fragment": os.path.join(run_dir, "model_plan_and_fragment.yaml"),
             "query_tree": os.path.join(run_dir, "model_query_tree.yaml")}
    head = "# Base stages' config for this run, written by pipeline.py from configs/pipeline_config.toml. Do not edit.\n\n"
    open(paths["base"], "w").write(head + to_toml(base_config(cfg, a, run_dir)) + "\n")
    for stage in ("plan_and_fragment", "query_tree"):
        open(paths[stage], "w").write(model_settings_yaml(cfg, stage, a.max_samples))
    return paths


# ------------------------------------------------------------------ stopping on a signal
# Every stage subprocess starts in its own session (process group id = its pid), so the terminal's Ctrl-C or hangup
# reaches only this process. On SIGINT, SIGTERM or SIGHUP (SIGHUP only when not ignored, e.g. not under nohup) the
# handler stops each live stage's whole process group (SIGTERM, up to CHILD_STOP_WAIT_S, then SIGKILL) and raises
# KeyboardInterrupt, so stage_generation's writer cleanup still runs and main() returns 128 + the signal number.
CHILD_STOP_WAIT_S = 10
STOP_SIGNALS = (signal.SIGINT, signal.SIGTERM, signal.SIGHUP)
_LIVE: set[subprocess.Popen] = set()                          # the stage subprocesses running now
_STOP = {"starting": False, "pending": None, "signal": None}  # see _start and _on_signal


def _start(cmd: list[str], **kwargs) -> subprocess.Popen:
    """Popen in a new session, tracked in _LIVE; a signal that arrives while it starts is handled once it is tracked."""
    _STOP["starting"] = True
    try:
        proc = subprocess.Popen(cmd, start_new_session=True, **kwargs)
        _LIVE.add(proc)
    finally:
        _STOP["starting"] = False
    pending, _STOP["pending"] = _STOP["pending"], None
    if pending is not None:
        _on_signal(pending, None)
    return proc


def _signal_group(proc: subprocess.Popen, sig: int) -> None:
    try:
        os.killpg(proc.pid, sig)
    except (ProcessLookupError, PermissionError):   # the group is gone already
        pass


def stop_children(procs=None) -> None:
    """SIGTERM to the process group of each stage subprocess (default: every live one), wait up to CHILD_STOP_WAIT_S,
    then SIGKILL."""
    procs = list(_LIVE if procs is None else procs)
    for proc in procs:
        _signal_group(proc, signal.SIGTERM)
    deadline = time.monotonic() + CHILD_STOP_WAIT_S
    for proc in procs:
        try:
            proc.wait(timeout=max(0.0, deadline - time.monotonic()))
        except subprocess.TimeoutExpired:
            pass
    for proc in procs:
        _signal_group(proc, signal.SIGKILL)   # a leader that outlived SIGTERM, or group members it left behind
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            try:
                log(f"!! stage process {proc.pid} still alive after SIGKILL")
            except OSError:   # the console is gone
                pass
        _LIVE.discard(proc)


def _on_signal(signum: int, frame) -> None:
    if _STOP["starting"]:   # a child is being started: _start handles the signal once the child is tracked
        _STOP["pending"] = signum
        return
    for sig in STOP_SIGNALS:   # a second signal (Ctrl-C, then the serve script's TERM) must not cut the cleanup short
        signal.signal(sig, signal.SIG_IGN)
    _STOP["signal"] = signum
    stop_children()
    raise KeyboardInterrupt


def install_signal_handlers() -> None:
    for sig in STOP_SIGNALS:
        if sig == signal.SIGHUP and signal.getsignal(sig) == signal.SIG_IGN:   # nohup: keep running after hangup
            continue
        signal.signal(sig, _on_signal)


# ------------------------------------------------------------------ running stages
def run(cmd: list[str], cwd: str, env: dict, logfile: str) -> tuple[int, str]:
    """Run one command, stream its output to the console and the stage log; (exit code, output)."""
    with open(logfile, "a") as fh:
        fh.write(f"\n$ {' '.join(cmd)}  (cwd {cwd})\n")
        proc = _start(cmd, cwd=cwd, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        try:
            out = []
            for line in proc.stdout:
                sys.stdout.write(line)
                fh.write(line)
                out.append(line)
            return proc.wait(), "".join(out)
        except BaseException:   # e.g. the console pipe broke: never leave the stage running on its own
            if proc.returncode is None:
                stop_children([proc])
            raise
        finally:
            _LIVE.discard(proc)


def run_base(module: str, ctx: dict) -> None:
    rc, _ = run([sys.executable, "-m", module], CODE, ctx["env"], ctx["log"])
    if rc:
        raise StageFailed(f"{module} exited {rc}")


def run_base_retry(module: str, ctx: dict, ok) -> None:
    """The base stages' own resume used as a retry: a pass redoes only the questions without a result."""
    for attempt in (1, 2, 3):
        rc, out = run([sys.executable, "-m", module], CODE, ctx["env"], ctx["log"])
        if ok(rc, out):
            log(f"{module} pass {attempt}: ok")
            return
        log(f"{module} pass {attempt}: questions left")
    raise StageFailed(f"{module} still failing after 3 passes")


def ours(args: list[str], ctx: dict, env: dict | None = None) -> None:
    rc, _ = run([sys.executable, *args], CODE, env or ctx["env"], ctx["log"])
    if rc:
        raise StageFailed(f"{' '.join(args[:2])} exited {rc}")


def stage_generation(ctx: dict) -> None:
    """One generation step with all three writers working at the same time: the divide-and-conquer and few-shot
    writers (4 SQL each) and our Plan-and-Fragment writer (4 SQL) on the same linked questions. When both
    are done, Plan-and-Fragment's SQL are appended to each question's list (dc 4, icl 4, literal 4, writer labels),
    as in the reported run."""
    a = ctx["args"]
    ours_args = ["-m", "stage4_generation.plan_and_fragment.run_plan_and_fragment", "--run-dir", ctx["run"],
                 "--prep-root", a.prep, "--db-root", a.db_root, "--split", "dev",
                 "--writers", "literal", "--n", "4", "--settings", ctx["writer_settings"], "--workers", "16",
                 "--base-url", a.base_url]
    ours_log = os.path.join(ctx["run"], "plan_and_fragment.log")
    with open(ours_log, "a") as fh:   # ours in the background, its output in its own log
        writer = _start([sys.executable, *ours_args, "--prewrite"], cwd=CODE, env=ctx["env"],
                        stdout=fh, stderr=subprocess.STDOUT, text=True)
        try:
            run_base("stage4_generation.run_sql_generation", ctx)
        except BaseException:
            writer.terminate()
            writer.wait()
            _LIVE.discard(writer)
            raise
        rc = writer.wait()
        _LIVE.discard(writer)
    if rc:
        raise StageFailed(f"Plan-and-Fragment exited {rc} (see {ours_log})")
    ours(ours_args, ctx)   # merge: resumes from the finished per-question files, so no new model calls


def stage_query_tree(ctx: dict) -> None:
    run_dir = ctx["run"]
    env = dict(ctx["env"], THESIS_MODEL_SETTINGS=ctx["tree_settings"], THESIS_TREE_RULES_FACTS="off",
               THESIS_TREE_RULES_LOOP="on", THESIS_TREE_RULES_SAMPLES="1", THESIS_TREE_RULES_FILE=TREE_RULES,
               THESIS_TREE_RULES_CACHE=os.path.join(run_dir, "tree_rules", "llm_cache"),
               THESIS_TREE_RULES_RUN_ID=os.path.basename(run_dir),
               THESIS_TREE_RULES_LOG=os.path.join(run_dir, "tree_rules", "tree_rules_log.jsonl"))
    ours(["-m", "stage6_query_tree.run_query_tree", "--run-dir", run_dir, "--output", "sql_revision_rules.snapshot",
          "--fixes", "rules", "--rules-module", "stage6_query_tree.qtree.rule_fix:apply_rules",
          "--db-root", ctx["args"].db_root],
         ctx, env)
    os.makedirs(os.path.join(run_dir, "tree_rules"), exist_ok=True)
    shutil.copy(TREE_RULES, os.path.join(run_dir, "tree_rules", "rules_file_used.json"))


def stage_selection(ctx: dict) -> None:
    run_dir, vdir = ctx["run"], os.path.join(ctx["run"], "variants")
    os.makedirs(vdir, exist_ok=True)
    env = dict(ctx["env"], THESIS_SELECTION_INPUT=os.path.join(run_dir, "sql_revision_rules.snapshot"),
               THESIS_SELECTION_OUTPUT=os.path.join(vdir, f"{REPORTED}.snapshot"), THESIS_SELECTION_WRITERS="",
               THESIS_SELECTION_LOG=os.path.join(vdir, f"{REPORTED}.selection.jsonl"),
               THESIS_LLM_CACHE_DIR=os.path.join(vdir, "llm_cache"))
    rc, _ = run([sys.executable, "-m", "stage7_selection.run_sql_selection"], CODE, env, ctx["log"])
    if rc:
        raise StageFailed(f"selection exited {rc}")


def stage_score(ctx: dict) -> None:
    vdir = os.path.join(ctx["run"], "variants")
    rc, out = run([sys.executable, "-m", "stage8_scoring.run_scoring", "--score-only",
                   "--snapshot", os.path.join(vdir, f"{REPORTED}.snapshot"), "--db-root", ctx["args"].db_root,
                   "--out", os.path.join(vdir, f"{REPORTED}.score.json")], CODE, ctx["env"], ctx["log"])
    open(os.path.join(vdir, f"{REPORTED}.score.txt"), "w").write(out)
    if rc:
        raise StageFailed(f"scoring exited {rc}")


STAGE_FUNCS = {
    "dataset": lambda c: run_base("stage0_dataset.preprocess_dataset", c),
    "value_index": lambda c: run_base("stage1_value_finding.create_vector_db_parallel", c),
    "value_retrieval": lambda c: run_base_retry("stage1_value_finding.run_value_retrieval", c, lambda rc, out: rc == 0),
    "fewshot_index": lambda c: run_base("stage2_similar_examples.build_few_shot_index", c),
    "fewshot_prep": lambda c: run_base_retry("stage2_similar_examples.run_few_shot_preparation", c,
                                             lambda rc, out: (re.findall(r"failed=\d+", out) or ["?"])[-1] == "failed=0"),
    "column_finding": lambda c: run_base_retry("stage3_column_finding.run_schema_linking", c, lambda rc, out: rc == 0),
    "generation": stage_generation,
    "repair": lambda c: run_base_retry("stage5_repair.run_sql_revision", c, lambda rc, out: rc == 0),
    "query_tree": stage_query_tree,
    "selection": stage_selection,
    "score": stage_score,
}


# ------------------------------------------------------------------ main
def build_env(a: argparse.Namespace, cfg_path: str, writer_settings: str) -> dict:
    env = dict(os.environ)
    env.update(CONFIG_PATH=cfg_path, HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", THESIS_LLM_TIMEOUT="3600",
               THESIS_MODEL_SETTINGS=writer_settings, THESIS_LABELS="1", THESIS_FEWSHOT_EXCLUDE_SAME_DB="1",
               TP_BASE_URL=a.base_url, PORT_DISTINCT_FIX="off",
               PYTHONPATH=PYSHIM + (os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else ""))
    env.pop("THESIS_PROMPT_FIXES", None)
    return env


def preflight(a: argparse.Namespace, settings: dict) -> None:
    problems = []
    for name, path in (("BIRD dev", os.path.join(a.data_root, "dev", "dev.json")),
                       ("train file", os.path.join(a.data_root, "train", "train.json")),
                       ("dev databases", a.db_root), ("prep", a.prep), ("embedding model", a.embed_model)):
        if not os.path.exists(path):
            problems.append(f"{name} missing: {path}")
    for path in settings.values():
        print(open(path).read() if a.verbose else f"settings written: {path}")
    print("\n".join(problems) or "inputs: all present")
    if problems:
        raise SystemExit(2)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run-dir", required=True, help="output folder of this run; its name is the run name")
    ap.add_argument("--data-root", required=True, help="BIRD root: dev/dev.json, dev/dev_databases/, train/train.json")
    ap.add_argument("--db-root", required=True, help="the dev databases folder")
    ap.add_argument("--prep", required=True, help="our writer's prep (columns.json per database)")
    ap.add_argument("--vector-store", required=True, help="value index (built where missing)")
    ap.add_argument("--fewshot-index", required=True, help="few-shot index (built when missing)")
    ap.add_argument("--embed-model", required=True, help="local Qwen3-Embedding-0.6B snapshot folder")
    ap.add_argument("--base-url", required=True, help="the vLLM server, e.g. http://127.0.0.1:8000")
    ap.add_argument("--max-samples", type=int, default=None, help="smoke test: first N questions, shorter outputs")
    ap.add_argument("--preflight", action="store_true", help="write the settings files and check inputs only")
    ap.add_argument("--verbose", action="store_true")
    a = ap.parse_args()
    for k in ("data_root", "db_root", "prep", "vector_store", "fewshot_index", "embed_model", "run_dir"):
        setattr(a, k, os.path.abspath(getattr(a, k)))
    run_dir = a.run_dir
    os.makedirs(run_dir, exist_ok=True)
    settings = write_settings(load_config(), a, run_dir)
    cfg_path = settings["base"]
    shutil.copy(cfg_path, os.path.join(run_dir, f"port_config.{datetime.datetime.now():%Y%m%dT%H%M%S}.toml"))
    if a.preflight:
        preflight(a, settings)
        return 0
    writer_settings = settings["plan_and_fragment"]
    ctx = {"args": a, "run": run_dir, "log": os.path.join(run_dir, "pipeline.log"),
           "env": build_env(a, cfg_path, writer_settings), "writer_settings": writer_settings,
           "tree_settings": settings["query_tree"]}
    install_signal_handlers()
    try:
        log(f"pipeline run={run_dir} max_samples={a.max_samples or 'all'}")
        for stage in STAGES:
            done = os.path.join(run_dir, f"done_{stage}")
            if os.path.exists(done):
                log(f"{stage}: already done, skipped")
                continue
            log(f"{stage} start")
            try:
                STAGE_FUNCS[stage](ctx)
            except StageFailed as exc:
                log(f"!! {stage} failed: {exc}. Fix the cause and run the same command again to resume.")
                return 1
            open(done, "w").write(datetime.datetime.now().isoformat())
            log(f"{stage} done")
        print(open(os.path.join(run_dir, "variants", f"{REPORTED}.score.txt")).read())
        return 0
    except KeyboardInterrupt:
        signum = _STOP["signal"] or signal.SIGINT
        try:
            log(f"!! run interrupted by {signal.Signals(signum).name}; the running stage was stopped. "
                "Run the same command again to resume.")
        except OSError:   # the console is gone (e.g. run_all.sh's tee died with the terminal)
            pass
        return 128 + signum


if __name__ == "__main__":
    raise SystemExit(main())
