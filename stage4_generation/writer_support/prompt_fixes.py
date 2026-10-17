"""Switch for the 2026-10-07 prompt fix in OUR pipeline code (literal_values).

A fix is active only when the environment variable THESIS_PROMPT_FIXES names it or says "all"
(comma-separated). Unset or empty = every fix off = the behaviour before the fixes existed (pipeline.py removes the
variable, so the fix is off in the pipeline).
  literal_values  the literal head (stage4_generation.plan_and_fragment 'literal') sees this question's retrieved
                  stored values (stage4_generation.writer_support.literal_values -> describe.py's "Relevant stored values:" block)
"""
from __future__ import annotations

import os

ENV = "THESIS_PROMPT_FIXES"


def fix_on(name: str) -> bool:
    names = {part.strip() for part in os.environ.get(ENV, "").split(",")}
    return "all" in names or name in names
