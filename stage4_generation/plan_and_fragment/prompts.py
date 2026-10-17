"""Prompt templates of the literal SQL writer (Plan-and-Fragment), kept as constants for diffing.

SQLITE_RULES, OUTPUT_FORMAT, EVIDENCE_NOTE and INPUT_TEMPLATE come from an earlier version of this pipeline's
prompts (its rules, write-guide and input template).
Changes (all deliberate):
  - Output format: the old <reasoning>..</reasoning><result>SQL</result> XML is replaced by
    "end with exactly one ```sql block". gpt-oss already returns its chain of thought in
    message.reasoning, and common.sql.extract.extract_sql takes the LAST fenced block, so
    intermediate fragments must not be fenced (dc writes pseudo-SQL as plain text).
  - The 13 rules + write-guide are cut to 6 SQLite rules (design: <= 6).
  - Evidence is labelled authoritative (the old prompt called it "Hints").
  - All env-gated variants (NP_SHAPE_CONTRACT, NP_GEN_WRITE_GUIDE, NP_VALUE_HINTS, capcards)
    are removed: one fixed prompt per generator.
"""
from __future__ import annotations

SQLITE_RULES = """# SQLite rules:
1. Use only SQLite syntax and functions; use STRFTIME() for dates (e.g. STRFTIME('%Y', col) for the year).
2. Wrap any table or column name that contains spaces or special characters in backticks, e.g. `Free Meal Count (K-12)`.
3. For division, ratios, averages of integers and percentages force real division: CAST(x AS REAL) / y.
4. SELECT only the columns the question asks for, in the order it asks for them; do not concatenate strings or add extra columns.
5. JOIN tables only through the foreign keys shown in the schema; qualify columns with their table (or alias) when joined tables share a column name.
6. Copy filter literals exactly as they appear in the value examples or the evidence (same case, spacing and zero-padding)."""

OUTPUT_FORMAT = """# Output format:
Write your reasoning as plain text (no code blocks for intermediate steps).
End your answer with exactly one ```sql code block that contains only the final SQLite query:
no comments, no explanation text, no second statement."""

EVIDENCE_NOTE = (
    "The evidence is authoritative: it defines the question's terms, formulas and stored values. "
    "Follow it exactly."
)

INPUT_TEMPLATE = """# Database schema:
{schema}

# Evidence:
{evidence}

# Question:
{question}"""

# ---------------------------------------------------------------- literal head (2026-09-28, generator gap study)
# New (not from the earlier prompts). Designed on train failures only. The three steps
# target the winnable no-correct causes measured there: output columns, evidence applied literally, and the
# model's habit of adding COUNT(DISTINCT) / DISTINCT / NULL guards / CTEs that BIRD's plain SQL does not have.
LITERAL_HEAD = """# Task:
You are an experienced database expert. Translate the question into the plainest SQLite query that answers it
literally, the way a careful annotator writes the reference SQL for this benchmark.

# Instructions:
Work in three steps and show them in your reasoning.
1. **Output plan**: list exactly the items the question asks to return, in the order it asks for them; nothing else.
   "How many ..." is one number. Return the column the question or evidence names (an id stays an id, a name stays
   a name); do not add helper columns such as ids, counts or sort keys.
2. **Evidence fragments**: rewrite every clause of the evidence as the SQL fragment it defines, and use each
   fragment verbatim: its columns, literals, formula and scale (a percentage multiplies by 100 when the evidence or
   question says percentage).
3. **Plain SQL**: write the most direct query over the tables that hold the facts.
   - COUNT(column) or COUNT(*) counts rows; use COUNT(DISTINCT ...) or SELECT DISTINCT only when the question says
     distinct, different, unique, kinds or types, or the evidence writes DISTINCT.
   - No IS NOT NULL, COALESCE, IFNULL or NULLIF guards unless the question or evidence asks about missing values.
   - No CTEs (WITH), window functions or extra subqueries when a join, ORDER BY ... LIMIT or a single aggregate does it.
   - Use the fewest tables: do not add a second table (e.g. a historical copy) that the question does not ask for.
   - "the highest / lowest / most ..." returns the top row with ORDER BY ... LIMIT 1 unless the question asks for all ties."""

LITERAL_REASK = """Your query uses COUNT(DISTINCT ...), but the question does not say distinct, different, unique, kinds or types,
and the evidence does not write DISTINCT. In this benchmark "how many" usually counts rows with COUNT(column) or COUNT(*).
Re-read the question. If its wording really requires counting distinct values, keep your query unchanged; otherwise
rewrite it with a plain COUNT. Reply with the final query only, in one ```sql code block."""
