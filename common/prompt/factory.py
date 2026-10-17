from .prompt_template import *
from typing import List, Dict, Any, Tuple


def _format_few_shot_example(index: int, example: Dict[str, Any]) -> str:
    lines = [f"- Example {index}:", f"Question: {example['question']}"]
    evidence = str(example.get("evidence", example.get("hint", ""))).strip()
    if evidence:
        lines.append(f"Hint: {evidence}")
    lines.append(f"SQL: {example['sql']}")
    return "\n".join(lines)


class PromptFactory:
    
    @staticmethod
    def format_keywords_extraction_prompt(question: str, hint: str) -> str:
        return KEYWORDS_EXTRACTION_PROMPT.format(QUESTION=question, HINT=hint)
    
    @staticmethod
    def format_direct_linking_prompt(database_schema: str, question: str, hint: str) -> str:
        return DIRECT_LINKING_PROMPT.format(DATABASE_SCHEMA=database_schema, QUESTION=question, HINT=hint)
    
    @staticmethod
    def format_skeleton_sql_generation_prompt(database_schema: str, question: str, hint: str) -> str:
        return SKELETON_SQL_GENERATION_PROMPT.format(DATABASE_SCHEMA=database_schema, QUESTION=question, HINT=hint)
    
    @staticmethod
    def format_dc_sql_generation_prompt(database_schema: str, question: str, hint: str) -> str:
        return DC_SQL_GENERATION_PROMPT.format(DATABASE_SCHEMA=database_schema, QUESTION=question, HINT=hint)
    
    @staticmethod
    def format_icl_sql_generation_prompt(few_shot_examples: List[Dict[str, Any]], database_schema: str, question: str, hint: str) -> str:
        few_shot_examples_str = "\n".join(
            _format_few_shot_example(i + 1, example)
            for i, example in enumerate(few_shot_examples)
        )
        return ICL_SQL_GENERATION_PROMPT.format(FEW_SHOT_EXAMPLES=few_shot_examples_str, DATABASE_SCHEMA=database_schema, QUESTION=question, HINT=hint)

    @staticmethod
    def format_execution_checker_prompt(database_schema: str, question: str, hint: str, sql: str, execution_result: str) -> str:
        return EXECUTION_CHECKER_PROMPT.format(DATABASE_SCHEMA=database_schema, QUESTION=question, HINT=hint, QUERY=sql, RESULT=execution_result)
    
    @staticmethod
    def format_common_checker_prompt(database_schema: str, question: str, hint: str, sql: str, suggestions: str) -> str:
        return COMMON_CHECKER_PROMPT.format(DATABASE_SCHEMA=database_schema, QUESTION=question, HINT=hint, QUERY=sql, SUGGESTIONS=suggestions)
    
    @staticmethod
    def format_br_pair_selection_prompt(database_schema: str, question: str, hint: str, query_a: str, result_a: str, query_b: str, result_b: str) -> str:
        return BR_PAIR_SELECTION_PROMPT.format(DATABASE_SCHEMA=database_schema, QUESTION=question, HINT=hint, QUERY_A=query_a, RESULT_A=result_a, QUERY_B=query_b, RESULT_B=result_b)