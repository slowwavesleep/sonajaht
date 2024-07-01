from dataclasses import dataclass


@dataclass
class SearchResult:
    head_id: int
    matched_word_ids: list[int]
    matched_definition_ids: list[int]
    true_ids: list[int]
    multiple_definitions: bool


@dataclass
class PrecisionRecall:
    precision: float
    recall: float
    k: int
