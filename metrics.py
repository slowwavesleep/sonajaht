from collections import defaultdict

import numpy as np
from tqdm import tqdm

from data import PrecisionRecall


def relevant_at_k(reference_ids: list[int], retrieved_items: list[int], k: int) -> int:
    items_retrieved_at_k = retrieved_items[:k]
    items_relevant_at_k = sum(
        1 for item in items_retrieved_at_k if item in reference_ids
    )
    return items_relevant_at_k


def precision_and_recall_at_k(
    reference_ids: list[int], retrieved_items: list[int], k: int
) -> PrecisionRecall:
    if k < 1:
        raise ValueError("`k` must be >= 1")
    if len(reference_ids) > len(set(reference_ids)):
        raise ValueError("`reference_ids` must contain unique ids")
    items_relevant_at_k = relevant_at_k(reference_ids, retrieved_items, k)
    precision = items_relevant_at_k / k
    recall = items_relevant_at_k / len(reference_ids)

    return PrecisionRecall(precision=precision, recall=recall, k=k)


def global_recall_at_k(
    reference_ids_list: list[list[int]], retrieved_items_list: list[list[int]], k: int
) -> float:
    assert len(reference_ids_list) == len(retrieved_items_list)
    assert len(reference_ids_list) > 0
    sum_relevant_at_k = 0
    sum_relevant_items = 0
    for reference_ids, retrieved_items in zip(reference_ids_list, retrieved_items_list):
        items_relevant_at_k = sum(
            1 for item in retrieved_items[:k] if item in reference_ids
        )
        sum_relevant_at_k += items_relevant_at_k
        sum_relevant_items += len(reference_ids)
    return sum_relevant_at_k / sum_relevant_items


def reciprocal_rank(
    reference_items: list[int | str], retrieved_items: list[int | str]
) -> float:
    reference_items_set = set(reference_items)

    for rank, item in enumerate(retrieved_items, 1):
        if item in reference_items_set:
            return 1.0 / rank

    return 0.0


def discounted_cumulative_gain(relevance_scores: list[int | float]) -> float:
    return sum(
        [score / np.log2(i + 1) for i, score in enumerate(relevance_scores, start=1)]
    )


def calculate_all_metrics(
    predictions: list[dict[str, list[int] | int]],
    k_list: list[int],
    count_self: bool,
    count_synonyms: bool = True,
    small: bool = False,
) -> dict[str, float]:
    ranks = []
    precisions = defaultdict(lambda: [])
    recalls = defaultdict(lambda: [])
    accuracies = defaultdict(lambda: [])
    reciprocal_ranks = []
    average_precisions = []
    ndcg_list = []
    sum_relevant_at_k = 0
    sum_relevant_items = 0
    n_queries = 0
    for pred in tqdm(predictions):
        if small:
            target_id = pred["target_id"]
            synonym_ids = pred["synonym_ids"]
            true_ids = set()
            if count_self:
                true_ids.add(target_id)
            if count_synonyms and synonym_ids:
                true_ids.update(synonym_ids)
            matched_word_ids = pred["matched_word_ids"]
        else:
            head_id: int = pred["head_id"]
            true_ids: set[int] = set()
            synonym_ids = set(pred["true_ids"])
            # true_ids: set[int] = set(pred["true_ids"])
            matched_word_ids: list[int] = pred["matched_word_ids"]
            multiple_definitions: bool = pred.get("multiple_definitions", False)
            if count_self and multiple_definitions:
                true_ids = true_ids | {head_id}
            if count_synonyms and synonym_ids:
                true_ids = true_ids | synonym_ids
        if true_ids:
            # count queries that do have ground truth
            n_queries += 1

            num_true_items = len(set(true_ids))
            hits = [
                1 if item in true_ids else 0 for item in matched_word_ids
            ]  # not correct?

            # global recall
            sum_relevant_at_k += sum(hits)
            sum_relevant_items += len(true_ids)

            # median rank
            if 1 in hits:
                ranks.append(hits.index(1))  # make 1-indexed?
            else:
                ranks.append(1000)

            # ndcg
            dcg = discounted_cumulative_gain(hits)
            idcg = discounted_cumulative_gain(sorted(hits, reverse=True))
            if idcg:
                ndcg = dcg / idcg
            else:
                ndcg = 0.0

            ndcg_list.append(ndcg)

            # mean average precision
            indices = np.where(hits)[0].tolist()
            cur_precisions = []
            if indices:
                for index in indices:
                    num_relevant_items = sum(hits[: index + 1])
                    cur_precisions.append(num_relevant_items / (index + 1))
            else:
                cur_precisions.append(0)
            average_precisions.append(np.sum(cur_precisions) / num_true_items)

            # mean reciprocal rank
            reciprocal_ranks.append(reciprocal_rank(list(true_ids), matched_word_ids))

            # precision and recall and accuracy @ k
            for k in sorted(k_list):
                num_relevant_items = sum(hits[:k])
                precision_score = num_relevant_items / k
                recall_score = num_relevant_items / num_true_items
                precisions[k].append(precision_score)
                recalls[k].append(recall_score)
                accuracies[k].append(bool(num_relevant_items))
    result = (
        {
            "n_queries": n_queries,
            "median_rank": np.median(ranks),
            "mean_rank": np.mean(ranks),
            "rank_std": np.std(ranks),
            "mean_ndcg": np.mean(ndcg_list),
            "map": np.mean(average_precisions),
            "mrr": np.mean(reciprocal_ranks),
            "global_recall": sum_relevant_at_k / sum_relevant_items,
        }
        | {
            f"average_precision_at_{k}": np.mean(value)
            for k, value in precisions.items()
        }
        | {f"average_recall_at_{k}": np.mean(value) for k, value in recalls.items()}
        | {f"accuracy_at_{k}": np.mean(value) for k, value in accuracies.items()}
    )
    return result
