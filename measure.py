import json
from dataclasses import asdict
from pathlib import Path
from time import time
import argparse

from sqlalchemy import func
from sqlalchemy.sql import union_all
from qdrant_client import models
from loguru import logger
import numpy as np
from tqdm import tqdm

from data import SearchResult
from schemas import Synonym, Definition
from db_utils import init_qdrant_client, init_sql_session


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scroll_step", type=int, default=3000)
    parser.add_argument("--step", type=int, default=1000)

    query_lang_group = parser.add_mutually_exclusive_group()

    query_lang_group.add_argument(
        "--est_only",
        action="store_true",
        help="Consider as queries only definitions in Estonian",
    )

    query_lang_group.add_argument(
        "--rest_only",
        action="store_true",
        help="Consider as queries only definitions that are not in Estonian",
    )

    # parser.add_argument("--est_only", action="store_true")
    parser.add_argument("--random_baseline", action="store_true")
    parser.add_argument("--max_k", type=int, default=100)
    parser.add_argument(
        "--query_collection_name", type=str, default="sonaveeb-semantic-search"
    )
    parser.add_argument(
        "--passage_collection_name", type=str, default="sonaveeb-semantic-search"
    )
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--out_file_name", type=str, default="results.jsonl")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_arguments()
    qd_client = init_qdrant_client()
    session_maker = init_sql_session()

    # mirror all synonymy relations and use them as the ground truth
    cur = time()
    with session_maker() as session:
        subquery1 = session.query(
            Synonym.head_id.label("head_id"), Synonym.tail_id.label("tail_id")
        )
        subquery2 = session.query(
            Synonym.tail_id.label("head_id"), Synonym.head_id.label("tail_id")
        )
        all_relations = union_all(subquery1, subquery2).alias()

        result = (
            session.query(
                all_relations.c.head_id,
                func.group_concat(all_relations.c.tail_id).label("unique_tail_ids"),
            )
            .filter(all_relations.c.head_id != all_relations.c.tail_id)
            .group_by(all_relations.c.head_id)
            .all()
        )

    logger.debug(f"Number of unique words with synonyms: {len(result)}")
    ground_truth = {}
    unique_word_ids = []
    for head_id, unique_tail_ids in result:
        tail_ids = [int(el) for el in unique_tail_ids.split(",")]
        unique_word_ids.append(head_id)
        unique_word_ids.extend(tail_ids)
        ground_truth[head_id] = list(set(tail_ids))

    logger.info(
        f"Unique words in mirrored synonymy relations: {len(set(unique_word_ids))}"
    )
    elapsed = time() - cur
    logger.info(f"Time elapsed on mirroring synonyms SQL query: {elapsed}")

    # find word_ids with multiple definitions
    cur = time()
    definitions_query = (
        session.query(Definition.word_id)
        .group_by(Definition.word_id)
        .having(func.count(Definition.definition_id) > 1)
    )
    if args.est_only:
        definitions_query = definitions_query.filter(Definition.lang == "est")
    definitions_results = definitions_query.all()
    multiple_definitions_word_ids = set([el[0] for el in definitions_results])
    logger.info(
        f"Total words with multiple definitions: {len(multiple_definitions_word_ids)}"
    )
    elapsed = time() - cur
    logger.info(f"Time elapsed on querying words with multiple definitions: {elapsed}")

    # determine the number of items to scroll
    if not args.limit:
        limit = qd_client.count(collection_name=args.query_collection_name).count
    else:
        limit = args.limit
    logger.debug(f"Total items: {limit}")
    scroll_step: int = args.scroll_step
    points = []

    # define language filter
    cur = time()
    if args.est_only:
        logger.info("Filtering out definitions that are not in Estonian")
        qd_filter = models.Filter(
            must=[
                models.FieldCondition(
                    key="lang",
                    match=models.MatchValue(
                        value="est",
                    ),
                )
            ]
        )
    elif args.rest_only:
        logger.info("Filtering out definitions that are in Estonian")
        qd_filter = models.Filter(
            must_not=[
                models.FieldCondition(
                    key="lang",
                    match=models.MatchValue(
                        value="est",
                    ),
                )
            ]
        )
    else:
        qd_filter = None

    # scroll over every item (except for filtered out items)
    cur_offset = 0
    for i in range(0, limit, scroll_step):
        logger.debug(f"Current offset {cur_offset}")
        cur_points, cur_offset = qd_client.scroll(
            collection_name=args.query_collection_name,
            limit=scroll_step,
            with_payload=True,
            with_vectors=True,
            offset=cur_offset,
            scroll_filter=qd_filter,
        )
        logger.debug(f"Current points: {len(cur_points)}")
        points.extend(cur_points)
        if not cur_offset:
            logger.debug("Scrolled through all the items")
            break
    elapsed = time() - cur
    logger.info(f"Time elapsed on scroll: {elapsed}")

    # process the retrieved items
    head_ids = [point.payload["word_id"] for point in points]
    definition_ids = [point.payload["definition_id"] for point in points]
    vectors = [point.vector for point in points]
    logger.info(f"Total points retrieved: {len(head_ids)}")

    # use every retrieved item as a search query
    cur = time()
    step: int = args.step
    max_k: int = args.max_k
    matches = []
    if not args.random_baseline:
        logger.info("Commencing vector search")
        for i in range(0, len(vectors), step):
            logger.debug(f"Current step: {i}")
            search_queries = [
                models.SearchRequest(
                    vector=vector,
                    limit=max_k,
                    filter=qd_filter,
                    with_payload=True,
                    offset=1,  # skip the item itself
                )
                for vector in vectors[i : i + step]
            ]
            matched_word_ids = qd_client.search_batch(
                collection_name=args.passage_collection_name,
                requests=search_queries,
            )
            matches.extend(matched_word_ids)
    else:
        logger.info("Random baseline: skipping vector search")

    # process search results
    search_results = []
    if not args.random_baseline:
        for head_id, definition_id, match in zip(head_ids, definition_ids, matches):
            matched_word_ids = [point.payload["word_id"] for point in match]
            matched_definition_ids = [point.payload["definition_id"] for point in match]
            true_ids = ground_truth.get(head_id, [])
            search_results.append(
                SearchResult(
                    head_id=head_id,
                    matched_word_ids=matched_word_ids,
                    matched_definition_ids=matched_definition_ids,
                    true_ids=true_ids,
                    multiple_definitions=head_id in multiple_definitions_word_ids,
                )
            )
    else:
        logger.info("Sampling random `matched_word_ids`")
        matched_word_ids_list = np.random.choice(
            a=list(set(unique_word_ids)), size=(len(head_ids), max_k), replace=True
        ).tolist()
        for head_id, matched_word_ids in zip(head_ids, matched_word_ids_list):
            true_ids = ground_truth.get(head_id, [])
            search_results.append(
                SearchResult(
                    head_id=head_id,
                    matched_word_ids=matched_word_ids,
                    matched_definition_ids=[],  # the field is only used for debugging
                    true_ids=true_ids,
                    multiple_definitions=head_id in multiple_definitions_word_ids,
                )
            )
    logger.debug(f"{len(search_results)}")
    elapsed = time() - cur
    logger.info(f"Time elapsed on search: {elapsed}")

    raw_output_dir = "./raw_output/large"
    raw_output_path = Path(raw_output_dir).resolve()
    raw_output_path.mkdir(parents=False, exist_ok=True)
    out_file_name = args.out_file_name
    out_file_path = raw_output_path / out_file_name
    logger.info(f"Writing results to `{out_file_name}` in `{raw_output_dir}`")
    with open(out_file_path, "w") as file:
        for el in tqdm(search_results):
            file.write(json.dumps(asdict(el)) + "\n")
    logger.success(f"Succeeded in writing results to `{out_file_name}`")
