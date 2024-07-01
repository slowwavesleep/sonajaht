import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from qdrant_client import models
from loguru import logger
from tqdm import tqdm

from db_utils import init_qdrant_client, init_sql_session
from model_utils import init_model
from constants import PREFIXES


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("model_name", type=str, default="word2vec")
    parser.add_argument("--test_file", type=str, default="defs.tsv")
    parser.add_argument("--prefix", choices=["query", "passage"], default=None)
    parser.add_argument(
        "--collection_name", type=str, default="sonaveeb-semantic-search"
    )
    parser.add_argument("--target_col", type=str, default="word_ee")
    parser.add_argument("--target_id_col", type=str, default="word_ee_id")
    parser.add_argument("--syn_col", type=str, default="synonym_ids")
    parser.add_argument("--eng_col", type=str, default="def_en")
    parser.add_argument("--est_col", type=str, default="def_ee")
    parser.add_argument("--rus_col", type=str, default="def_ru")

    parser.add_argument("--word2vec_baseline", action="store_true")

    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--est", action="store_true", help="Use definitions in Estonian for search"
    )
    group.add_argument(
        "--eng", action="store_true", help="Use definitions in English for search"
    )
    group.add_argument(
        "--rus", action="store_true", help="Use definitions in Russian for search"
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_arguments()
    collection_name = args.collection_name
    target_col = args.target_col
    target_id_col = args.target_id_col
    syn_col = args.syn_col
    data = pd.read_csv(args.test_file, sep="\t")
    session_maker = init_sql_session()
    _, model = init_model(args.model_name)

    if args.est:
        logger.info("Using definitions in Estonian")
        definition_lang = "est"
        definitions = data[args.est_col].tolist()
    elif args.eng:
        logger.info("Using definitions in English")
        definition_lang = "eng"
        definitions = data[args.eng_col].tolist()
    elif args.rus:
        logger.info("Using definitions in Russian")
        definition_lang = "rus"
        definitions = data[args.rus_col].tolist()
    else:
        raise ValueError("No language option selected")

    if args.prefix:
        logger.info(f"Prepending definitions with prefix `{args.prefix}`")
        definitions = [
            f"{PREFIXES[args.prefix]}{definition}" for definition in definitions
        ]
    logger.info("Encoding definitions")
    vectors = model.encode(definitions).tolist()
    target_words = data[target_col].tolist()
    target_ids = data[target_id_col].tolist()
    synonym_ids = [
        [int(i) for i in el.split(",")] if isinstance(el, str) else None
        for el in data[syn_col].tolist()
    ]

    if not args.word2vec_baseline:
        qd_client = init_qdrant_client()
        search_queries = [
            models.SearchRequest(
                vector=vector,
                limit=100,
                with_payload=True,
                offset=0,  # not skipping the first item because the query definitions are different from what we have
            )
            for vector in vectors
        ]

        logger.info("Performing vector search")
        matches = qd_client.search_batch(
            collection_name=collection_name,
            requests=search_queries,
        )
        matched_ids: list[list[int]] = [
            [point.payload["word_id"] for point in match] for match in matches
        ]

    elif args.model_name == "word2vec":
        logger.info("Performing baseline word2vec search")
        matched_ids = []
        for vector in tqdm(vectors):
            matched_ids.append(model.search_by_vector(np.array(vector)))

    else:
        raise ValueError("word2vec baseline is only usable with word2vec model")
    assert len(matched_ids) == len(target_words)
    out_prefix = f"{args.model_name}-{definition_lang}"

    if args.prefix:
        query_prefix_name = f"-{args.prefix}"
    else:
        query_prefix_name = ""

    if "passage" in args.collection_name:
        collection_postfix_name = "-passage"
    elif "query" in args.collection_name:
        collection_postfix_name = "-query"
    else:
        collection_postfix_name = ""

    if args.word2vec_baseline:
        baseline_str = "-baseline"
    else:
        baseline_str = ""
    raw_output_dir = "./raw_output/small"
    raw_output_path = Path(raw_output_dir).resolve()
    raw_output_path.mkdir(exist_ok=True)
    out_suffix = "-preds.jsonl"
    out_file_name = (
        raw_output_path
        / f"{out_prefix}{query_prefix_name}{collection_postfix_name}{baseline_str}{out_suffix}"
    )
    logger.info(f"Writing out predictions to `{out_file_name}`")
    with open(out_file_name, "w") as file:
        for target_word, target_id, candidates, cur_syn_ids in zip(
            target_words, target_ids, matched_ids, synonym_ids
        ):
            cur_data = dict(
                target_word=target_word,
                target_id=target_id,
                matched_word_ids=candidates,
                synonym_ids=cur_syn_ids,
            )
            file.write(json.dumps(cur_data, ensure_ascii=False) + "\n")
