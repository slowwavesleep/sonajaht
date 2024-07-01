import argparse
from pathlib import Path
import time

from qdrant_client import models
from qdrant_client.models import Distance, VectorParams, PointStruct
from safetensors.numpy import load_file
from loguru import logger

from constants import LANGS
from db_utils import init_qdrant_client, init_sql_session


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("cur_model", type=str)
    parser.add_argument("--step", type=int, default=1000)
    parser.add_argument(
        "--collection_name", type=str, default="sonaveeb-semantic-search"
    )
    parser.add_argument("--postfix", choices=["query", "passage"], default=None)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_arguments()
    collection_name: str = args.collection_name
    if args.postfix:
        collection_name = f"{collection_name}-{args.postfix}"

    session_maker = init_sql_session()
    file_path = Path(f"embeddings/{args.cur_model}.safetensors").resolve()
    logger.info(f"Reading safetensors at {file_path}")
    loaded = load_file(file_path)
    logger.success(f"Loaded safetensors from {file_path}")

    client = init_qdrant_client()
    logger.info(f"Creating collection `{collection_name}`")
    client.recreate_collection(
        collection_name=collection_name,
        vectors_config=VectorParams(
            size=loaded["vectors"].shape[1], distance=Distance.COSINE
        ),
        optimizers_config=models.OptimizersConfigDiff(
            indexing_threshold=0,
        ),
    )
    logger.success("Created collection")
    logger.info("Inserting items")
    cur = time.time()
    step: int = args.step
    for i in range(0, loaded["vectors"].shape[0], step):
        logger.debug(f"Current step: {i}")
        points = []
        for index, (vector, definition_id, word_id, lang) in enumerate(
            zip(
                loaded["vectors"].tolist()[i : i + step],
                loaded["definition_ids"].tolist()[i : i + step],
                loaded["word_ids"].tolist()[i : i + step],
                loaded["langs"].tolist()[i : i + step],
            ),
            start=i,
        ):
            points.append(
                PointStruct(
                    id=index,
                    vector=vector,
                    payload=dict(
                        definition_id=definition_id,
                        word_id=word_id,
                        lang=LANGS[lang],
                    ),
                )
            )
        client.upsert(
            collection_name=collection_name,
            points=points,
        )
    logger.success(f"All items are inserted to `{collection_name}`")
    logger.info("Restoring indexing")
    client.update_collection(
        collection_name=collection_name,
        optimizer_config=models.OptimizersConfigDiff(indexing_threshold=20000),
    )
    logger.info(f"Time elapsed: {time.time() - cur}")
