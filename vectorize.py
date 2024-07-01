import argparse

from sqlalchemy import select
from safetensors.numpy import save_file
import numpy as np
from loguru import logger

from model_utils import init_model
from constants import LANGS, PREFIXES
from schemas import Definition
from db_utils import init_sql_session

lang2id = {lang: i for i, lang in enumerate(LANGS)}


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("model_name", type=str, help="Name of the model")
    parser.add_argument("--n_items", type=int, default=None)
    parser.add_argument("--prefix", choices=["query", "passage"], default=None)
    parser.add_argument("--show_progress_bar", action="store_true")
    parser.add_argument("--random_weights", action="store_true")
    parser.add_argument("--batch_size", type=int, default=1024)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_arguments()
    session_maker = init_sql_session()
    model_name = args.model_name
    model_name, model = init_model(model_name, args.random_weights)
    logger.info("Reading from the database")
    with session_maker() as session:
        statement = (
            select(
                Definition.definition_id,
                Definition.value,
                Definition.word_id,
                Definition.lang,
            )
            .distinct()
            .limit(args.n_items)
        )
        definition_records = session.execute(statement)

        definition_ids = []
        definitions = []
        word_ids = []
        langs = []
        for definition_id, value, word_id, lang in definition_records:
            definition_ids.append(definition_id)
            definitions.append(value)
            word_ids.append(word_id)
            langs.append(lang)
        if args.prefix:
            logger.info(f"Prepending definitions with prefix `{args.prefix}`")
            cur_definitions = [
                f"{PREFIXES[args.prefix]}{definition}" for definition in definitions
            ]
        else:
            cur_definitions = definitions
        logger.info("Encoding definitions")
        vectors = model.encode(
            cur_definitions,
            batch_size=args.batch_size,
            show_progress_bar=args.show_progress_bar,
        )
        assert len(definition_ids) == vectors.shape[0]
        output = dict(
            vectors=vectors,
            definition_ids=np.array(definition_ids),
            word_ids=np.array(word_ids),
            langs=np.array([lang2id[lang] for lang in langs]),
        )
        if args.prefix:
            file_name = f"{model_name}-{args.prefix}.safetensors"
        else:
            file_name = f"{model_name}.safetensors"
        logger.info(f"Saving to `{file_name}`")
        save_file(tensor_dict=output, filename=file_name)
        logger.success(f"Successfully saved to `{file_name}`")
