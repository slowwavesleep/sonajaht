import re

import pandas as pd
from loguru import logger

from schemas import Definition, Synonym
from db_utils import init_sql_session


def import_data():
    logger.info("Reading from `csv` files")
    words = pd.read_csv("tsv/PUBLIC_WORDS.tsv", sep="\t", quotechar="`")
    syns = pd.read_csv("tsv/SYNONYMS.tsv", sep="\t", quotechar="`")
    definitions = pd.read_csv("tsv/DEFINITIONS.tsv", sep="\t", quotechar="`")
    logger.info("Merging dataframes")
    definitions = definitions.merge(words, on="word_id", suffixes=["", "_word"]).rename(
        {"value_word": "word"}, axis=1
    )
    logger.info("Removing tags and merging multiple whitespaces")
    definitions.value = definitions.value.str.replace(
        re.compile(r"<[^>]*>"), "", regex=True
    ).apply(lambda el: " ".join(el.split()))

    session_maker = init_sql_session()
    def_records = [Definition(**rec) for rec in definitions.to_dict("records")]
    syn_records = [
        Synonym(**rec)
        for rec in syns.rename({"type": "rel_type"}, axis=1).to_dict("records")
    ]
    logger.info("Writing data to the database")
    with session_maker() as session:
        session.bulk_save_objects(def_records)
        session.bulk_save_objects(syn_records)
        session.commit()
    logger.success("Data imported")


if __name__ == "__main__":
    import_data()
