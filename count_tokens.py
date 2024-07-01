from tiktoken import get_encoding
from sqlalchemy import select
from loguru import logger

from schemas import Definition
from db_utils import init_sql_session

if __name__ == "__main__":
    session_maker = init_sql_session()
    logger.info("Reading from the database")
    with session_maker() as session:
        statement = select(Definition.value).distinct()
        definition_records = session.execute(statement)
        definitions = [value[0] for value in definition_records]
    logger.info("Initializing tokenizer")
    tokenizer = get_encoding("cl100k_base")
    logger.info("Tokenizing")
    tokenized = tokenizer.encode_batch(definitions, num_threads=8)
    num_tokens = 0
    for sent in tokenized:
        num_tokens += len(sent)
    logger.info(f"Total tokens: {num_tokens}")
