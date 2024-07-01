from loguru import logger
from sentence_transformers import SentenceTransformer

from constants import EMBEDDING_MODELS
from sentence import SentenceEmbedder
from openai_utils import OpenaiEmbedder
from w2v_utils import Word2Vec
from random_parameters import reset_parameters

VectorizationModel = SentenceTransformer | SentenceEmbedder | OpenaiEmbedder | Word2Vec


def init_model(
    model_name: str, random_weights: bool = False
) -> tuple[str, VectorizationModel]:
    logger.info(f"Initializing embedding model: `{model_name}`")
    if model_name in (
        "xlm-roberta-large",
        "estroberta",
        "e5-multilingual-large",
        "gte-large",
    ):
        model = SentenceEmbedder(EMBEDDING_MODELS[model_name])
    elif model_name in ("text-embedding-ada-002",):
        model = OpenaiEmbedder(model_name)
    elif model_name in ("word2vec",):
        model = Word2Vec.load(EMBEDDING_MODELS[model_name])
    else:
        model = SentenceTransformer(EMBEDDING_MODELS[model_name])
    if random_weights:
        logger.info(f"Initializing `{model_name}` with random parameters")
        reset_parameters(model.parameters())
        model_name = f"random-{model_name}"
    logger.success(f"`{model_name}` initialized")
    return model_name, model
