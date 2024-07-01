from qdrant_client import models
from sentence_transformers import SentenceTransformer
import gradio as gr

from constants import EMBEDDING_MODELS
from model_utils import init_model
from schemas import Definition
from sentence import SentenceEmbedder
from db_utils import init_qdrant_client, init_sql_session

CUR_MODEL = "e5-multilingual-large"
# CUR_MODEL = "labse"
# CUR_MODEL = "xlm-roberta-large"
# CUR_MODEL = "estroberta"
# CUR_MODEL = "distiluse-multilingual-v2"

_, model = init_model(CUR_MODEL)

qd_client = init_qdrant_client()
session_maker = init_sql_session()


def search_by_definition(
    q: str, embedding_model, client
) -> list[list[float, str, str, int]]:
    search_filter = models.Filter(
        must=[
            models.FieldCondition(
                key="lang",
                match=models.MatchValue(
                    value="est",
                ),
            )
        ]
    )
    hits = client.search(
        collection_name="sonaveeb-semantic-search-query",
        query_vector=embedding_model.encode([f"query: {q}"]).tolist()[0],  # note
        # query_vector=embedding_model.encode(q).tolist(),
        limit=100,
        offset=1,
        # search_filter=search_filter,
    )
    ids = [hit.payload["definition_id"] for hit in hits]
    scores = [hit.score for hit in hits]
    out = []
    with session_maker() as session:
        output = session.query(Definition).filter(Definition.definition_id.in_(ids))
        results = output.all()
        for index, (score, output) in enumerate(zip(scores, results), start=1):
            out.append([index, score, output.word, output.value, output.word_id])
    return out


def search_interface(definition: str) -> list[list[float, str, str, int]]:
    results = search_by_definition(definition, embedding_model=model, client=qd_client)
    return results


iface = gr.Interface(
    fn=search_interface,
    inputs=gr.Textbox(),
    outputs=gr.Dataframe(
        headers=["index", "score", "word", "definition", "word_id"],
        wrap=True,
        min_width=300,
    ),
    live=False,
)

iface.launch(debug=True)
