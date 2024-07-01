import torch
import numpy as np
import openai
from tenacity import retry, stop_after_attempt, wait_random_exponential
from tqdm import tqdm


@retry(wait=wait_random_exponential(min=1, max=20), stop=stop_after_attempt(6))
def get_embeddings(
    list_of_text: list[str], engine="text-embedding-ada-002", **kwargs
) -> list[list[float]]:
    assert len(list_of_text) <= 2048, "The batch size should not be larger than 2048."
    list_of_text = [text.replace("\n", " ") for text in list_of_text]
    data = openai.Embedding.create(input=list_of_text, engine=engine, **kwargs).data
    return [d["embedding"] for d in data]


class OpenaiEmbedder:
    def __init__(self, model_name: str = "text-embedding-ada-002"):
        self.model_name = model_name

    def encode(
        self,
        sentence_list: list[str],
        batch_size: int = 1024,
        show_progress_bar: bool = False,
    ) -> np.array:
        final_embeddings = []
        for index in tqdm(
            range(0, len(sentence_list), batch_size), disable=not show_progress_bar
        ):
            embeddings = torch.tensor(
                get_embeddings(
                    list_of_text=sentence_list[index : index + batch_size],
                    engine=self.model_name,
                )
            )
            final_embeddings.append(embeddings)
        return torch.cat(final_embeddings, dim=0).numpy()


if __name__ == "__main__":
    model = OpenaiEmbedder(model_name="text-embedding-ada-002")
    print(
        model.encode(
            sentence_list=["test1", "test2"], batch_size=1, show_progress_bar=True
        ).shape
    )
