import numpy as np
from transformers import AutoTokenizer, AutoModel
import torch
from torch.nn import functional
from tqdm import tqdm
from loguru import logger


class SentenceEmbedder:
    def __init__(self, model_name: str, device: str | None = None):
        self.model_name = model_name
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)
        if not device:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device
        logger.info(f"Initializing encoding model on `{self.device}`")
        self.model.to(self.device)

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
            encoded = self.tokenizer(
                sentence_list[index : index + batch_size],
                padding=True,
                truncation=True,
                return_tensors="pt",
                # max_length=512,
            ).to(self.device)
            with torch.no_grad():
                output = self.model(**encoded)
            embeddings = mean_pooling(output, encoded["attention_mask"])
            embeddings = functional.normalize(embeddings, p=2, dim=1).detach().cpu()
            final_embeddings.append(embeddings)
        return torch.cat(final_embeddings, dim=0).numpy()

    def parameters(self):
        return self.model.parameters()


# Mean Pooling - Take attention mask into account for correct averaging
def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output[
        0
    ]  # First element of model_output contains all token embeddings
    input_mask_expanded = (
        attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    )
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(
        input_mask_expanded.sum(1), min=1e-9
    )
