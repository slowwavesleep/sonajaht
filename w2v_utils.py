import gzip
from pathlib import Path
import re

import torch
import numpy as np
from tqdm import tqdm
from safetensors.numpy import save_file, load_file


class Word2Vec:
    def __init__(self, vectors, words):
        self.vectors = vectors
        self.words = words
        self.word2id = {word: i for i, word in enumerate(self.words)}
        self.id2word = {value: key for key, value in self.word2id.items()}

    @classmethod
    def from_gz(cls, path):
        with gzip.open(Path(path).resolve(), "rt", encoding="utf-8") as file:
            num_words, vector_size = file.readline().strip().split()
            num_words = int(num_words)
            vector_size = int(vector_size)
            words = []
            vectors = []
            for line in tqdm(file, total=num_words):
                if line:
                    word, *vector = line.strip().split()
                    vector = torch.tensor([float(el) for el in vector]).unsqueeze(0)
                    words.append(word)
                    vectors.append(vector)
            vectors = torch.cat(vectors, dim=0)
            if vectors.shape[0] != num_words:
                raise ValueError(
                    f"The number of words doesn't match: `{num_words}` != `{vectors.shape[0]}`"
                )
            if vectors.shape[1] != vector_size:
                raise ValueError(
                    f"Vector size mismatch: `{vector_size}` != `{vectors.shape[1]}`"
                )
            return cls(vectors=vectors, words=words)

    @classmethod
    def load(cls, save_dir: str):
        save_dir = Path(save_dir).resolve()
        vectors = load_file(save_dir / "word2vec.safetensors")["vectors"]
        with open(save_dir / "vocab.txt") as file:
            words = file.read().split("\n")
        return cls(vectors=vectors, words=words)

    def save(self, save_dir):
        save_dir = Path(save_dir).resolve()
        save_dir.mkdir(exist_ok=True, parents=True)
        save_file(
            tensor_dict=dict(vectors=self.vectors.numpy()),
            filename=save_dir / "word2vec.safetensors",
        )
        with open(save_dir / "vocab.txt", "w") as file:
            file.write("\n".join(self.words))

    @staticmethod
    def tokenize(text: str):
        return re.findall(r"\w+", text)

    def encode_single(self, text: str):
        tokens = self.tokenize(text)
        vectors = []
        for token in tokens:
            word_id = self.word2id.get(token, None)
            if word_id is not None:
                vectors.append(self.vectors[word_id])
        if vectors:
            return np.mean(np.vstack(vectors), axis=0)
        else:
            return np.zeros(self.vectors.shape[1])

    def encode(
        self,
        sentence_list: list[str],
        batch_size: int = 1024,
        show_progress_bar: bool = False,
    ) -> np.array:
        final_embeddings = []
        for index in tqdm(range(0, len(sentence_list)), disable=not show_progress_bar):
            embeddings = self.encode_single(sentence_list[index])
            final_embeddings.append(embeddings)
        return np.vstack(final_embeddings)

    def _search(self, vector: np.array, k: int):
        similarities = np.dot(
            np.expand_dims(vector / np.linalg.norm(vector), axis=0),
            (self.vectors / np.linalg.norm(self.vectors)).transpose(),
        ).squeeze()
        sorted_similarities = np.argsort(similarities)[::-1][:k].tolist()
        return [self.id2word[index] for index in sorted_similarities]

    def search_by_text(self, sentence: str, k: int = 100):
        vector = self.encode_single(sentence)
        return self._search(vector, k)

    def search_by_vector(self, vector: np.array, k: int = 100):
        return self._search(vector, k)
