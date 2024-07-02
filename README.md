# Sõnajaht: Definition Embeddings and Semantic Search for Reverse Dictionary Creation


## Import Data

Having downloaded the data from [HuggingFace Datasets](https://huggingface.co/datasets/adorkin/sonajaht) into
`tsv` folder, run the data import.

`python preprocess.py`

## Vectorize

```
python vectorize.py "e5-multilingual-large" --prefix query --show_progress_bar
```

## Create Qdrant Collection

Run Qdrant container in Docker and mount local storage to persist the collection.
```
docker run -p 6333:6333 -v $(pwd)/qdrant_mount/qdrant_storage:/qdrant/storage qdrant/qdrant     
```

Create the collection in Qdrant.
```
python collection.py "e5-multilingual-large-query" --postfix "query"
```

## Gradio Demo

```
python search_ui.py
```

## Citation

```
@inproceedings{dorkin-sirts-2024-sonajaht,
    title = "S{\~o}najaht: Definition Embeddings and Semantic Search for Reverse Dictionary Creation",
    author = "Dorkin, Aleksei  and
      Sirts, Kairit",
    editor = "Bollegala, Danushka  and
      Shwartz, Vered",
    booktitle = "Proceedings of the 13th Joint Conference on Lexical and Computational Semantics (*SEM 2024)",
    month = jun,
    year = "2024",
    address = "Mexico City, Mexico",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2024.starsem-1.33",
    pages = "410--420",
}
```