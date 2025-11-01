import json
from pprint import pprint
from typing import List
import torch

from elastic_transport import ObjectApiResponse
from elasticsearch import Elasticsearch
from tqdm import tqdm

from config import INDEX_NAME_EMBEDDING
from utils import get_es_client
from sentence_transformers import SentenceTransformer


def index_data(docs: List[dict], model: SentenceTransformer) -> None:
    """
    Index documents into Elasticsearch.

    Args:
        docs: List of documents to index
        use_n_gram_tokenizer: If True, use n-gram tokenizer; otherwise use standard tokenizer
    """
    es = get_es_client(max_retries=5, sleep_time=2)
    _ = _create_index(es=es)
    _ = _insert_documents(es=es, docs=docs, model=model)

    pprint(
        f'Indexed {len(docs)} documents into Elasticsearch index "{INDEX_NAME_EMBEDDING}"'
    )


def _create_index(es: Elasticsearch) -> ObjectApiResponse:
    index_name = INDEX_NAME_EMBEDDING

    _ = es.indices.delete(index=index_name, ignore_unavailable=True)
    return es.indices.create(
        index=index_name,
        mappings={
            "properties": {
                "embedding": {
                    "type": "dense_vector",
                }
            }
        },
    )


def _insert_documents(
    es: Elasticsearch, docs: List[dict], model: SentenceTransformer
) -> ObjectApiResponse:
    """
    Insert documents into Elasticsearch index with embeddings.
    """
    operations = []
    for document in tqdm(docs, total=len(docs), desc="Indexing documents"):
        operations.append({"index": {"_index": INDEX_NAME_EMBEDDING}})
        operations.append({
            **document,
            "embedding": model.encode(document["explanation"])
        })
    return es.bulk(operations=operations)


# to execute script from terminal
if __name__ == "__main__":
    with open("../../../data/apod.json", encoding="utf-8") as f:
        documents = json.load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SentenceTransformer("all-MiniLM-L6-v2").to(device)
    index_data(docs=documents, model=model)
