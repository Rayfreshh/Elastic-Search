import json
from pprint import pprint
from typing import List

from elastic_transport import ObjectApiResponse
from elasticsearch import Elasticsearch
from tqdm import tqdm

from config import INDEX_NAME, INDEX_NAME_N_GRAM
from utils import get_es_client


def index_data(docs: List[dict], use_n_gram_tokenizer: bool = False) -> None:
    """
    Index documents into Elasticsearch.

    Args:
        docs: List of documents to index
        use_n_gram_tokenizer: If True, use n-gram tokenizer; otherwise use standard tokenizer
    """
    es = get_es_client(max_retries=5, sleep_time=2)
    _ = _create_index(es=es, use_n_gram_tokenizer=use_n_gram_tokenizer)
    _ = _insert_documents(es=es, docs=docs, use_n_gram_tokenizer=use_n_gram_tokenizer)

    index_name = INDEX_NAME_N_GRAM if use_n_gram_tokenizer else INDEX_NAME
    pprint(f'Indexed {len(docs)} documents into Elasticsearch index "{index_name}"')


def _create_index(es: Elasticsearch, use_n_gram_tokenizer: bool) -> ObjectApiResponse:
    """
    Create an Elasticsearch index with specified tokenizer configuration.

    Args:
        es: Elasticsearch client instance
        use_n_gram_tokenizer: If True, use n-gram tokenizer; otherwise use standard tokenizer

    Returns:
        Elasticsearch index creation response
    """
    tokenizer = "n_gram_tokenizer" if use_n_gram_tokenizer else "standard"
    index_name = INDEX_NAME_N_GRAM if use_n_gram_tokenizer else INDEX_NAME

    _ = es.indices.delete(index=index_name, ignore_unavailable=True)
    return es.indices.create(
        index=index_name,
        body={
            "settings": {
                "analysis": {
                    "analyzer": {
                        "default": {
                            "type": "custom",
                            "tokenizer": tokenizer,
                        },
                    }
                }
            }
        },
    )


def _insert_documents(
    es: Elasticsearch, docs: List[dict], use_n_gram_tokenizer: bool
) -> ObjectApiResponse:
    """
    Insert documents into Elasticsearch index using bulk API.

    Args:
        es: Elasticsearch client instance
        docs: List of documents to insert
        use_n_gram_tokenizer: If True, use n-gram index; otherwise use standard index

    Returns:
        Elasticsearch bulk API response
    """
    operations = []
    index_name = INDEX_NAME_N_GRAM if use_n_gram_tokenizer else INDEX_NAME
    for document in tqdm(docs, total=len(docs), desc="Indexing documents"):
        operations.append({"index": {"_index": index_name}})
        operations.append(document)
    return es.bulk(operations=operations)


# to execute script from terminal
if __name__ == "__main__":
    with open("../../../data/apod.json", encoding="utf-8") as f:
        documents = json.load(f)

    index_data(docs=documents, use_n_gram_tokenizer=True)
