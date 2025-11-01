import json
from pprint import pprint
from typing import List

from elastic_transport import ObjectApiResponse
from elasticsearch import Elasticsearch
from tqdm import tqdm

from config import INDEX_NAME_RAW
from utils import get_es_client


def index_data(docs: List[dict]) -> None:
    """
    Index documents into Elasticsearch.

    Args:
        docs: List of documents to index
        use_n_gram_tokenizer: If True, use n-gram tokenizer; otherwise use standard tokenizer
    """

    pipeline_id = "apod_pipeline"
    es = get_es_client(max_retries=5, sleep_time=2)
    _ = _create_index(es=es)
    _ = _insert_documents(es=es, docs=docs, pipeline_id=pipeline_id)

    pprint(f'Indexed {len(docs)} documents into Elasticsearch index "{INDEX_NAME_RAW}"')


def create_pipeline(es: Elasticsearch, pipeline_id: str) -> ObjectApiResponse:
    pipeline_body = {
        "description": "Pipeline that strips HTML tags from the explanation field and title field",
        "processors": [
            {"html_strip": {"field": "explanation"}},
            {"html_strip": {"field": "title"}},
        ],
    }
    return es.ingest.put_pipeline(id=pipeline_id, body=pipeline_body)


def _create_index(es: Elasticsearch) -> ObjectApiResponse:
    _ = es.indices.delete(index=INDEX_NAME_RAW, ignore_unavailable=True)
    return es.indices.create(index=INDEX_NAME_RAW)


def _insert_documents(
    es: Elasticsearch, docs: List[dict], pipeline_id: str
) -> ObjectApiResponse:
    operations = []
    for document in tqdm(docs, total=len(docs), desc="Indexing documents"):
        operations.append({"index": {"_index": INDEX_NAME_RAW}})
        operations.append(document)
    return es.bulk(operations=operations, pipeline=pipeline_id)


# to execute script from terminal
if __name__ == "__main__":
    with open("../../../data/apod_raw.json", encoding="utf-8") as f:
        documents = json.load(f)

    index_data(docs=documents)
