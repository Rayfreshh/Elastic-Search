from typing import List
from tqdm import tqdm
from utils import get_es_client
from elasticsearch import Elasticsearch
from config import INDEX_NAME
from pprint import pprint
import json
from config import INDEX_NAME_DEFAULT, INDEX_NAME_N_GRAM

print(f"DEBUG: INDEX_NAME = {INDEX_NAME}")


# this is the main function
def index_data(documents: List[dict], use_n_gram_tokenizer: bool = False) -> None:
    es = get_es_client(max_retries=5, sleep_time=2)
    _ = _create_index(es=es, use_n_gram_tokenizer=use_n_gram_tokenizer)
    _ = _insert_documents(es=es, documents=documents, use_n_gram_tokenizer=use_n_gram_tokenizer)
    pprint(
        f'Indexed {len(documents)} documents into Elasticsearch index "{INDEX_NAME}"'
    )


def _create_index(es: Elasticsearch, use_n_gram_tokenizer: bool) -> dict:
   tokenizer = 'n_gram_tokenizer' if use_n_gram_tokenizer else 'standard'
   index_name = INDEX_NAME_N_GRAM if use_n_gram_tokenizer else INDEX_NAME_DEFAULT

    _ = es.indices.delete(index=index_name, ignore_unavailable=True)
    return es.indices.create(
        index=index_name,
        body={
            "settings":{
                "analysis": {
                    "analyzer": {
                        "default": {
                            "type": "custom",
                            "tokenizer": tokenizer,
                            
                        },
                        
                    }
                }
            }
        })


def _insert_documents(es: Elasticsearch, documents: List[dict]) -> dict:
    operations = []
    for document in tqdm(documents, total=len(documents), desc="Indexing documents"):
        operations.append({"index": {"_index": INDEX_NAME}})
        operations.append(document)
    return es.bulk(operations=operations)


# to execute script from terminal
if __name__ == "__main__":
    with open("../../../data/apod.json") as f:
        document = json.load(f)

    index_data(documents=document, use_n_gram_tokenizer=False)
