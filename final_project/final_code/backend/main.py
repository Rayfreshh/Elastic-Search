import torch
from typing import Union
from config import INDEX_NAME_DEFAULT, INDEX_NAME_EMBEDDING, INDEX_NAME_N_GRAM
from elastic_transport import ObjectApiResponse
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from sentence_transformers import SentenceTransformer
from utils import get_es_client

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = SentenceTransformer("all-MiniLM-L6-v2").to(device)


@app.get("/api/v1/regular_search", response_model=None)
async def regular_search(
    search_query: str,
    skip: int = 0,
    limit: int = 10,
    year: Union[str, None] = None,
    tokenizer: str = "Standard",
) -> Union[dict, HTMLResponse]:
    try:
        es = get_es_client(max_retries=1, sleep_time=0)
        query = {
            "bool": {
                "must": [
                    {
                        "multi_match": {
                            "query": search_query,
                            "fields": ["title", "explanation"],
                        }
                    }
                ]
            }
        }

        if year:
            query["bool"]["filter"] = [
                {
                    "range": {
                        "date": {
                            "gte": f"{year}-01-01",
                            "lte": f"{year}-12-31",
                            "format": "yyyy-MM-dd",
                        }
                    }
                }
            ]

        index_name = (
            INDEX_NAME_DEFAULT if tokenizer == "Standard" else INDEX_NAME_N_GRAM
        )
        response = es.search(
            index=index_name,
            body={
                "query": query,
                "from": skip,
                "size": limit,
            },
            filter_path=[
                "hits.hits._source",
                "hits.hits._score",
                "hits.total",
            ],
        )

        total_hits = get_total_hits(response)
        max_pages = calculate_max_pages(total_hits, limit)

        return {
            "hits": response["hits"].get("hits", []),
            "max_pages": max_pages,
        }
    except Exception as e:
        return handle_error(e)


@app.get("/api/v1/semantic_search", response_model=None)
async def semantic_search(
    search_query: str, skip: int = 0, limit: int = 10, year: Union[str, None] = None
) -> Union[dict, HTMLResponse]:
    try:
        es = get_es_client(max_retries=1, sleep_time=0)
        embedded_query = model.encode(search_query)

        query = {
            "bool": {
                "must": [
                    {
                        "knn": {
                            "field": "embedding",
                            "query_vector": embedded_query,
                            # Because we have 3333 documents, we can return them all.
                            "k": 1e4,
                        }
                    }
                ]
            }
        }

        if year:
            query["bool"]["filter"] = [
                {
                    "range": {
                        "date": {
                            "gte": f"{year}-01-01",
                            "lte": f"{year}-12-31",
                            "format": "yyyy-MM-dd",
                        }
                    }
                }
            ]

        response = es.search(
            index=INDEX_NAME_EMBEDDING,
            body={
                "query": query,
                "from": skip,
                "size": limit,
            },
            filter_path=[
                "hits.hits._source",
                "hits.hits._score",
                "hits.total",
            ],
        )

        total_hits = get_total_hits(response)
        max_pages = calculate_max_pages(total_hits, limit)

        return {
            "hits": response["hits"].get("hits", []),
            "max_pages": max_pages,
        }
    except Exception as e:
        return handle_error(e)


def get_total_hits(response: ObjectApiResponse) -> int:
    return response["hits"]["total"]["value"]


def calculate_max_pages(total_hits: int, limit: int) -> int:
    return (total_hits + limit - 1) // limit


@app.get("/api/v1/get_docs_per_year_count", response_model=None)
async def get_docs_per_year_count(
    search_query: str, tokenizer: str = "Standard"
) -> Union[dict, HTMLResponse]:
    try:
        es = get_es_client(max_retries=1, sleep_time=0)
        query = {
            "bool": {
                "must": [
                    {
                        "multi_match": {
                            "query": search_query,
                            "fields": ["title", "explanation"],
                        }
                    }
                ]
            }
        }

        index_name = (
            INDEX_NAME_DEFAULT if tokenizer == "Standard" else INDEX_NAME_N_GRAM
        )
        response = es.search(
            index=index_name,
            body={
                "query": query,
                "aggs": {
                    "docs_per_year": {
                        "date_histogram": {
                            "field": "date",
                            "calendar_interval": "year",  # Group by year
                            "format": "yyyy",  # Format the year in the response
                        }
                    }
                },
            },
            filter_path=["aggregations.docs_per_year"],
        )
        return {"docs_per_year": extract_docs_per_year(response)}
    except Exception as e:
        return handle_error(e)


def extract_docs_per_year(response: ObjectApiResponse) -> dict:
    aggregations = response.get("aggregations", {})
    docs_per_year = aggregations.get("docs_per_year", {})
    buckets = docs_per_year.get("buckets", [])
    return {bucket["key_as_string"]: bucket["doc_count"] for bucket in buckets}


def handle_error(e: Exception) -> HTMLResponse:
    error_message = f"An error occurred: {str(e)}"
    return HTMLResponse(content=error_message, status_code=500)
