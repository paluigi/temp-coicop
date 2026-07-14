from datetime import datetime

import pandas as pd
from tinydb import TinyDB, Query
from qdrant_client import QdrantClient, models
from sentence_transformers import SentenceTransformer

from minio import Minio
from dotenv import dotenv_values

config_dict = dotenv_values(".env")

db = TinyDB('knn_results/db.json')
Item = Query()

encoder_list = [
#    "jinaai/jina-embeddings-v3",
#    "paraphrase-multilingual-mpnet-base-v2",
#    "sentence-transformers/distiluse-base-multilingual-cased-v1",
#    "sentence-transformers/LaBSE",
#    "Alibaba-NLP/gte-multilingual-base",
#    "Alibaba-NLP/gte-Qwen2-7B-instruct",
#    "ibm-granite/granite-embedding-278m-multilingual",
#    "intfloat/multilingual-e5-base",
#    "intfloat/multilingual-e5-large",
#    "RamsesDIIP/me5-large-construction-v2",
    "intfloat/e5-mistral-7b-instruct"
]

def search_metrics(client, encoder, collection_name, metric, cat, enc):
    results = []
    search_item = "{} {}".format(cat["name"],  cat["category"])
    
    hits = client.search(
        collection_name=collection_name,
        query_vector=encoder.encode(search_item).tolist(),
        limit=9
    )
    for hit in hits:
        results.append({
            **hit.payload,
            "score": hit.score,
            "search_item": search_item,
            "name": cat["name"],
            "category": cat["category"],
            "manual_code": cat["code"],
            "distance": metric,
            "encoder": enc
        })
    return results


def minio_upload(config_dict: dict, filename: str) -> bool:
    """
    Function to load a file into Minio

    Parameters:
    config_dict (dict): dictionary with configuration for Minio
                        object storage
    filename (str): string with the local name of the file. It
                    will be the same on Minio

    Returns:
    response_h (bool): True if file was uploaded, else False

    """
    mclient = Minio(
        config_dict.get("minio_url"),
        access_key=config_dict.get("minio_account"),
        secret_key=config_dict.get("minio_key"),
    )
    try:
        response_h = mclient.fput_object(
            config_dict.get("bucket"),
            filename,
            filename,
            content_type="application/csv",
        )
    except:
        response_h = False
    return bool(response_h)


test_df = pd.read_csv("manual_labels/manual_labels_coicop2018.csv")

test_dict = test_df.to_dict(orient="records")

client = QdrantClient(path="qdrant")

for enc in encoder_list:
    encoder =  SentenceTransformer(enc, trust_remote_code=True)
    # Clean up from products already worked
    done_items = db.search((Item.encoder == enc))
    done_df = pd.DataFrame(done_items)
    unique_list = list(done_df.search_item.unique())
    new_test_dict = [item for item in test_dict if "{} {}".format(item["name"],  item["category"]) not in unique_list ]
    
    for prod in new_test_dict:
        res = []
        res_cos = search_metrics(client, encoder, f"{enc}_cosine", "cosine", prod, enc)
        res_dot = search_metrics(client, encoder, f"{enc}_dot", "dot", prod, enc)
        res.extend(res_cos)
        res.extend(res_dot)
        db.insert_multiple(res)

results_df = pd.DataFrame(db.all())

file_name = "knn_comparison_{}.csv".format(datetime.now().strftime("%Y-%m-%d"))
results_df.to_csv(file_name, index=False)
minio_upload(config_dict=config_dict, filename=file_name)