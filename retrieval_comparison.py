from datetime import datetime
from collections import Counter
import os

import pandas as pd
from qdrant_client import QdrantClient
from qdrant_client import models as qdrant_models
from sentence_transformers import SentenceTransformer
from tinydb import TinyDB

from minio import Minio
from dotenv import dotenv_values

config_dict = dotenv_values(".env")

METRICS = ["cosine", "dot"]

db = TinyDB("retrieval_results/db.json")

encoder_list = [
#    "jinaai/jina-embeddings-v3",
#    "paraphrase-multilingual-mpnet-base-v2",
#    "sentence-transformers/distiluse-base-multilingual-cased-v1",
#    "sentence-transformers/LaBSE",
#    "Alibaba-NLP/gte-multilingual-base",
#    "Alibaba-NLP/gte-Qwen2-7B-instruct", # to be completed with the latest 350 cases for the dot product
#    "ibm-granite/granite-embedding-278m-multilingual",
#    "intfloat/multilingual-e5-base",  
#    "intfloat/multilingual-e5-large",
#    "RamsesDIIP/me5-large-construction-v2",
    "intfloat/e5-mistral-7b-instruct"
]


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
    for metric in METRICS:
        coll = "{}_{}".format(enc, metric)
        print(coll) 
        for item in test_dict:
            # first 12 results
            search_item = "{} {}".format(item["name"],  item["category"])

            search_result = client.query_points(
                collection_name=coll,
                query=encoder.encode(search_item).tolist(),
                limit=12
            )
        
            search_dict = {
                p.payload.get("title"): p.payload.get("code")
                for p in search_result.points
            }
            correct_option = item["code"] in search_dict.values()
            correct_n = list(search_dict.values()).count(item["code"])
            db.insert({
                "item": item,
                "search_dict": search_dict,
                "method": "12",
                "encoder": enc,
                "distance": metric,
                "correct_option": correct_option,
                "correct_n": correct_n
            })
            # 6 + 6 results
            search_result1 = client.query_points(
                collection_name=coll,
                query=encoder.encode(item["name"]).tolist(),
                limit=6
            )
        
            search_dict1 = {
                p.payload.get("title"): p.payload.get("code")
                for p in search_result1.points
            }

            search_result2 = client.query_points(
                collection_name=coll,
                query=encoder.encode(item["category"]).tolist(),
                limit=6
            )
        
            search_dict2 = {
                p.payload.get("title"): p.payload.get("code")
                for p in search_result2.points
            }
            search_dict1.update(search_dict2)
            correct_option1 = item["code"] in search_dict1.values()
            correct_n1 = list(search_dict1.values()).count(item["code"])
            db.insert({
                "item": item,
                "search_dict": search_dict1,
                "method": "6+6",
                "encoder": enc,
                "distance": metric,
                "correct_option": correct_option1,
                "correct_n": correct_n1
            })

results_df = pd.DataFrame([ 
    {
        "method": item["method"],
        "encoder": item["encoder"],
        "distance": item["distance"],
        "correct_option": item["correct_option"],
        "correct_n": item["correct_n"]
    }
    for item in db.all()
])

# Create a summary dataframe with percentage of correct_option
summary = results_df.groupby(['method', 'encoder', 'distance']).agg(
    correct_percentage=('correct_option', lambda x: x.mean() * 100),  # Convert to percentage
    correct_n_mean=('correct_n', 'mean'),
    correct_n_std=('correct_n', 'std'),
    count=('correct_option', 'count')  # Number of samples in each group
).reset_index()

# Format the summary dataframe
summary['correct_percentage'] = summary['correct_percentage'].round(2)
summary['correct_n_mean'] = summary['correct_n_mean'].round(2)
summary['correct_n_std'] = summary['correct_n_std'].round(2)

# Sort by method, encoder, and distance for better readability
summary = summary.sort_values(['encoder', 'distance', 'method'])

file_name = "retrieval_comparison_summary_{}.csv".format(datetime.now().strftime("%Y-%m-%d"))
summary.to_csv(file_name, index=False)
minio_upload(config_dict=config_dict, filename=file_name)
