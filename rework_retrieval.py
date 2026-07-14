from datetime import datetime

import pandas as pd
from tinydb import TinyDB, Query
from qdrant_client import QdrantClient
from qdrant_client import models as qdrant_models
from sentence_transformers import SentenceTransformer

from minio import Minio
from dotenv import dotenv_values

config_dict = dotenv_values(".env")


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

db = TinyDB('retrieval_results/db.json')
Item = Query()
client = QdrantClient(path="qdrant")

## Rework mistral dot
metric="dot"
enc="intfloat/e5-mistral-7b-instruct"
# enc="intfloat/multilingual-e5-base"
coll = "{}_{}".format(enc, metric)

done_12 = db.search((Item.encoder == enc) & (Item.distance == metric) & (Item.method == "12"))
done_6 = db.search((Item.encoder == enc) & (Item.distance == metric) & ~(Item.method == "12"))

encoder =  SentenceTransformer(enc, trust_remote_code=True)

## Method 12
processed = [d.get("item") for d in done_12]
todo = [d for d in test_dict if d not in processed]
for item in todo:
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

## Method 6+6
processed = [d.get("item") for d in done_6]
todo = [d for d in test_dict if d not in processed]
for item in todo:
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

## Rework mistral cosine
metric="cosine"
# enc="intfloat/multilingual-e5-base"
coll = "{}_{}".format(enc, metric)

done_12 = db.search((Item.encoder == enc) & (Item.distance == metric) & (Item.method == "12"))
done_6 = db.search((Item.encoder == enc) & (Item.distance == metric) & ~(Item.method == "12"))

## Method 12
processed = [d.get("item") for d in done_12]
todo = [d for d in test_dict if d not in processed]
for item in todo:
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

## Method 6+6
processed = [d.get("item") for d in done_6]
todo = [d for d in test_dict if d not in processed]
for item in todo:
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


# Send summary
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