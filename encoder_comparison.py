from datetime import datetime

import pandas as pd
from qdrant_client import models, QdrantClient
from sentence_transformers import SentenceTransformer

from minio import Minio
from dotenv import dotenv_values

config_dict = dotenv_values(".env")

METRICS = ["cosine", "dot"]

NUM = 9

encoder_list = [
    #"jinaai/jina-embeddings-v3",
    #"paraphrase-multilingual-mpnet-base-v2",
    #"sentence-transformers/distiluse-base-multilingual-cased-v1",
    #"sentence-transformers/LaBSE",
    #"Alibaba-NLP/gte-multilingual-base",
    #"Alibaba-NLP/gte-Qwen2-7B-instruct",
    #"ibm-granite/granite-embedding-278m-multilingual",
    #"intfloat/multilingual-e5-base",
    #"intfloat/multilingual-e5-large",
    #"RamsesDIIP/me5-large-construction-v2",
    "intfloat/e5-mistral-7b-instruct"
]

def create_collections(client, encoder, distance_function, collection_name, data_dict):
    collection_s = f"{collection_name}_{distance_function}"

    if (distance_function == METRICS[0]):
        distance_model = models.Distance.COSINE
    else:
        distance_model = models.Distance.DOT

    client.create_collection(
        collection_name=collection_s,
        vectors_config=models.VectorParams(
            size=encoder.get_sentence_embedding_dimension(),  # Vector size is defined by used model
            distance=distance_model
        ),
    )
    client.upload_points(
        collection_name=collection_s,
        points=[
            models.PointStruct(
                id=idx+1, vector=encoder.encode(doc["title"]).tolist(), payload=doc
            )
            for idx, doc in enumerate(data_dict)
        ],
    )

    return print("Created collection {} at {}".format(
        collection_s,  datetime.now().strftime("%Y-%m-%d %H:%M:%S")))


def search_metrics(client, encoder, collection_name, metric, cat, NUM):
    category_result = []

    hits = client.search(
        collection_name=collection_name,
        query_vector=encoder.encode(cat).tolist(),
        limit=NUM
    )
    for hit in hits:
        category_result.append({
            **hit.payload,
            "score": hit.score,
            "name": cat
        })

    result_df = pd.DataFrame(category_result)

    highest = add_methodology(highest_score(result_df), f"{metric}_highest")
    simple = add_methodology(simple_count(result_df), f"{metric}_simple")
    weighted = add_methodology(weighted_count(result_df), f"{metric}_weighted")

    long_df = pd.concat([highest, simple, weighted])

    return long_df


def rename_columns(df: pd.DataFrame, suffix:str) -> pd.DataFrame:
    """Function to add a suffix to specific columns
    """
    df = df.rename(columns={
        "code": f"code_{suffix}",
        "score": f"score_{suffix}",
        "count": f"score_{suffix}"})
    return df


def add_methodology(df: pd.DataFrame, methodology:str) -> pd.DataFrame:
    """Function to add a methodology column and harmonize column names
    """
    df = df.rename(columns={"count": "score"})
    df["method"] = methodology
    return df


def weighted_count(df: pd.DataFrame) -> pd.DataFrame:
    """Function to select the classification with the highest count
    of classification weighted by scores
    """
    grouped = df.groupby(['name', 'code']).sum("score").reset_index()
    result = grouped.loc[grouped.groupby('name')['score'].idxmax()]
    return result


def simple_count(df: pd.DataFrame) -> pd.DataFrame:
    """Function to select the classification with the highest count
    of classification
    """
    counted = df.groupby(['name', 'code']).size().reset_index(name='count')
    # Find the 'code' with the highest count for each 'category'
    result = counted.loc[counted.groupby('name')['count'].idxmax()]
    return result


def highest_score(df: pd.DataFrame) -> pd.DataFrame:
    """Function to select the classification with the highest score
    """
    # Find the index of the maximum score for each category
    idx = df.groupby('name')['score'].idxmax()

    # Select the rows corresponding to these indices
    result = df.loc[idx]
    return result


def majority_vote(df: pd.DataFrame) -> pd.DataFrame:
    counted = df.groupby(['name', 'code']).size().reset_index(name='count')
    # Find the 'code' with the highest count for each 'category'
    result = counted.loc[counted.groupby('name')['count'].idxmax()]

    result["confidence"] = result["count"] / 6

    return result.to_dict(orient='records')[0]


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


grid_df = pd.read_csv("results/consolidated_coicop2018_2025-03-16.csv")
test_df = pd.read_csv("manual_labels/manual_labels_coicop2018.csv")

test_dict = test_df.to_dict(orient="records")
grid_dict = grid_df.to_dict(orient="records")

client = QdrantClient(path="qdrant")

for enc in encoder_list:
    encoder =  SentenceTransformer(enc, trust_remote_code=True)
    for metric in METRICS:
        create_collections(client, encoder, metric, enc, grid_dict)

result_list = []
for enc in encoder_list:
    encoder =  SentenceTransformer(enc, trust_remote_code=True)
    for prod in test_dict:
        search_item = "{} {}".format(prod["name"],  prod["category"])
        results_cosine = search_metrics(client, encoder, f"{enc}_cosine", "cosine", search_item, NUM)
        results_dot = search_metrics(client, encoder, f"{enc}_dot", "dot", search_item, NUM)
        results_concat = pd.concat([results_cosine, results_dot])
        result = majority_vote(results_concat)
        result_list.append({
            **result,
            "name" : prod["name"],
            "category":  prod["category"], 
            "manual_code": prod["code"],
            "encoder": enc})

res_df = pd.DataFrame(result_list)
res_df["correct"] = res_df["code"] == res_df["manual_code"]
file_name = "encoder_comparison_{}_{}.csv".format(NUM, datetime.now().strftime("%Y-%m-%d"))
res_df.to_csv(file_name, index=False)
minio_upload(config_dict=config_dict, filename=file_name)

