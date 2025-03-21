from datetime import datetime
from collections import Counter
import os

import pandas as pd
from qdrant_client import QdrantClient
from qdrant_client import models as qdrant_models
from sentence_transformers import SentenceTransformer

from outlines import models, generate
from openai import AsyncOpenAI
from outlines.models.openai import OpenAIConfig

from minio import Minio
from dotenv import dotenv_values

config = dotenv_values(".env")

def count_value_occurrences(my_list):
    """Counts the occurrences of each value in a list and returns a dictionary.

    Args:
        my_list: The input list.

    Returns:
        A dictionary where keys are the unique values in the list (in the order they first appear)
        and values are the number of times each value appears.
    """

    # Use Counter for efficient counting:
    value_counts = Counter(my_list)

    # Preserve order of first appearance using a list comprehension:
    ordered_keys = sorted(list(set(my_list)))
    
    # Create the ordered dictionary:
    ordered_dict = {key: value_counts[key] for key in ordered_keys}

    return ordered_dict


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



client = QdrantClient(":memory:")

encoder = SentenceTransformer("paraphrase-multilingual-mpnet-base-v2")

client.create_collection(
    collection_name="coicop2018",
    vectors_config=qdrant_models.VectorParams(
        size=encoder.get_sentence_embedding_dimension(),  # Vector size is defined by used model
        distance=qdrant_models.Distance.DOT, # Try with Cosine as well
    ),
)

client.upload_points(
    collection_name="coicop2018",
    points=[
        qdrant_models.PointStruct(
            id=idx+1, vector=encoder.encode(doc["title"]).tolist(), payload=doc
        )
        for idx, doc in enumerate(grid_dict)
    ],
)

prompt_template = """You are an expert data curator. You are given a product name and the commercial category where it belongs.
Your task is to find the most similar match from a list of possible options. 
If no option is suitable, you should output "none of the above". The options are:

1. {options[0]}
2. {options[1]}
3. {options[2]}
4. {options[3]}
5. {options[4]}
6. {options[5]}
7. {options[6]}
8. {options[7]}
9. {options[8]}
10. {options[9]}
11. {options[10]}
12. {options[11]}
13. none of the above

Your output should only be the exact text of one of the options above, and nothing else.

The product name is: {name}
The commercial category where the product belongs is: {category}
The most similar option is:"""


# Llama3.3 70b
modelname = "llama3.3:70b"
safe_modelname = modelname.replace(".", "").replace(":", "-")
llmclient = AsyncOpenAI(
    api_key="none",
    base_url='http://localhost:11434/v1/',
)
config = OpenAIConfig(model=modelname, temperature=0)
model = models.openai(llmclient, config)

processed_items = []
for i, item in enumerate(test_dict):
    if i % 500 == 0:
        print(f"Processing item {i+1} out of {len(test_dict)}")
    search_result = client.query_points(
        collection_name="coicop2018",
        query=encoder.encode("{} {}".format(item["name"], item["category"])).tolist(),
        limit=12
    )

    search_dict = {
        p.payload.get("title"): p.payload.get("code")
        for p in search_result.points
    }
    correct_option = item["code"] in search_dict.values()
    correct_n = list(search_dict.values()).count(item["code"])
    options = list(search_dict.keys()) + ["none of the above"]
    generator = generate.choice(model, options)
    res = generator(prompt_template.format(options=options, name=item["name"], category=item["category"]))
    processed_items.append({**item, "prediction": search_dict.get(res), "correct_option": correct_option, "correct_n": correct_n})
    
results_df = pd.DataFrame(processed_items)
results_df["match"] = results_df["prediction"] == results_df["code"]
filename = "rac_coicop2018_{}_{}.csv".format(safe_modelname, datetime.now().strftime("%Y-%m-%d_%H%M%S"))

results_df.to_csv(filename, index=False)

minio_upload(config_dict=config, filename=filename)


# Qwen 2.5 32b
modelname = "qwen2.5:32b"
safe_modelname = modelname.replace(".", "").replace(":", "-")
llmclient = AsyncOpenAI(
    api_key="none",
    base_url='http://localhost:11434/v1/',
)
config = OpenAIConfig(model=modelname, temperature=0)
model = models.openai(llmclient, config)

processed_items = []
for i, item in enumerate(test_dict):
    if i % 500 == 0:
        print(f"Processing item {i+1} out of {len(test_dict)}")
    search_result = client.query_points(
        collection_name="coicop2018",
        query=encoder.encode("{} {}".format(item["name"], item["category"])).tolist(),
        limit=12
    )

    search_dict = {
        p.payload.get("title"): p.payload.get("code")
        for p in search_result.points
    }
    correct_option = item["code"] in search_dict.values()
    correct_n = list(search_dict.values()).count(item["code"])
    options = list(search_dict.keys()) + ["none of the above"]
    generator = generate.choice(model, options)
    res = generator(prompt_template.format(options=options, name=item["name"], category=item["category"]))
    processed_items.append({**item, "prediction": search_dict.get(res), "correct_option": correct_option, "correct_n": correct_n})
    
results_df = pd.DataFrame(processed_items)
results_df["match"] = results_df["prediction"] == results_df["code"]
filename = "rac_coicop2018_{}_{}.csv".format(safe_modelname, datetime.now().strftime("%Y-%m-%d_%H%M%S"))

results_df.to_csv(filename, index=False)

minio_upload(config_dict=config, filename=filename)


# Deepseek 70b
modelname = "deepseek-r1:70b"
safe_modelname = modelname.replace(".", "").replace(":", "-")
llmclient = AsyncOpenAI(
    api_key="none",
    base_url='http://localhost:11434/v1/',
)
config = OpenAIConfig(model=modelname, temperature=0)
model = models.openai(llmclient, config)

processed_items = []
for i, item in enumerate(test_dict):
    if i % 500 == 0:
        print(f"Processing item {i+1} out of {len(test_dict)}")
    search_result = client.query_points(
        collection_name="coicop2018",
        query=encoder.encode("{} {}".format(item["name"], item["category"])).tolist(),
        limit=12
    )

    search_dict = {
        p.payload.get("title"): p.payload.get("code")
        for p in search_result.points
    }
    correct_option = item["code"] in search_dict.values()
    correct_n = list(search_dict.values()).count(item["code"])
    options = list(search_dict.keys()) + ["none of the above"]
    generator = generate.choice(model, options)
    res = generator(prompt_template.format(options=options, name=item["name"], category=item["category"]))
    processed_items.append({**item, "prediction": search_dict.get(res), "correct_option": correct_option, "correct_n": correct_n})
    
results_df = pd.DataFrame(processed_items)
results_df["match"] = results_df["prediction"] == results_df["code"]
filename = "rac_coicop2018_{}_{}.csv".format(safe_modelname, datetime.now().strftime("%Y-%m-%d_%H%M%S"))

results_df.to_csv(filename, index=False)

minio_upload(config_dict=config, filename=filename)

