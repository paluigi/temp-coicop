from datetime import datetime
import os
from typing import Literal

import pandas as pd
from tinydb import TinyDB, Query
from pydantic import BaseModel

from pydantic_ai import Agent
from pydantic_ai.models.openai import OpenAIModel
from pydantic_ai.providers.openai import OpenAIProvider

from minio import Minio
from dotenv import dotenv_values

config_dict = dotenv_values(".env")

db = TinyDB("retrieval_results/db.json")

# Use the best encoder, metric, and method from retrieval comparison results
ENCODER = "Alibaba-NLP/gte-Qwen2-7B-instruct"
METRIC = "dot"
METHOD = "12"
Item = Query()

test_dict = db.search(
    (Item.encoder == ENCODER) & 
    (Item.distance == METRIC) & 
    (Item.method == METHOD))

model_list = [
    "llama3.3:70b",
    "deepseek-r1:70b",
    "qwen2.5:32b",
    "mistral:latest",
    "llama3:8b",
    "deepseek-r1:8b"
]

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
The most similar option is: """

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


def rac_function(modelname, test_dict, prompt_template, config_dict):
    """Function to retrieve search results and perform
    RAC with a specidic model, plus saving results to Minio"""
    processed_items = []
    errors = []
    # initialize model
    safe_modelname = modelname.replace(".", "").replace(":", "-")
    ollama_model = OpenAIModel(
        model_name=modelname, provider=OpenAIProvider(base_url='http://localhost:11434/v1')
    )

    # Loop
    for prod in test_dict:
        search_dict = prod.get("search_dict")
        options = list(search_dict.keys()) + ["none of the above"]
        class Choice(BaseModel):
            option: Literal[tuple(options)]
        agent = Agent(ollama_model, result_type=Choice, retries=3,)
        try:
            res = agent.run_sync(prompt_template.format(
                options=options, 
                name=prod["item"]["name"], 
                category=prod["item"]["category"]), model_settings={'temperature': 0.0})
            processed_items.append({**prod.get("item"), 
                                    'correct_option': prod.get("correct_option"), 
                                    'correct_n': prod.get("correct_n"),
                                    "llm" : modelname,
                                    "prediction": search_dict.get(res.data.model_dump().get("option"))})
        except:
            errors.append(prod)
            
    # Save results to disk and to Minio
    if len(processed_items)>0:
        results_df = pd.DataFrame(processed_items)
        results_df["match"] = results_df["prediction"] == results_df["code"]
        filename = "rac_pydantic_{}_{}.csv".format(safe_modelname, datetime.now().strftime("%Y-%m-%d_%H%M%S"))
        results_df.to_csv(filename, index=False)
        minio_upload(config_dict=config_dict, filename=filename)

    if len(errors)>0:
        errors_df = pd.DataFrame(errors)
        errorname = "errors_pydantic_{}_{}.csv".format(safe_modelname, datetime.now().strftime("%Y-%m-%d_%H%M%S"))
        errors_df.to_csv(errorname, index=False)
        minio_upload(config_dict=config_dict, filename=errorname)
    
    
    return True


for modelname in model_list:
    _ = rac_function(modelname, test_dict, prompt_template, config_dict)

