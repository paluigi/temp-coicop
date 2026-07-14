from datetime import datetime
import os
from typing import Literal

from ftfy import fix_text
from unidecode import unidecode

import pandas as pd
from tinydb import TinyDB, Query
from tinydb.storages import MemoryStorage

from outlines import models, generate
from openai import AsyncOpenAI
from outlines.models.openai import OpenAIConfig

from minio import Minio
from dotenv import dotenv_values

config_dict = dotenv_values(".env")

# Load COICOP subclass defitions
usecols = ["code","title","intro", "includes", "alsoIncludes", "excludes"]

coicop_df = pd.read_excel(
    "coicop_2018/COICOP_2018_English_structure_edited.xlsx",
    usecols=usecols)
# 1. Select rows where Code has exactly 3 dots
coicop_df = coicop_df[coicop_df['code'].str.count(r'\.') == 3]

# Fix encoding issues
for col in ['intro', 'includes', 'alsoIncludes', 'excludes']:
    coicop_df[col] = coicop_df[col].fillna("")
    coicop_df[col] = coicop_df[col].apply(fix_text)
    coicop_df[col] = coicop_df[col].apply(unidecode)
    coicop_df[col] = coicop_df[col].str.replace("_x000D_", " ")

# 3. Remove classification markers from Description
markers_pattern = r'\s*\((ND|SD|S|D)\)'
coicop_df['title'] = coicop_df['title'].str.replace(markers_pattern, '', regex=True)
subclasses = TinyDB(storage=MemoryStorage)
_ = subclasses.insert_multiple(coicop_df.to_dict("records"))



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
#    "llama3.3:70b",
#    "deepseek-r1:70b",
#    "qwen2.5:32b",
#    "mistral:latest",
    "llama3:8b",
    "deepseek-r1:8b"
]

prompt_start = """You are an expert data curator. You are given a product name and the commercial category where it belongs.
Your task is to find in which COICOP subclass it belongs, chosing from a list of possible options listed below with their description.
You have a residual subclass, titled "None of the above", which is to be used only if no other proposed subclass is suitable.
The options provided are:

{options_prompt}
{last_option_n}. Title: None of the above
Description: This subclass should be only used if none of the other options presented above is suitable for this product.

Your output should only be the exact title of one of the options above, and nothing else.

The product name is: {name}
The commercial category of this product is: {category}
The title of the COICOP subclass where the product belongs is: """

option_title_template = """{n}. Title: {title}
"""

option_description_template = """Description: {intro}
"""

option_inclusion_template = """This subclass includes:
{includes}
{alsoIncludes}

"""
option_exclusion_template = """This subclass excludes:
{excludes}

"""

rac_list = []

for prod in test_dict:
    option_codes = list(set([v for v in prod["search_dict"].values()]))
    option_dicts = subclasses.search(Item.code.one_of(option_codes))
    options_search_dict = {item["title"]: item["code"] for item in option_dicts}
    options_search_dict.update({"None of the above": "99.9.9.9"})

    options_prompt = ""
    for i, option_dict in enumerate(option_dicts):
        options_prompt += option_title_template.format(n=i+1, title=option_dict.get("title"))
        if option_dict.get("intro"):
            options_prompt += option_description_template.format(intro=option_dict.get("intro"))
        if option_dict.get("includes"):
            options_prompt += option_inclusion_template.format(includes=option_dict.get("includes"),
                                                                alsoIncludes=option_dict.get("alsoIncludes", ""))
        if option_dict.get("excludes"):
            options_prompt += option_exclusion_template.format(excludes=option_dict.get("excludes"))

    rac_list.append({
        "name": prod["item"]["name"],
        "category": prod["item"]["category"],
        "manual_code": prod["item"]["code"],
        "options_prompt": options_prompt,
        "last_option_n": len(option_dicts)+1,
        "options_search_dict": options_search_dict,
        'correct_option': prod.get("correct_option"), 
        'correct_n': prod.get("correct_n"),
        }
    )


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


def rac_function(modelname, rac_list, prompt_start, config_dict):
    """Function to retrieve search results and perform
    RAC with a specidic model, plus saving results to Minio"""
    processed_items = []
    errors = []
    # initialize model
    llmclient = AsyncOpenAI(
        api_key="none",
        base_url='http://localhost:11434/v1/',
    )
    config = OpenAIConfig(model=modelname, temperature=0)
    model = models.openai(llmclient, config)
    safe_modelname = modelname.replace(".", "").replace(":", "-")
    # Loop
    for prod in rac_list:
        search_dict = prod.get("options_search_dict")
        options = list(search_dict.keys()) + ["none of the above"]
        generator = generate.choice(model, options)
        try:
            res = generator(prompt_start.format(**prod))
            processed_items.append({**prod, 
                                    'correct_option': prod.get("correct_option"), 
                                    'correct_n': prod.get("correct_n"),
                                    "llm" : modelname,
                                    "prediction": search_dict.get(res)})
        except:
            errors.append(prod)
            
    # Save results to disk and to Minio
    if len(processed_items)>0:
        results_df = pd.DataFrame(processed_items)
        results_df = results_df.drop(columns=["options_prompt", "options_search_dict", "last_option_n"])
        results_df["match"] = results_df["prediction"] == results_df["manual_code"]
        filename = "rac_outlines_v2_{}_{}.csv".format(safe_modelname, datetime.now().strftime("%Y-%m-%d_%H%M%S"))
        results_df.to_csv(filename, index=False)
        minio_upload(config_dict=config_dict, filename=filename)

    if len(errors)>0:
        errors_df = pd.DataFrame(errors)
        errorname = "errors_outlines_v2_{}_{}.csv".format(safe_modelname, datetime.now().strftime("%Y-%m-%d_%H%M%S"))
        errors_df.to_csv(errorname, index=False)
        minio_upload(config_dict=config_dict, filename=errorname)
    
    
    return True


for modelname in model_list:
    _ = rac_function(modelname, rac_list, prompt_start, config_dict)

