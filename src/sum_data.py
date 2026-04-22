"""
SPDX-License-Identifier: Apache-2.0
Copyright : JP Morgan Chase & Co

Data Preparation for CyberBench Summarization
"""

import os
import urllib.request

import pandas as pd

try:
    from utils import split_dataset, assign_instructions
except ImportError:
    from .utils import split_dataset, assign_instructions

sum_folder = os.path.join("data", "sum")
cynews_folder = os.path.join(sum_folder, "cynews")

sum_instructions = [
    "Given the following text, generate a concise and informative title that captures the main theme of the cybersecurity content.",
    "Summarize the main points of this cybersecurity-related text into a single, catchy headline.",
    "Create a title that encapsulates the key findings and implications of this cybersecurity-related text.",
    "Based on the details of this text related to cybersecurity, what would be an appropriate title that highlights the main event and its impact?",
    "Generate a title that summarizes the main points discussed in this cybersecurity-related text.",
    "What would be a fitting headline for this text discussing recent advancements or incidents in cybersecurity?",
    "Analyze the following cybersecurity-related text and generate a title that accurately reflects its main theme.",
    "From the given cybersecurity text, create a headline that encapsulates the primary focus.",
    "Read the following cybersecurity-related text. What would be a suitable title that summarizes its key points?",
    "Given this text on a cybersecurity issue, generate a headline that highlights the main concern and its implications.",
]


def download_cynews():
    """
    Download the CyNews files.
    """
    os.makedirs(cynews_folder, exist_ok=True)

    cynews_url = (
        "https://github.com/cypher-07/Cybersecurity-News-Article-Dataset/raw/"
        "31c6eb7e3121686f8f00148e49fcaac43a4305c2/TheHackerNews_Dataset.xlsx"
    )
    cynews_path = os.path.join(cynews_folder, "TheHackerNews_Dataset.xlsx")
    cynews_csv_path = os.path.join(cynews_folder, "TheHackerNews_Dataset.csv")

    if not os.path.exists(cynews_csv_path):
        if not os.path.exists(cynews_path):
            urllib.request.urlretrieve(cynews_url, cynews_path)
        pd.read_excel(cynews_path).to_csv(cynews_csv_path, index=False, header=True)
        if os.path.exists(cynews_path):
            os.remove(cynews_path)


def get_df_cynews():
    """
    Get the CyNews data.
    """
    cynews_file_path = os.path.join(cynews_folder, "TheHackerNews_Dataset.csv")
    df_cynews = pd.read_csv(cynews_file_path).rename(
        columns={"Article": "input", "Title": "output"}
    )[["input", "output"]]

    df_cynews["input"] = df_cynews["input"].str.replace("_x0081_", "", regex=False)
    df_cynews = assign_instructions(
        df_cynews,
        outputs=False,
        instructions=sum_instructions,
    )
    df_cynews = split_dataset(df_cynews)
    df_cynews["task"] = "sum"
    df_cynews["dataset"] = "cynews"
    return df_cynews