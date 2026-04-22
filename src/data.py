"""
SPDX-License-Identifier: Apache-2.0
Copyright : JP Morgan Chase & Co

Data Preparation for CyberBench
"""

import json
import os

import pandas as pd

try:
    from ner_data import download_cyner, download_aptner, get_df_cyner, get_df_aptner
    from sum_data import download_cynews, get_df_cynews
    from mc_data import download_secmmlu, download_cyquiz, get_df_secmmlu, get_df_cyquiz
    from tc_data import (
        download_mitre,
        download_cve,
        download_web,
        download_email,
        download_http,
        get_df_mitre,
        get_df_cve,
        get_df_web,
        get_df_email,
        get_df_http,
    )
except ImportError:
    from .ner_data import download_cyner, download_aptner, get_df_cyner, get_df_aptner
    from .sum_data import download_cynews, get_df_cynews
    from .mc_data import download_secmmlu, download_cyquiz, get_df_secmmlu, get_df_cyquiz
    from .tc_data import (
        download_mitre,
        download_cve,
        download_web,
        download_email,
        download_http,
        get_df_mitre,
        get_df_cve,
        get_df_web,
        get_df_email,
        get_df_http,
    )

WEB_CSV_PATH = os.path.join("data", "tc", "web", "dataset_phishing.csv")


if __name__ == "__main__":
    os.makedirs("data", exist_ok=True)

    # Download datasets
    print("Downloading CyNER ...")
    download_cyner()

    print("Downloading APTNER ...")
    download_aptner()

    print("Downloading CyNews ...")
    download_cynews()

    print("Downloading SecMMLU ...")
    download_secmmlu()

    print("Downloading CyQuiz ...")
    download_cyquiz()

    print("Downloading MITRE ...")
    download_mitre()

    print("Downloading CVE ...")
    download_cve()

    print("Downloading Web ...")
    web_available = download_web()
    if not web_available:
        print("Warning: Web dataset could not be downloaded automatically. Continuing without it.")

    print("Downloading Email ...")
    download_email()

    print("Downloading HTTP ...")
    download_http()

    print("All downloading done!")

    # Collect datasets
    dfs = []

    print("Loading CyNER ...")
    df_cyner = get_df_cyner()
    dfs.append(df_cyner)

    print("Loading APTNER ...")
    df_aptner = get_df_aptner()
    dfs.append(df_aptner)

    print("Loading CyNews ...")
    df_cynews = get_df_cynews()
    dfs.append(df_cynews)

    print("Loading SecMMLU ...")
    df_secmmlu = get_df_secmmlu()
    dfs.append(df_secmmlu)

    print("Loading CyQuiz ...")
    df_cyquiz = get_df_cyquiz()
    dfs.append(df_cyquiz)

    print("Loading MITRE ...")
    df_mitre = get_df_mitre()
    dfs.append(df_mitre)

    print("Loading CVE ...")
    df_cve = get_df_cve()
    dfs.append(df_cve)

    if os.path.exists(WEB_CSV_PATH):
        print("Loading Web ...")
        df_web = get_df_web()
        dfs.append(df_web)
    else:
        print("Skipping Web loading because dataset_phishing.csv is not available.")

    print("Loading Email ...")
    df_email = get_df_email()
    dfs.append(df_email)

    print("Loading HTTP ...")
    df_http = get_df_http()
    dfs.append(df_http)

    print("All loading done!")

    # Save the CSV file (for generative models)
    columns = ["task", "dataset", "instruction", "input", "output", "split"]
    df_all = pd.concat([df[columns] for df in dfs], ignore_index=True)

    csv_file_path = os.path.join("data", "cyberbench.csv")
    df_all.to_csv(csv_file_path, index=False)
    print(f"CyberBench data saved to {csv_file_path}")

    # Save the JSON file (for BERT-based models)
    datasets = {df["dataset"].iloc[0]: df.to_dict(orient="records") for df in dfs}
    json_file_path = os.path.join("data", "cyberbench.json")
    with open(json_file_path, "w", encoding="utf-8") as file:
        json.dump(datasets, file, ensure_ascii=False)
    print(f"CyberBench data saved to {json_file_path}")

    # View datasets
    df_count = df_all.value_counts(subset=["task", "dataset", "split"]).unstack(fill_value=0)
    for split_name in ["train", "val", "test"]:
        if split_name not in df_count.columns:
            df_count[split_name] = 0
    df_count = df_count[["train", "val", "test"]]
    df_count["sum"] = df_count.sum(axis="columns")

    print("-" * 50)
    print(df_count)
    print("-" * 50)