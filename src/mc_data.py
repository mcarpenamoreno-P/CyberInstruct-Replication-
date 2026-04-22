"""
SPDX-License-Identifier: Apache-2.0
Copyright : JP Morgan Chase & Co

Data Preparation for CyberBench Multiple Choice
"""

import os
import urllib.request

import pandas as pd

try:
    from utils import split_dataset, assign_instructions
except ImportError:
    from .utils import split_dataset, assign_instructions

mc_folder = os.path.join("data", "mc")
secmmlu_folder = os.path.join(mc_folder, "secmmlu")
cyquiz_folder = os.path.join(mc_folder, "cyquiz")

CS_DEV_CSV = "computer_security_dev.csv"
CS_VAL_CSV = "computer_security_val.csv"
CS_TEST_CSV = "computer_security_test.csv"

mc_instructions = [
    "Given the cybersecurity question and the following options, which one is the most accurate answer?",
    "In the context of cybersecurity, choose the best response for the given multiple-choice question.",
    "Identify the correct choice for this cybersecurity-related question from the given options.",
    "Out of the available answers, determine the most appropriate solution for this cybersecurity query.",
    "For the following cybersecurity scenario, select the optimal choice from the provided alternatives.",
    "Please assess the cybersecurity question and indicate the most suitable answer among the given choices.",
    "Analyze the given cybersecurity problem and choose the most relevant answer from the list.",
    "Considering the cybersecurity subject matter, pick the most accurate solution for the presented question.",
    "Examine the multiple-choice options and select the correct response for this cybersecurity issue.",
    "In regards to the posed cybersecurity question, identify the best answer from the available selections.",
]


def download_secmmlu():
    """
    Download the SecMMLU files.
    """
    os.makedirs(secmmlu_folder, exist_ok=True)

    for csv_file in [CS_DEV_CSV, CS_VAL_CSV, CS_TEST_CSV]:
        csv_url = (
            "https://raw.githubusercontent.com/zefang-liu/cybersecurity-data/"
            "7d77003027de028cd2daa25e5d03efd07ca09bd7/computer_security/"
            + csv_file
        )
        csv_path = os.path.join(secmmlu_folder, csv_file)
        if not os.path.exists(csv_path):
            urllib.request.urlretrieve(csv_url, csv_path)


def download_cyquiz():
    """
    Download the CyQuiz files.
    """
    os.makedirs(cyquiz_folder, exist_ok=True)

    cyquiz_url = (
        "https://raw.githubusercontent.com/Ebazhanov/linkedin-skill-assessments-quizzes/"
        "ef2690a374567275a4bc24742a5405809e843cf6/cybersecurity/cybersecurity-quiz.md"
    )
    cyquiz_path = os.path.join(cyquiz_folder, "cybersecurity-quiz.md")
    if not os.path.exists(cyquiz_path):
        urllib.request.urlretrieve(cyquiz_url, cyquiz_path)


def combine_choices(df_mc):
    """
    Combine choices of one question.
    """
    df_mc = df_mc.copy()
    df_mc["choices"] = df_mc.apply(
        lambda row: [row["A"], row["B"], row["C"], row["D"]],
        axis="columns",
    )
    df_mc["answer"] = df_mc["answer"].apply(
        lambda answer: ord(answer) - ord("A") if isinstance(answer, str) else answer
    )
    return df_mc[df_mc.columns.drop(["A", "B", "C", "D"])]


def format_choices(row):
    """
    Format choices as:
    Question: ...
    A. ...
    B. ...
    C. ...
    D. ...
    """
    question = row["question"]
    choices = "\n".join(
        [f"{chr(ord('A') + i)}. {choice}" for i, choice in enumerate(row["choices"])]
    )
    return f"Question: {question}\n{choices}"


def load_secmmlu_data(mmlu_folder):
    """
    Load the SecMMLU data.
    """
    mmlu_column_names = ["question", "A", "B", "C", "D", "answer"]

    df_mmlu_cs_dev = pd.read_csv(
        os.path.join(mmlu_folder, CS_DEV_CSV),
        header=None,
        names=mmlu_column_names,
    )
    df_mmlu_cs_val = pd.read_csv(
        os.path.join(mmlu_folder, CS_VAL_CSV),
        header=None,
        names=mmlu_column_names,
    )
    df_mmlu_cs_test = pd.read_csv(
        os.path.join(mmlu_folder, CS_TEST_CSV),
        header=None,
        names=mmlu_column_names,
    )

    df_mmlu_cs_dev["split"] = "train"
    df_mmlu_cs_val["split"] = "val"
    df_mmlu_cs_test["split"] = "test"

    df_secmmlu = pd.concat(
        [df_mmlu_cs_dev, df_mmlu_cs_val, df_mmlu_cs_test],
        ignore_index=True,
    )
    df_secmmlu = combine_choices(df_secmmlu)
    return df_secmmlu


def get_df_secmmlu():
    """
    Get the SecMMLU data.
    """
    df_secmmlu = load_secmmlu_data(secmmlu_folder)
    df_secmmlu["input"] = df_secmmlu.apply(format_choices, axis="columns")
    df_secmmlu["output"] = df_secmmlu["answer"].apply(
        lambda answer: chr(ord("A") + answer)
    )
    df_secmmlu = assign_instructions(
        df_secmmlu,
        outputs=None,
        instructions=mc_instructions,
    )
    df_secmmlu["task"] = "mc"
    df_secmmlu["dataset"] = "secmmlu"
    return df_secmmlu


def drop_questions(data, verbose=False):
    """
    Drop questions without four choices or with multiple-true choices.
    """
    cleaned_data = {"question": [], "choices": [], "answer": []}

    for i, (question, choices, answers) in enumerate(
        zip(data["question"], data["choices"], data["answers"])
    ):
        if len(choices) == 4 and len(answers) == 1:
            cleaned_data["question"].append(question)
            cleaned_data["choices"].append(choices)
            cleaned_data["answer"].append(answers[0])
        elif verbose:
            print(
                f"Drop the question {i}: {len(choices)} choices and {len(answers)} answers"
            )

    return cleaned_data


def load_cyquiz_data(cyquiz_file_path):
    """
    Load the CyQuiz data.
    """
    with open(cyquiz_file_path, "r", encoding="utf-8") as file:
        lines = file.read().splitlines()

    data = {"question": [], "choices": [], "answers": []}
    question_prefix = "#### Q"
    choice_prefix = "- ["
    choice_suffix = "] "

    for line in lines:
        if line.startswith(question_prefix):
            question = ". ".join(line[len(question_prefix):].split(". ")[1:]).strip()
            data["question"].append(question)
            data["choices"].append([])
            data["answers"].append([])
        elif line.startswith(choice_prefix) and data["choices"]:
            if line[len(choice_prefix):len(choice_prefix) + 1] == "x":
                data["answers"][-1].append(len(data["choices"][-1]))
            choice = line[len(choice_prefix) + 1 + len(choice_suffix):]
            data["choices"][-1].append(choice)

    return pd.DataFrame(drop_questions(data))


def get_df_cyquiz():
    """
    Get the CyQuiz data.
    """
    cyquiz_file_path = os.path.join(cyquiz_folder, "cybersecurity-quiz.md")
    df_cyquiz = load_cyquiz_data(cyquiz_file_path)
    df_cyquiz["input"] = df_cyquiz.apply(format_choices, axis="columns")
    df_cyquiz["output"] = df_cyquiz["answer"].apply(
        lambda answer: chr(ord("A") + answer)
    )
    df_cyquiz = assign_instructions(
        df_cyquiz,
        outputs=None,
        instructions=mc_instructions,
    )
    df_cyquiz = split_dataset(df_cyquiz, train_size=5, val_size=23, test_size=100)
    df_cyquiz["task"] = "mc"
    df_cyquiz["dataset"] = "cyquiz"
    return df_cyquiz