"""
SPDX-License-Identifier: Apache-2.0
Copyright : JP Morgan Chase & Co

Data Preparation for CyberBench Named-Entity Recognition
"""

import os
import re
import json
import urllib.request
from collections import defaultdict

import pandas as pd
from nltk.tokenize.treebank import TreebankWordDetokenizer

try:
    from utils import split_dataset, assign_instructions
except ImportError:
    from .utils import split_dataset, assign_instructions


ner_folder = os.path.join("data", "ner")
cyner_folder = os.path.join(ner_folder, "cyner")
aptner_folder = os.path.join(ner_folder, "aptner")

ner_instructions = [
    "Identify the entity types {entity_types} in the given sentence related to cybersecurity. "
    "Entity definitions are as follows: {entity_definitions}. "
    "Extract the entities and present them in a JSON format that follows the structure {output_format}. "
    "Do not generate any extra entities outside the given sentence.",
    "Analyze the provided sentence to find entities related to the cybersecurity domain "
    "that correspond to the following entity types: {entity_types}. "
    "The definitions of these entities are: {entity_definitions}. "
    "Extract the identified entities and format them into a JSON object using the format: {output_format}. "
    "Do not include entities that are not present in the sentence.",
    "Examine the sentence for entities that belong to these cybersecurity-related entity types: {entity_types}. "
    "Use the following definitions to help you identify the entities: {entity_definitions}. "
    "Once you have found them, organize the entities into a JSON object using this format: {output_format}. "
    "Do not create any additional entities that are not in the sentence.",
    "In the given sentence, "
    "search for entities that match the cybersecurity-related entity types listed here: {entity_types}. "
    "Refer to these definitions for clarification: {entity_definitions}. "
    "Present the identified entities in a JSON object, adhering to the structure: {output_format}. "
    "Do not generate entities that are not part of the sentence.",
    "Look for entities in the sentence that fit the following cybersecurity-related entity types: {entity_types}. "
    "Use these definitions to help with identification: {entity_definitions}. "
    "Once you have found the entities, compile them into a JSON object using the format: {output_format}. "
    "Do not add any entities that are not present in the sentence.",
    "Within the provided sentence, "
    "find entities that correspond to these cybersecurity domain entity types: {entity_types}. "
    "To assist you, here are the definitions of the entities: {entity_definitions}. "
    "Extract and arrange the entities in a JSON object according to this format: {output_format}. "
    "Do not include entities that are not part of the sentence.",
    "Assess the sentence for entities that align with these cybersecurity entity types: {entity_types}. "
    "Use the following definitions for guidance: {entity_definitions}. "
    "Organize the identified entities into a JSON object, adhering to the format: {output_format}. "
    "Do not generate entities that are not found in the sentence.",
    "In the sentence provided, "
    "search for entities that belong to these cybersecurity-related entity types: {entity_types}. "
    "The definitions of these entities are as follows: {entity_definitions}. "
    "Compile the entities you find into a JSON object using this format: {output_format}. "
    "Do not create any additional entities that are not in the sentence.",
    "Examine the given sentence to identify entities "
    "that fall under these cybersecurity-related entity types: {entity_types}. "
    "Refer to these definitions for assistance: {entity_definitions}. "
    "Once identified, present the entities in a JSON object that follows this format: {output_format}. "
    "Do not include entities that are not present in the sentence.",
    "Detect entities in the provided sentence that match the following cybersecurity entity types: {entity_types}. "
    "Utilize these definitions for clarification: {entity_definitions}. "
    "Extract the identified entities and arrange them into a JSON object according to the format: {output_format}. "
    "Do not generate any extra entities outside of the given sentence.",
]

ner_output_format = '{"entity type": ["entity 1", "entity 2", ...]}'
detokenizer = TreebankWordDetokenizer()


def read_text_file_with_fallbacks(file_path):
    """
    Read a text file trying several common encodings.
    This avoids Windows cp1252 decoding errors.
    """
    encodings = ["utf-8", "utf-8-sig", "latin-1", "cp1252"]

    for encoding in encodings:
        try:
            with open(file_path, "r", encoding=encoding) as file:
                return file.read().splitlines()
        except UnicodeDecodeError:
            pass

    raise ValueError(f"Could not decode file: {file_path}")


def filter_matching_token_tag_lengths(df):
    """
    Keep only rows where tokens and tags have the same length.
    """
    return df[df["tokens"].apply(len) == df["tags"].apply(len)].copy()


def format_ner_output_from_lists(tokens, tags):
    """
    Format the NER output as a JSON string from token/tag lists.
    """
    output = defaultdict(list)

    for token, tag in zip(tokens, tags):
        if tag.startswith("B-"):
            entity_type = tag[2:]
            output[entity_type].append([token])

        elif tag.startswith("I-"):
            entity_type = tag[2:]

            if len(output[entity_type]) == 0:
                output[entity_type].append([token])
            else:
                output[entity_type][-1].append(token)

    for entity_type, entities in output.items():
        output[entity_type] = list(
            dict.fromkeys(detokenizer.detokenize(entity) for entity in entities)
        )

    return json.dumps(output)


def format_ner_output(row):
    """
    Backward-compatible wrapper.
    """
    return format_ner_output_from_lists(row.tokens, row.tags)


def download_cyner():
    """
    Download the CyNER files.
    """
    os.makedirs(cyner_folder, exist_ok=True)

    cyner_train_url = (
        "https://raw.githubusercontent.com/aiforsec/CyNER/"
        "37aff53bd7235605638320e0c06de2fb82847070/dataset/mitre/train.txt"
    )
    cyner_val_url = (
        "https://raw.githubusercontent.com/aiforsec/CyNER/"
        "37aff53bd7235605638320e0c06de2fb82847070/dataset/mitre/valid.txt"
    )
    cyner_test_url = (
        "https://raw.githubusercontent.com/aiforsec/CyNER/"
        "37aff53bd7235605638320e0c06de2fb82847070/dataset/mitre/test.txt"
    )

    cyner_train_path = os.path.join(cyner_folder, "train.txt")
    cyner_val_path = os.path.join(cyner_folder, "valid.txt")
    cyner_test_path = os.path.join(cyner_folder, "test.txt")

    if not os.path.exists(cyner_train_path):
        urllib.request.urlretrieve(cyner_train_url, cyner_train_path)
    if not os.path.exists(cyner_val_path):
        urllib.request.urlretrieve(cyner_val_url, cyner_val_path)
    if not os.path.exists(cyner_test_path):
        urllib.request.urlretrieve(cyner_test_url, cyner_test_path)


def download_aptner():
    """
    Download the APTNER files.
    """
    os.makedirs(aptner_folder, exist_ok=True)

    aptner_train_url = (
        "https://raw.githubusercontent.com/wangxuren/APTNER/"
        "b730e6deccd583abdaab1ab75a06391533e14ada/APTNERtrain.txt"
    )
    aptner_dev_url = (
        "https://raw.githubusercontent.com/wangxuren/APTNER/"
        "b730e6deccd583abdaab1ab75a06391533e14ada/APTNERdev.txt"
    )
    aptner_test_url = (
        "https://raw.githubusercontent.com/wangxuren/APTNER/"
        "b730e6deccd583abdaab1ab75a06391533e14ada/APTNERtest.txt"
    )

    aptner_train_path = os.path.join(aptner_folder, "APTNERtrain.txt")
    aptner_dev_path = os.path.join(aptner_folder, "APTNERdev.txt")
    aptner_test_path = os.path.join(aptner_folder, "APTNERtest.txt")

    if not os.path.exists(aptner_train_path):
        urllib.request.urlretrieve(aptner_train_url, aptner_train_path)
    if not os.path.exists(aptner_dev_path):
        urllib.request.urlretrieve(aptner_dev_url, aptner_dev_path)
    if not os.path.exists(aptner_test_path):
        urllib.request.urlretrieve(aptner_test_url, aptner_test_path)


def split_ner_sentences(tokens, tags):
    """
    Split sentences in NER data.
    """
    split_tokens = []
    split_tags = []

    for _tokens, _tags in zip(tokens, tags):
        split_tokens.append([])
        split_tags.append([])
        in_quotation = False

        for token, tag in zip(_tokens, _tags):
            if not token:
                continue

            if (
                len(split_tokens[-1]) > 0
                and split_tokens[-1][-1] == "."
                and token[0].isupper()
            ) or (
                len(split_tokens[-1]) > 1
                and split_tokens[-1][-2] == "."
                and split_tokens[-1][-1] == '"'
                and not in_quotation
            ):
                split_tokens.append([])
                split_tags.append([])

            split_tokens[-1].append(token)
            split_tags[-1].append(tag)

            if token == '"':
                in_quotation = not in_quotation

    return split_tokens, split_tags


def parse_cyner_line(line, tokens, tags):
    """
    Parse one CyNER line.
    """
    if line == "" and tokens[-1] != []:
        tokens.append([])
        tags.append([])

    elif line != "":
        token, tag = line.split("\t")

        if tag.startswith("I-"):
            if len(tags[-1]) == 0:
                if len(tags) > 1 and len(tags[-2]) > 0 and tags[-2][-1][2:] == tag[2:]:
                    tokens.pop(-1)
                    tags.pop(-1)
                else:
                    tag = "O"
            elif tags[-1][-1][2:] != tag[2:]:
                tag = "O"

        tokens[-1].append(token)
        tags[-1].append(tag)

    return tokens, tags


def load_cyner_data(cyner_folder, split):
    """
    Load CyNER data.
    """
    split_path = os.path.join(cyner_folder, f"{split}.txt")
    lines = read_text_file_with_fallbacks(split_path)

    tokens = [[]]
    tags = [[]]

    for line in lines:
        tokens, tags = parse_cyner_line(line, tokens, tags)

    tokens, tags = split_ner_sentences(tokens, tags)

    df_split = pd.DataFrame(
        {
            "tokens": tokens,
            "tags": tags,
            "split": split if split != "valid" else "val",
        }
    )

    return df_split


cyner_entity_definitions = {
    "Malware": "viruses, trojans, ransomware, etc.",
    "System": "operating systems (e.g., Android, Windows), software (e.g., Adobe flash player, Skype), and hardware",
    "Organization": "firms, companies, groups, manufacturers, developers, etc.",
    "Indicator": "indicators of compromise, domain names, URLs, IP addresses, filenames, hashes, emails, port numbers, etc.",
    "Vulnerability": "CVE IDs (e.g., CVE-2012-2825) and mentions of exploits (e.g., master key vulnerability)",
}


def get_df_cyner():
    """
    Get the CyNER data.
    """
    df_cyner = pd.concat(
        [load_cyner_data(cyner_folder, split) for split in ["train", "valid", "test"]],
        ignore_index=True,
    )

    df_cyner["text"] = df_cyner.tokens.apply(lambda tokens: " ".join(tokens))
    df_cyner = df_cyner[df_cyner.text.str.len() > 1].drop_duplicates(
        subset=["text"], keep="first", ignore_index=True
    )[df_cyner.columns.drop("text")]

    df_cyner = filter_matching_token_tag_lengths(df_cyner)

    cyner_tags = ["O"] + [
        f"{prefix}-{entity_name}"
        for entity_name in cyner_entity_definitions.keys()
        for prefix in ["B", "I"]
    ]
    cyner_label2id = {label: idx for idx, label in enumerate(cyner_tags)}

    df_cyner["labels"] = df_cyner.tags.apply(
        lambda tags: [cyner_label2id[tag] for tag in tags]
    )
    df_cyner["input"] = df_cyner.tokens.apply(detokenizer.detokenize)
    df_cyner["output"] = [
        format_ner_output_from_lists(tokens, tags)
        for tokens, tags in zip(df_cyner["tokens"], df_cyner["tags"])
    ]

    df_cyner = assign_instructions(
        df_cyner,
        outputs=False,
        instructions=ner_instructions,
    )

    df_cyner["task"] = "ner"
    df_cyner["dataset"] = "cyner"

    df_cyner["instruction"] = df_cyner["instruction"].apply(
        lambda instruction: instruction.format(
            entity_types=", ".join(cyner_entity_definitions.keys()),
            entity_definitions="; ".join(
                [f"{key}: {value}" for key, value in cyner_entity_definitions.items()]
            ),
            output_format=ner_output_format,
        )
    )

    return df_cyner


aptner_line_mapper = {
    " S-MALon I-ACT": "on I-ACT",
    " S-MALvictim I-ACT": "victim I-ACT",
    "0c458dfe0a2a01ab300c857fdc3373b75fbb8ccfa23d16eff0d6ab888a1a28f6 O":
    "0c458dfe0a2a01ab300c857fdc3373b75fbb8ccfa23d16eff0d6ab888a1a28f6 S-SHA2",
    " S-SHA2init S-FILE": "init S-FILE",
    "93ce211a71867017723cd78969aa4cac9d21c3d8f72c96ee3e1b2712c0eea494 O":
    "93ce211a71867017723cd78969aa4cac9d21c3d8f72c96ee3e1b2712c0eea494 S-SHA2",
    " S-SHA2init2 S-FILE": "init2 S-FILE",
    "firewall O": "firewall S-TOOL",
    " S-TOOLwill O": "will O",
    "：O": "： O",
    ":M B-TOOL": ": O",
    "icrosoft I-TOOL": "Microsoft B-TOOL",
    "– I-TOO B-IDTYL": "– I-TOOL",
    "2 E-IDTY003 I-TOOL": "2003 I-TOOL",
    "Troja S-MALn E-MAL S-MAL": "Trojan E-MAL",
    "side-l S-TOOLoading E-TOOL": "side-loading E-TOOL",
    "Office B-IDTY I-TOOL E-IDTY": "Office I-TOOL",
    "docume S-TOOLnts E-TOOL": "documents E-TOOL",
    "optimiza S-IDTYtion I-TOOL": "optimization I-TOOL",
    "CobaltGoblin S-APT/Carbanak S-APT/EmpireMonkey S-APT":
    "CobaltGoblin S-APT\t/ O\tCarbanak S-APT\t/ O\tEmpireMonkey S-APT",
    "WSDL I-TOOL B-VULNAME": "WSDL B-VULNAME",
    "IP S-PROT B-TOOL": "IP B-TOOL",
    "botnet S-ACTs E-TOOL": "botnets E-TOOL",
    "C2 S-TOOL B-TOOL": "C2 B-TOOL",
    "Georgia S-LOC-NATO S-IDTY": "Georgia-NATO S-LOC",
}


def select_aptner_tags(token, token_tags, last_tag):
    """
    Select an APTNER tag among several candidates based on the last tag.
    """
    assert all(token_tag.isupper() for token_tag in token_tags)

    if last_tag == "O":
        for prefix in ["B-", "S-"]:
            for token_tag in token_tags:
                if token_tag.startswith(prefix):
                    return token_tag
        return token_tags[0]

    entity_type = last_tag[2:]
    if "I-" + entity_type in token_tags:
        return "I-" + entity_type
    if "E-" + entity_type in token_tags:
        return "E-" + entity_type
    return token_tags[0]


def process_aptner_lines(file_lines, aptner_line_mapper=aptner_line_mapper):
    """
    Process APTNER lines by mapping and splitting lines.
    """
    lines = []

    for line in file_lines:
        if line in aptner_line_mapper:
            line = aptner_line_mapper[line]

        line = line.strip()

        if "\xa0" in line:
            split_lines = line.split("\xa0")
            if len(split_lines) > 2:
                lines.extend(split_lines)
            else:
                lines.append(line.replace("\xa0", " "))
        else:
            split_lines = line.split("\t")
            if len(split_lines) > 1:
                lines.extend(split_lines)
            else:
                lines.append(line)

    return lines


def fix_aptner_tokens(line_tokens, tokens, tags, line=None, split=None):
    """
    Fix APTNER tokens.
    """
    second_tokens = re.split(r"([/:])", line_tokens[1])

    if len(second_tokens) >= 3 and second_tokens[0].isupper():
        tokens[-1].extend([line_tokens[0], second_tokens[1], "".join(second_tokens[2:])])
        tags[-1].extend([second_tokens[0], "O", line_tokens[2]])
    elif (not line_tokens[1].isupper()) and line_tokens[2].isupper():
        tokens[-1].extend([line_tokens[0], line_tokens[1]])
        tags[-1].extend(["O", line_tokens[2]])
    elif line_tokens[1].isupper() and (not line_tokens[2].isupper()):
        tokens[-1].extend([line_tokens[0], line_tokens[2]])
        tags[-1].extend([line_tokens[1], "O"])
    elif line_tokens[1].isupper() and line_tokens[2].isupper():
        last_tag = tags[-1][-1] if tags[-1] != [] else "O"
        selected_tag = select_aptner_tags(line_tokens[0], line_tokens[1:], last_tag)
        tokens[-1].append(line_tokens[0])
        tags[-1].append(selected_tag)

    return tokens, tags


def parse_aptner_line(line, tokens, tags, split=None):
    """
    Parse one APTNER line.
    """
    line_tokens = line.split()

    if (len(line_tokens) == 2) or (
        len(line_tokens) == 3 and line_tokens[1] == line_tokens[2]
    ):
        tokens[-1].append(line_tokens[0])
        tags[-1].append(line_tokens[1])

    elif len(line_tokens) == 1:
        if line_tokens[0] != "O":
            tokens[-1].append(line_tokens[0])
            tags[-1].append("O")

    elif len(line_tokens) == 3:
        tokens, tags = fix_aptner_tokens(line_tokens, tokens, tags, split=split)

    else:
        last_tag = tags[-1][-1] if tags[-1] != [] else "O"
        selected_tag = select_aptner_tags(line_tokens[0], line_tokens[1:], last_tag)
        tokens[-1].append(line_tokens[0])
        tags[-1].append(selected_tag)

    return tokens, tags


def process_aptner_tokens(lines, split=None):
    """
    Process APTNER tokens and tags for common issues.
    """
    tokens = [[]]
    tags = [[]]

    for line in lines:
        if line.strip() == "" and tokens[-1] != []:
            tokens.append([])
            tags.append([])
        elif line.strip() != "":
            tokens, tags = parse_aptner_line(line, tokens, tags, split)

    return tokens, tags


def get_formatted_tag(tag, formatted_sentence_tags):
    """
    Convert APTNER BIESO tags to BIO tags.
    """
    tag = tag.replace("-S-SECTEAM", "-SECTEAM")

    if tag == "O":
        formatted_tag = tag
    elif tag == "PROT":
        formatted_tag = "B-PROT"
    elif tag.startswith("B-") or tag.startswith("S-"):
        formatted_tag = "B-" + tag[2:]
    elif tag.startswith("I-") or tag.startswith("E-"):
        if len(formatted_sentence_tags) == 0 or formatted_sentence_tags[-1][2:] != tag[2:]:
            formatted_tag = "B-" + tag[2:]
        else:
            formatted_tag = "I-" + tag[2:]
    else:
        formatted_tag = None

    return formatted_tag


def format_aptner_tags(tags, split=None):
    """
    Change the APTNER tag format from BIESO to BIO.
    """
    formatted_tags = []

    for sentence_tags in tags:
        formatted_sentence_tags = []

        for tag in sentence_tags:
            formatted_tag = get_formatted_tag(tag, formatted_sentence_tags)
            if formatted_tag is not None:
                formatted_sentence_tags.append(formatted_tag)

        formatted_tags.append(formatted_sentence_tags)

    return formatted_tags


def load_aptner_data(aptner_folder, split):
    """
    Load the APTNER data.
    """
    split_path = os.path.join(aptner_folder, f"APTNER{split}.txt")
    file_lines = read_text_file_with_fallbacks(split_path)

    lines = process_aptner_lines(file_lines)
    tokens, tags = process_aptner_tokens(lines, split)
    tokens, tags = split_ner_sentences(tokens, tags)
    tags = format_aptner_tags(tags, split)

    df_split = pd.DataFrame(
        {
            "tokens": tokens,
            "tags": tags,
            "split": split if split != "dev" else "val",
        }
    )

    return df_split


aptner_entity_definitions = {
    "APT": "threat participant",
    "SECTEAM": "security team",
    "IDTY": "authentication identity",
    "OS": "operating system",
    "EMAIL": "malicious mailbox",
    "LOC": "location",
    "TIME": "time",
    "IP": "IP address",
    "DOM": "domain",
    "URL": "URL",
    "PROT": "protocol",
    "FILE": "file sample",
    "TOOL": "tool",
    "MD5": "MD5 hash value",
    "SHA1": "SHA1 hash value",
    "SHA2": "SHA2 hash value",
    "MAL": "malware",
    "ENCR": "encryption algorithm",
    "VULNAME": "vulnerability name",
    "VULID": "vulnerability number",
    "ACT": "attack action",
}


def get_df_aptner():
    """
    Get the APTNER data.
    """
    df_aptner = pd.concat(
        [load_aptner_data(aptner_folder, split) for split in ["train", "dev", "test"]],
        ignore_index=True,
    )

    df_aptner["text"] = df_aptner.tokens.apply(lambda tokens: " ".join(tokens))
    df_aptner = df_aptner[df_aptner.text.str.len() > 1].drop_duplicates(
        subset=["text"], keep="first", ignore_index=True
    )[df_aptner.columns.drop("text")]

    df_aptner = filter_matching_token_tag_lengths(df_aptner)

    aptner_tags = ["O"] + [
        f"{prefix}-{entity_name}"
        for entity_name in aptner_entity_definitions.keys()
        for prefix in ["B", "I"]
    ]
    aptner_label2id = {label: idx for idx, label in enumerate(aptner_tags)}

    df_aptner["labels"] = df_aptner.tags.apply(
        lambda tags: [aptner_label2id[tag] for tag in tags]
    )
    df_aptner["input"] = df_aptner.tokens.apply(detokenizer.detokenize)
    df_aptner["output"] = [
        format_ner_output_from_lists(tokens, tags)
        for tokens, tags in zip(df_aptner["tokens"], df_aptner["tags"])
    ]

    df_aptner = assign_instructions(
        df_aptner,
        outputs=False,
        instructions=ner_instructions,
    )

    df_aptner["task"] = "ner"
    df_aptner["dataset"] = "aptner"

    df_aptner["instruction"] = df_aptner["instruction"].apply(
        lambda instruction: instruction.format(
            entity_types=", ".join(aptner_entity_definitions.keys()),
            entity_definitions="; ".join(
                [f"{key}: {value}" for key, value in aptner_entity_definitions.items()]
            ),
            output_format=ner_output_format,
        )
    )

    return df_aptner