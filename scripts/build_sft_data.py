####---------------------- Libraries ----------------------####
import os
import json
import sys
import hashlib
import pandas as pd
from datasets import load_dataset
####------------------------------------------------------------------####


# This function creates a hash based on the instruction and input text, which helps identify duplicate or overlapping examples between splits.
def text_hash(instruction, input_text):
    raw = f"{str(instruction).strip()}\n<<<SEP>>>\n{str(input_text).strip()}"
    return hashlib.sha1(raw.encode("utf-8")).hexdigest()


# Utility function to save a list of records as JSONL.
def write_jsonl(path, records):
    with open(path, "w", encoding="utf-8") as f:
        for rec in records:
            obj = {
                "instruction": str(rec["instruction"]),
                "input": str(rec["input"]),
                "output": str(rec["output"]),
            }
            f.write(json.dumps(obj, ensure_ascii=False) + "\n")


print("=" * 80)
print("BLOCK 1: CyberBench train / val / test split")
print("=" * 80)

# Load the cyberbench.csv file and identify the column that indicates the data split.
df = pd.read_csv("data/cyberbench.csv")
print(f"Total rows in cyberbench.csv: {len(df)}")
print(f"Columns: {list(df.columns)}")

split_col = None
for cand in ["split", "set", "subset", "partition", "fold"]:
    if cand in df.columns:
        split_col = cand
        break

if split_col is None:
    print("ERROR: Split column not found.")
    sys.exit(1)

print(f"Split column detected: {split_col}")
print(df[split_col].value_counts(dropna=False).to_string())

train_values = {"train", "training", "dev_train"}
val_values = {"val", "validation", "valid", "dev"}
test_values = {"test", "testing", "eval", "evaluation"}

split_series = df[split_col].astype(str).str.strip().str.lower()
df_train = df[split_series.isin(train_values)].copy()
df_val = df[split_series.isin(val_values)].copy()
df_test = df[split_series.isin(test_values)].copy()

# Handle cases where the split column might have missing values or unexpected entries
if len(df_train) == 0:
    print("ERROR: The train filter returned 0 rows.")
    sys.exit(1)

if len(df_val) == 0:
    print("ERROR: The val filter returned 0 rows.")
    sys.exit(1)

if len(df_test) == 0:
    print("ERROR: The test filter returned 0 rows.")
    sys.exit(1)

for col in ["instruction", "input", "output"]:
    if col not in df_train.columns:
        print(f"ERROR: Missing column '{col}'")
        sys.exit(1)

# Delete any rows with missing instruction, input, or output in all splits.
df_train = df_train[["instruction", "input", "output"]].dropna().copy()
df_val = df_val[["instruction", "input", "output"]].dropna().copy()
df_test = df_test[["instruction", "input", "output"]].dropna().copy()

# Create a hash for each instruction-input pair in each split to identify overlaps.
df_train["pair_hash"] = df_train.apply(
    lambda r: text_hash(r["instruction"], r["input"]), axis=1
)
df_val["pair_hash"] = df_val.apply(
    lambda r: text_hash(r["instruction"], r["input"]), axis=1
)
df_test["pair_hash"] = df_test.apply(
    lambda r: text_hash(r["instruction"], r["input"]), axis=1
)

train_hashes = set(df_train["pair_hash"])
val_hashes = set(df_val["pair_hash"])
test_hashes = set(df_test["pair_hash"])

overlap_train_val = train_hashes & val_hashes
overlap_train_test = train_hashes & test_hashes
overlap_val_test = val_hashes & test_hashes

print(f"Train examples CyberBench (before): {len(df_train)}")
print(f"Val examples CyberBench           : {len(df_val)}")
print(f"Test examples CyberBench          : {len(df_test)}")
print(f"Overlap train/val by hash         : {len(overlap_train_val)}")
print(f"Overlap train/test by hash        : {len(overlap_train_test)}")
print(f"Overlap val/test by hash          : {len(overlap_val_test)}")

# Conservative policy:
# - Remove from train anything also present in val or test
# - Remove from val anything also present in test
bad_train_hashes = overlap_train_val | overlap_train_test
bad_val_hashes = overlap_val_test

if bad_train_hashes:
    print("WARNING: Overlapping examples detected in TRAIN against VAL/TEST.")
    print("These will be removed from the train set to avoid data leakage.")
    overlap_rows = df_train[df_train["pair_hash"].isin(bad_train_hashes)][["instruction", "input", "output"]].head(5)
    if len(overlap_rows) > 0:
        print("\nSample of overlapping train examples:")
        print(overlap_rows.to_string())
    df_train = df_train[~df_train["pair_hash"].isin(bad_train_hashes)].copy()

if bad_val_hashes:
    print("WARNING: Overlapping examples detected in VAL against TEST.")
    print("These will be removed from the validation set to avoid data leakage.")
    overlap_rows = df_val[df_val["pair_hash"].isin(bad_val_hashes)][["instruction", "input", "output"]].head(5)
    if len(overlap_rows) > 0:
        print("\nSample of overlapping val examples:")
        print(overlap_rows.to_string())
    df_val = df_val[~df_val["pair_hash"].isin(bad_val_hashes)].copy()

print(f"Train examples CyberBench (after): {len(df_train)}")
print(f"Val examples CyberBench   (after): {len(df_val)}")

# Remove any internal duplicates within train and val.
before_train_dedup = len(df_train)
before_val_dedup = len(df_val)

df_train = df_train.drop_duplicates(subset=["instruction", "input", "output"]).copy()
df_val = df_val.drop_duplicates(subset=["instruction", "input", "output"]).copy()

print(f"Internal duplicates removed in train: {before_train_dedup - len(df_train)}")
print(f"Internal duplicates removed in val  : {before_val_dedup - len(df_val)}")

# pair_hash is no longer needed.
df_train = df_train.drop(columns=["pair_hash"], errors="ignore")
df_val = df_val.drop(columns=["pair_hash"], errors="ignore")
df_test = df_test.drop(columns=["pair_hash"], errors="ignore")

# Convert cleaned DataFrames into lists of dictionaries.
cyberbench_train_records = df_train.to_dict(orient="records")
cyberbench_val_records = df_val.to_dict(orient="records")

print("\n" + "=" * 80)
print("BLOCK 2: MMLU auxiliary_train by subject")
print("=" * 80)

default_subjects = [
    "elementary_mathematics",
    "high_school_biology",
    "high_school_chemistry",
    "high_school_physics",
]

# The script allows users to specify which MMLU subjects to include in the training data via an environment variable (MMLU_SUBJECTS).
# If the variable is not set, it defaults to a predefined list of subjects.
subjects_env = os.environ.get("MMLU_SUBJECTS", "")
mmlu_subjects = [s.strip() for s in subjects_env.split(",") if s.strip()] or default_subjects

print("Selected Subjects MMLU:")
for s in mmlu_subjects:
    print(" -", s)

choice_labels = ["A", "B", "C", "D"]
mmlu_records = []
mmlu_stats = {}

# For each selected MMLU subject, load the "auxiliary_train" split.
for subject in mmlu_subjects:
    print(f"\nLoading MMLU subject={subject} split=auxiliary_train")
    try:
        ds = load_dataset("cais/mmlu", subject, split="auxiliary_train")
        print("Columns:", ds.column_names)

        count = 0
        subject_name = subject.replace("_", " ")
        for row in ds:
            if "question" not in row or "choices" not in row or "answer" not in row:
                continue

            choices = row["choices"]
            if not isinstance(choices, list) or len(choices) != 4:
                continue

            choices_str = "\n".join(
                f"{choice_labels[i]}. {choices[i]}" for i in range(4)
            )
            answer_letter = choice_labels[int(row["answer"])]

            mmlu_records.append({
                "instruction": (
                    f"Answer the following {subject_name} multiple-choice question. "
                    f"Choose the best answer from the options below and respond with only the letter "
                    f"(A, B, C, or D)."
                ),
                "input": f"{row['question']}\n{choices_str}",
                "output": answer_letter,
            })
            count += 1

        mmlu_stats[subject] = count
        print(f"  -> examples loaded: {count}")

    except Exception as e:
        print(f"  -> WARNING: could not load {subject}: {e}")
        mmlu_stats[subject] = 0

# Finally, combine all datasets.
# IMPORTANT: MMLU is added ONLY to TRAIN, not to VAL.
print("\n" + "=" * 80)
print("COMBINING DATASETS")
print("=" * 80)

all_train_records = cyberbench_train_records + mmlu_records
all_val_records = cyberbench_val_records

print(f"CyberBench train: {len(cyberbench_train_records)}")
print(f"CyberBench val  : {len(cyberbench_val_records)}")
print(f"MMLU train      : {len(mmlu_records)}")
print(f"TOTAL train     : {len(all_train_records)}")
print(f"TOTAL val       : {len(all_val_records)}")

write_jsonl("data/cyberinstruct_train.jsonl", all_train_records)
write_jsonl("data/cyberinstruct_val.jsonl", all_val_records)

with open("data/mmlu_subjects_used.json", "w", encoding="utf-8") as f:
    json.dump(
        {
            "subjects_requested": mmlu_subjects,
            "counts": mmlu_stats,
            "note": "Public approximation of the paper's MMLU science subset using per-subject auxiliary_train splits.",
        },
        f,
        indent=2,
        ensure_ascii=False,
    )

print("\nSaved: data/cyberinstruct_train.jsonl")
print("Saved: data/cyberinstruct_val.jsonl")
print("Saved: data/mmlu_subjects_used.json")
