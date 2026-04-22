"""
SPDX-License-Identifier: Apache-2.0
Copyright : JP Morgan Chase & Co

Utilities
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

train_size = 0.8
val_size = 0.1
test_size = 0.1


def split_dataset(
    df,
    train_size=train_size,
    val_size=val_size,
    test_size=test_size,
    stratify=False,
    verbose=False,
):
    """
    Split the dataset and add a "split" column.
    """
    if stratify:
        if "label" not in df.columns:
            raise ValueError("stratify=True requires a 'label' column.")
        if df["label"].isna().any():
            raise ValueError(
                f"stratify=True but 'label' contains {df['label'].isna().sum()} NaN values."
            )

        df_train, df_val_test = train_test_split(
            df,
            train_size=train_size,
            shuffle=True,
            stratify=df["label"],
            random_state=0,
        )
        df_val, df_test = train_test_split(
            df_val_test,
            train_size=val_size / (test_size + val_size),
            shuffle=True,
            stratify=df_val_test["label"],
            random_state=0,
        )
    else:
        df_train, df_val_test = train_test_split(
            df,
            train_size=train_size,
            shuffle=True,
            random_state=0,
        )
        df_val, df_test = train_test_split(
            df_val_test,
            train_size=val_size / (test_size + val_size),
            shuffle=True,
            random_state=0,
        )

    df_train = df_train.copy()
    df_val = df_val.copy()
    df_test = df_test.copy()

    df_train["split"] = "train"
    df_val["split"] = "val"
    df_test["split"] = "test"

    df = pd.concat([df_train, df_val, df_test]).sort_index()

    if verbose:
        print("Split:")
        print(df["split"].value_counts())

    return df


def assign_instructions(df, outputs=None, instructions=None, verbose=False):
    """
    Map labels and assign instructions.
    """
    if instructions is None:
        instructions = [None]

    if outputs is None:
        outputs = sorted(df["output"].unique().tolist())

    if outputs is not False:
        output2label = {output: label for label, output in enumerate(outputs)}
        df = df.copy()
        df["label"] = df["output"].apply(output2label.get)

    np.random.seed(0)
    df = df.copy()
    df["instruction"] = np.random.choice(instructions, size=len(df), replace=True)

    if verbose and "label" in df.columns:
        print("Label:")
        print(df["label"].value_counts())

    return df


def drop_long_sequences(df, max_chars=2000):
    """
    Drop long sequences by the maximum number of characters.
    """
    df = df.copy()
    if len(df) == 0:
        print("WARNING: empty dataset received in drop_long_sequences")
        return df
    df["num_chars"] = df["input"].str.len()
    df_new = df[df["num_chars"] <= max_chars].copy()

    keep_ratio = len(df_new) / len(df)
    dataset_name = str(df.iloc[0].dataset).upper() if "dataset" in df.columns else "DATASET"
    print(
        f"{dataset_name}: keep {len(df_new)} of {len(df)} ({keep_ratio:.2%}), "
        f"drop {len(df) - len(df_new)} ({1 - keep_ratio:.2%})"
    )

    return df_new.drop(columns=["num_chars"])