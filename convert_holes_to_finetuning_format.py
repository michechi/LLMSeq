#!/usr/bin/env python3
"""
Convert sequences_with_holes.csv to the format expected by Finetuning_FOX_time_simulation_tiny.py

This script:
1. Reads sequences_with_holes.csv
2. Converts sequence format from "A -> B -> C" to "A\x1fB\x1fC"
3. Uses is_ordered as the outcome label
4. Splits into train/val/test sets
5. Saves as separate X and y CSV files
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s: %(message)s")
logger = logging.getLogger(__name__)

# Configuration
INPUT_FILE = "/root/MIMICIV/data/simulation/holes/sequences_with_holes.csv"
OUTPUT_DIR = "/root/MIMICIV/data/simulation/"
FILE_NUMBER = "big"
SEQUENCE_COLUMN = "original_seq"  # Use sequences with holes (or use 'original_seq') seq_with_holes
OUTCOME_COLUMN = "is_ordered"

# Train/val/test split ratios (80/10/10)
TRAIN_SIZE = 0.80
VAL_SIZE = 0.10
TEST_SIZE = 0.10
RANDOM_STATE = 42

def convert_sequence_format(sequence_str):
    """
    Convert from "A -> B -> C" format to "A\x1fB\x1fC" format
    """
    if pd.isna(sequence_str):
        return ""

    # Split by arrow and strip whitespace
    parts = [part.strip() for part in sequence_str.split("->")]
    # Join with \x1f delimiter
    return "\x1f".join(parts)

def main():
    logger.info(f"Reading input file: {INPUT_FILE}")
    df = pd.read_csv(INPUT_FILE)

    logger.info(f"Input file shape: {df.shape}")
    logger.info(f"Columns: {df.columns.tolist()}")

    # Check if required columns exist
    if SEQUENCE_COLUMN not in df.columns:
        raise ValueError(f"Column '{SEQUENCE_COLUMN}' not found in input file")
    if OUTCOME_COLUMN not in df.columns:
        raise ValueError(f"Column '{OUTCOME_COLUMN}' not found in input file")

    # Convert sequence format
    logger.info(f"Converting sequences from '-> ' format to '\\x1f' delimiter format...")
    df["Sequences"] = df[SEQUENCE_COLUMN].apply(convert_sequence_format)

    # Create outcome column
    logger.info(f"Using '{OUTCOME_COLUMN}' as outcome label...")
    df["Outcome"] = df[OUTCOME_COLUMN].astype(int)

    # Select relevant columns
    X = df[["Sequences"]].copy()
    y = df[["Outcome"]].copy()

    # Check for any invalid sequences
    invalid_count = (X["Sequences"] == "").sum()
    if invalid_count > 0:
        logger.warning(f"Found {invalid_count} invalid/empty sequences, removing them...")
        mask = X["Sequences"] != ""
        X = X[mask].reset_index(drop=True)
        y = y[mask].reset_index(drop=True)

    logger.info(f"Total sequences after cleaning: {len(X)}")
    logger.info(f"Outcome distribution:\n{y['Outcome'].value_counts()}")

    # First split: train + temp (which will be val + test)
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y,
        train_size=TRAIN_SIZE,
        random_state=RANDOM_STATE,
        stratify=y  # Maintain class distribution
    )

    # Second split: val and test from temp
    val_test_ratio = VAL_SIZE / (VAL_SIZE + TEST_SIZE)  # Adjust ratio for the remaining data
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp,
        train_size=val_test_ratio,
        random_state=RANDOM_STATE,
        stratify=y_temp
    )

    logger.info(f"\nTrain set size: {len(X_train)}")
    logger.info(f"Val set size: {len(X_val)}")
    logger.info(f"Test set size: {len(X_test)}")

    # Save files
    logger.info(f"\nSaving files to {OUTPUT_DIR}...")

    X_train.to_csv(f"{OUTPUT_DIR}X_train_{FILE_NUMBER}.csv", index=False)
    y_train.to_csv(f"{OUTPUT_DIR}y_train_{FILE_NUMBER}.csv", index=False)

    X_val.to_csv(f"{OUTPUT_DIR}X_val_{FILE_NUMBER}.csv", index=False)
    y_val.to_csv(f"{OUTPUT_DIR}y_val_{FILE_NUMBER}.csv", index=False)

    X_test.to_csv(f"{OUTPUT_DIR}X_test_{FILE_NUMBER}.csv", index=False)
    y_test.to_csv(f"{OUTPUT_DIR}y_test_{FILE_NUMBER}.csv", index=False)

    logger.info(f"✓ X_train_{FILE_NUMBER}.csv")
    logger.info(f"✓ y_train_{FILE_NUMBER}.csv")
    logger.info(f"✓ X_val_{FILE_NUMBER}.csv")
    logger.info(f"✓ y_val_{FILE_NUMBER}.csv")
    logger.info(f"✓ X_test_{FILE_NUMBER}.csv")
    logger.info(f"✓ y_test_{FILE_NUMBER}.csv")

    # Print sample to verify
    logger.info(f"\nSample train sequence:")
    logger.info(f"  Sequences: {X_train['Sequences'].iloc[0][:100]}...")
    logger.info(f"  Outcome: {y_train['Outcome'].iloc[0]}")

    logger.info("\nConversion complete! You can now use these files with Finetuning_FOX_time_simulation_tiny.py")
    logger.info(f"Example command:")
    logger.info(f"  python src/Finetuning_FOX_time_simulation_tiny.py --model_name meta-llama/Llama-2-7b --number_to_use {FILE_NUMBER}")

if __name__ == "__main__":
    main()
