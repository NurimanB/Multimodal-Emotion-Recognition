"""Download the MELD text CSVs and print split sizes and class balance.

    python scripts/download_meld.py --data_dir data/meld
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pandas as pd  # noqa: E402

from mer.data.meld import class_counts, download_meld, load_split  # noqa: E402

parser = argparse.ArgumentParser()
parser.add_argument("--data_dir", default="data/meld")
args = parser.parse_args()

download_meld(args.data_dir)
table = {}
for split in ["train", "dev", "test"]:
    df = load_split(args.data_dir, split)
    print(f"{split}: {len(df)} utterances, {df['dialogue_id'].nunique()} dialogues")
    table[split] = class_counts(df)
table = pd.DataFrame(table)
table["train_%"] = (100 * table["train"] / table["train"].sum()).round(1)
print("\nClass balance:")
print(table.to_string())
