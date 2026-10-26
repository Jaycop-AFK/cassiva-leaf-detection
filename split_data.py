from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split


ROOT = Path(__file__).resolve().parent

CLEANED_MANIFEST = (
    ROOT
    / "data"
    / "processed"
    / "cleaned_manifest.csv"
)


print("=" * 64)
print("STEP 4 - TRAIN / VAL / TEST SPLIT")
print("=" * 64)


df = pd.read_csv(CLEANED_MANIFEST)

print("Total rows:", len(df))
print()
print("Labels:")
print(df["label"].value_counts())

print()
print("Sources:")
print(df["source"].value_counts())

df["stratify_key"] = (
    df["label"]
    + "_"
    + df["source"]
)

print()
print("Stratification groups:")
print(df["stratify_key"].value_counts())


train_df, temp_df = train_test_split(
    df,
    test_size=0.30,
    random_state=42,
    stratify=df["stratify_key"],
)


val_df, test_df = train_test_split(
    temp_df,
    test_size=0.50,
    random_state=42,
    stratify=temp_df["stratify_key"],
)


print()
print("=" * 64)
print("SPLIT SUMMARY")
print("=" * 64)

print("Train:", len(train_df))
print("Val  :", len(val_df))
print("Test :", len(test_df))
print("Total:", len(train_df) + len(val_df) + len(test_df))


for name, split_df in [
    ("TRAIN", train_df),
    ("VAL", val_df),
    ("TEST", test_df),
]:
    print()
    print(f"--- {name} ---")

    print("Labels:")
    print(split_df["label"].value_counts())

    print()
    print("Sources:")
    print(split_df["source"].value_counts())
    
    
train_paths = set(train_df["filepath"])
val_paths = set(val_df["filepath"])
test_paths = set(test_df["filepath"])

print()
print("=" * 64)
print("OVERLAP CHECK")
print("=" * 64)

print("Train vs Val :", len(train_paths & val_paths))
print("Train vs Test:", len(train_paths & test_paths))
print("Val vs Test  :", len(val_paths & test_paths))



print()
print("=" * 64)
print("SAVE SPLIT MANIFEST")
print("=" * 64)

train_df = train_df.copy()
val_df = val_df.copy()
test_df = test_df.copy()

train_df["split"] = "train"
val_df["split"] = "val"
test_df["split"] = "test"

split_manifest = pd.concat(
    [train_df, val_df, test_df],
    ignore_index=True
)

split_manifest = split_manifest[
    ["filepath", "label", "source", "split"]
].copy()

SPLIT_PATH = (
    ROOT
    / "data"
    / "processed"
    / "split_manifest.csv"
)

split_manifest.to_csv(
    SPLIT_PATH,
    index=False,
    encoding="utf-8"
)

print("Saved:", SPLIT_PATH)
print("Rows :", len(split_manifest))


print()
print("VERIFY SPLIT MANIFEST")

saved = pd.read_csv(SPLIT_PATH)

print("Rows:", len(saved))

print()
print("Split counts:")
print(saved["split"].value_counts())

print()
print("Class x Split:")
print(
    pd.crosstab(
        saved["label"],
        saved["split"]
    )
)

print()
print("Source x Split:")
print(
    pd.crosstab(
        saved["source"],
        saved["split"]
    )
)

print()
print("Duplicate filepaths:",
      saved["filepath"].duplicated().sum())

print("Missing values:")
print(saved.isna().sum())


