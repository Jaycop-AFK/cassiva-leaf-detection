import pandas as pd 
from pathlib import Path

ROOT = Path(__file__).resolve().parent

DATA = ROOT / "data"
RAW = DATA / "raw"
PROCESSED = DATA / "processed"

KAGGLE = RAW / "kaggle"
KAGGLE_CSV = KAGGLE / "train.csv"
KAGGLE_IMAGES = KAGGLE / "train_images"

CCMT = RAW / "ccmt" / "Cassava"


print("Project root :", ROOT)
print("Kaggle CSV  :", KAGGLE_CSV)
print("Kaggle imgs :", KAGGLE_IMAGES)
print("CCMT        :", CCMT)

print()
print("Path check")

print("Kaggle CSV exists :", KAGGLE_CSV.exists())
print("Kaggle imgs exists:", KAGGLE_IMAGES.exists())
print("CCMT exists       :", CCMT.exists())

print()
print("=" * 64)
print("KAGGLE MANIFEST")
print("=" * 64)


kaggle_df = pd.read_csv(KAGGLE_CSV)

print("Original rows:", len(kaggle_df))
print("Original label counts:")
print(kaggle_df["label"].value_counts().sort_index())

KAGGLE_LABEL_MAP = {
    0: "cbb",
    3: "cmd",
    4: "healthy",
}

kaggle_df = kaggle_df[
    kaggle_df["label"].isin(KAGGLE_LABEL_MAP)
].copy()

kaggle_df["label"] = kaggle_df["label"].map(KAGGLE_LABEL_MAP)

kaggle_df["filepath"] = kaggle_df["image_id"].apply(
    lambda name: (KAGGLE_IMAGES / name).relative_to(ROOT).as_posix()
)

kaggle_df["source"] = "kaggle"

kaggle_manifest = kaggle_df[
    ["filepath", "label", "source"]
].copy()

print()
print("Filtered Kaggle rows:", len(kaggle_manifest))

print("Mapped label counts:")
print(kaggle_manifest["label"].value_counts())

print()
print("Sample:")
print(kaggle_manifest.head())


print()
print("=" * 64)
print("CCMT MANIFEST")
print("=" * 64)

CCMT_LABEL_MAP = {
    "bacterial blight": "cbb",
    "brown spot": "cbs",
    "healthy": "healthy",
    "mosaic": "cmd",
}

IMAGE_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
}

ccmt_rows = []

for folder_name, label in CCMT_LABEL_MAP.items():

    class_dir = CCMT / folder_name

    print()
    print(f"Reading: {folder_name} -> {label}")

    if not class_dir.exists():
        raise FileNotFoundError(
            f"ไม่เจอโฟลเดอร์: {class_dir}"
        )

    image_files = sorted(
        p
        for p in class_dir.rglob("*")
        if p.is_file()
        and p.suffix.lower() in IMAGE_EXTENSIONS
    )

    print(f"  images: {len(image_files)}")
    
    for img_path in image_files:
        rel_path = img_path.relative_to(ROOT).as_posix()
        ccmt_rows.append((rel_path, label, "ccmt"))

ccmt_manifest = pd.DataFrame(ccmt_rows, columns=["filepath", "label", "source"])




print()
print("CCMT manifest rows:", len(ccmt_manifest))

print("Mapped label counts:")
print(ccmt_manifest["label"].value_counts())

print()
print("Sample:")
print(ccmt_manifest.head())

print()
print("=" * 64)
print("COMBINED MANIFEST")
print("=" * 64)

manifest = pd.concat(
    [kaggle_manifest, ccmt_manifest],
    ignore_index=True
)

print("Total rows:", len(manifest))

print()
print("Combined label counts:")
print(manifest["label"].value_counts())

print()
print("Counts by source and label:")
print(
    pd.crosstab(
        manifest["source"],
        manifest["label"]
    )
)

print()
print("Duplicate filepaths:", manifest["filepath"].duplicated().sum())


print()
print("=" * 64)
print("SAVE MANIFEST")
print("=" * 64)

PROCESSED.mkdir(parents=True, exist_ok=True)

MANIFEST_PATH = PROCESSED / "manifest.csv"

manifest.to_csv(
    MANIFEST_PATH,
    index=False,
    encoding="utf-8"
)

print("Saved to :", MANIFEST_PATH)
print("File exists:", MANIFEST_PATH.exists())

print()
print("VERIFY SAVED MANIFEST")

saved_manifest = pd.read_csv(MANIFEST_PATH)

print("Rows    :", len(saved_manifest))
print("Columns :", saved_manifest.columns.tolist())

print()
print("Labels:")
print(saved_manifest["label"].value_counts())

print()
print("Sources:")
print(saved_manifest["source"].value_counts())

print()
print("Duplicate filepaths:",
      saved_manifest["filepath"].duplicated().sum())

print("Missing values:")
print(saved_manifest.isna().sum())

missing_files = []

for filepath in saved_manifest["filepath"]:
    full_path = ROOT / filepath

    if not full_path.is_file():
        missing_files.append(filepath)

print()
print("Missing image files:", len(missing_files))

if missing_files:
    print("Example missing files:")
    for path in missing_files[:10]:
        print(" ", path)
        

print()
print("=" * 64)
print("SAVE MANIFEST")
print("=" * 64)

PROCESSED.mkdir(parents=True, exist_ok=True)

MANIFEST_PATH = PROCESSED / "manifest.csv"

manifest.to_csv(
    MANIFEST_PATH,
    index=False,
    encoding="utf-8"
)

print("Saved to :", MANIFEST_PATH)
print("File exists:", MANIFEST_PATH.exists())

print()
print("VERIFY SAVED MANIFEST")

saved_manifest = pd.read_csv(MANIFEST_PATH)

print("Rows    :", len(saved_manifest))
print("Columns :", saved_manifest.columns.tolist())

print()
print("Labels:")
print(saved_manifest["label"].value_counts())

print()
print("Sources:")
print(saved_manifest["source"].value_counts())

print()
print(
    "Duplicate filepaths:",
    saved_manifest["filepath"].duplicated().sum()
)

print()
print("Missing values:")
print(saved_manifest.isna().sum())

missing_files = []

for filepath in saved_manifest["filepath"]:
    full_path = ROOT / filepath

    if not full_path.is_file():
        missing_files.append(filepath)

print()
print("Missing image files:", len(missing_files))

if missing_files:
    print("Example missing files:")
    for path in missing_files[:10]:
        print(" ", path)
        

print()
print("=" * 64)
print("SAVE MANIFEST")
print("=" * 64)

PROCESSED.mkdir(parents=True, exist_ok=True)

MANIFEST_PATH = PROCESSED / "manifest.csv"

manifest.to_csv(
    MANIFEST_PATH,
    index=False,
    encoding="utf-8"
)

print("Saved to :", MANIFEST_PATH)
print("File exists:", MANIFEST_PATH.exists())

print()
print("VERIFY SAVED MANIFEST")

saved_manifest = pd.read_csv(MANIFEST_PATH)

print("Rows    :", len(saved_manifest))
print("Columns :", saved_manifest.columns.tolist())

print()
print("Labels:")
print(saved_manifest["label"].value_counts())

print()
print("Sources:")
print(saved_manifest["source"].value_counts())

print()
print(
    "Duplicate filepaths:",
    saved_manifest["filepath"].duplicated().sum()
)

print()
print("Missing values:")
print(saved_manifest.isna().sum())

missing_files = []

for filepath in saved_manifest["filepath"]:
    full_path = ROOT / filepath

    if not full_path.is_file():
        missing_files.append(filepath)

print()
print("Missing image files:", len(missing_files))

if missing_files:
    print("Example missing files:")
    for path in missing_files[:10]:
        print(" ", path)