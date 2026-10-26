from pathlib import Path
from collections import Counter, defaultdict
import hashlib
import matplotlib.pyplot as plt
import imagehash

import pandas as pd
from PIL import Image
from tqdm import tqdm


ROOT = Path(__file__).resolve().parent

MANIFEST_PATH = ROOT / "data" / "processed" / "manifest.csv"


print("=" * 64)
print("STEP 3.1 - IMAGE INTEGRITY CHECK")
print("=" * 64)


manifest = pd.read_csv(MANIFEST_PATH)

print("Manifest rows:", len(manifest))


broken = []
modes = Counter()
sizes = Counter()

for row in tqdm(
    manifest.itertuples(index=False),
    total=len(manifest),
    desc="Checking images"
):
    image_path = ROOT / row.filepath

    try:
        with Image.open(image_path) as img:
            img.load()

            modes[img.mode] += 1
            sizes[img.size] += 1

    except Exception as e:
        broken.append({
            "filepath": row.filepath,
            "label": row.label,
            "source": row.source,
            "error": str(e),
        })
        
print()
print("=" * 64)
print("IMAGE AUDIT SUMMARY")
print("=" * 64)

print("Total checked :", len(manifest))
print("Broken images :", len(broken))

print()
print("Image modes:")
for mode, count in modes.most_common():
    print(f"  {mode:<10} {count}")

print()
print("Image sizes:")
for size, count in sizes.most_common():
    print(f"  {str(size):<15} {count}")
    
if broken:
    print()
    print("Broken image examples:")

    for item in broken[:10]:
        print(
            item["filepath"],
            "|",
            item["error"]
        )
        
print()
print("=" * 64)
print("STEP 3.2 - EXACT DUPLICATE CHECK")
print("=" * 64)


def sha256_file(path, chunk_size=1024 * 1024):
    hasher = hashlib.sha256()

    with open(path, "rb") as f:
        while True:
            chunk = f.read(chunk_size)

            if not chunk:
                break

            hasher.update(chunk)

    return hasher.hexdigest()


hash_groups = defaultdict(list)


for row in tqdm(
    manifest.itertuples(index=False),
    total=len(manifest),
    desc="Calculating SHA256"
):
    image_path = ROOT / row.filepath

    file_hash = sha256_file(image_path)

    hash_groups[file_hash].append({
        "filepath": row.filepath,
        "label": row.label,
        "source": row.source,
    })
    
duplicate_groups = {
    file_hash: items
    for file_hash, items in hash_groups.items()
    if len(items) > 1
}


duplicate_extra_files = sum(
    len(items) - 1
    for items in duplicate_groups.values()
)

print()
print("Exact duplicate groups :", len(duplicate_groups))
print("Extra duplicate files :", duplicate_extra_files)


conflict_groups = []

for file_hash, items in duplicate_groups.items():

    labels = {item["label"] for item in items}

    if len(labels) > 1:
        conflict_groups.append({
            "hash": file_hash,
            "items": items,
        })


print("Label conflict groups  :", len(conflict_groups))

if duplicate_groups:
    print()
    print("Duplicate examples:")

    for i, (file_hash, items) in enumerate(
        duplicate_groups.items(),
        start=1
    ):
        print()
        print(f"Group {i}")
        print(f"Hash: {file_hash}")

        for item in items:
            print(
                " ",
                item["label"],
                "|",
                item["source"],
                "|",
                item["filepath"]
            )

        if i >= 10:
            break
        

if conflict_groups:
    print()
    print("=" * 64)
    print("WARNING: LABEL CONFLICTS")
    print("=" * 64)

    for i, group in enumerate(conflict_groups[:10], start=1):

        print()
        print(f"Conflict {i}")

        for item in group["items"]:
            print(
                " ",
                item["label"],
                "|",
                item["source"],
                "|",
                item["filepath"]
            )
            
print()
print("=" * 64)
print("BUILD EXACT-DUPLICATE EXCLUSION LIST")
print("=" * 64)

exact_exclude = set()


for file_hash, items in duplicate_groups.items():

    labels = {item["label"] for item in items}

    # Case 1:
    # รูปเดียวกัน แต่คนละ label
    # ไม่รู้ว่า annotation ไหนถูก → ตัดทั้งหมด
    if len(labels) > 1:

        for item in items:
            exact_exclude.add(item["filepath"])

    # Case 2:
    # รูปเดียวกัน และ label เหมือนกัน
    # เก็บตัวแรกไว้หนึ่งรูป
    else:

        # เรียง path เพื่อให้ผล reproducible
        items_sorted = sorted(
            items,
            key=lambda x: x["filepath"]
        )

        # [0] เก็บไว้
        # [1:] ตัดออก
        for item in items_sorted[1:]:
            exact_exclude.add(item["filepath"])


print("Files excluded:", len(exact_exclude))


working_manifest = manifest[
    ~manifest["filepath"].isin(exact_exclude)
].copy()

working_manifest.reset_index(
    drop=True,
    inplace=True
)

print("Before exact dedupe :", len(manifest))
print("Excluded            :", len(exact_exclude))
print("After exact dedupe  :", len(working_manifest))


print()
print("Class counts after exact dedupe:")
print(
    working_manifest["label"]
    .value_counts()
)

print()
print("Source counts after exact dedupe:")
print(
    working_manifest["source"]
    .value_counts()
)


print()
print("=" * 64)
print("STEP 3.3A - PERCEPTUAL HASH CHECK")
print("=" * 64)

phash_groups = defaultdict(list)


for row in tqdm(
    working_manifest.itertuples(index=False),
    total=len(working_manifest),
    desc="Calculating pHash"
):
    image_path = ROOT / row.filepath

    with Image.open(image_path) as img:
        img_hash = imagehash.phash(
            img.convert("RGB"),
            hash_size=8
        )

    phash_groups[str(img_hash)].append({
        "filepath": row.filepath,
        "label": row.label,
        "source": row.source,
    })
    
    
phash_duplicate_groups = {
    phash: items
    for phash, items in phash_groups.items()
    if len(items) > 1
}


phash_extra_files = sum(
    len(items) - 1
    for items in phash_duplicate_groups.values()
)


print()
print("pHash duplicate groups :", len(phash_duplicate_groups))
print("Extra candidate files  :", phash_extra_files)


phash_conflict_groups = []

for phash, items in phash_duplicate_groups.items():

    labels = {
        item["label"]
        for item in items
    }

    if len(labels) > 1:
        phash_conflict_groups.append({
            "phash": phash,
            "items": items,
        })


print(
    "pHash label conflicts :",
    len(phash_conflict_groups)
)


if phash_duplicate_groups:

    print()
    print("pHash duplicate examples:")

    for i, (phash, items) in enumerate(
        phash_duplicate_groups.items(),
        start=1
    ):
        print()
        print(f"Group {i}")
        print(f"pHash: {phash}")

        for item in items:
            print(
                " ",
                item["label"],
                "|",
                item["source"],
                "|",
                item["filepath"]
            )

        if i >= 10:
            break
        
        
if phash_conflict_groups:

    print()
    print("=" * 64)
    print("WARNING: pHash LABEL CONFLICTS")
    print("=" * 64)

    for i, group in enumerate(
        phash_conflict_groups[:10],
        start=1
    ):
        print()
        print(f"Conflict {i}")

        for item in group["items"]:
            print(
                " ",
                item["label"],
                "|",
                item["source"],
                "|",
                item["filepath"]
            )
            
            
print()
print("=" * 64)
print("EXPORT pHash CANDIDATES")
print("=" * 64)

phash_candidate_rows = []

for group_id, (phash, items) in enumerate(
    phash_duplicate_groups.items(),
    start=1
):
    for item in items:
        phash_candidate_rows.append({
            "group": group_id,
            "phash": phash,
            "filepath": item["filepath"],
            "label": item["label"],
            "source": item["source"],
        })


phash_candidates_df = pd.DataFrame(phash_candidate_rows)

PHASH_CSV = (
    ROOT
    / "data"
    / "processed"
    / "phash_candidates.csv"
)

phash_candidates_df.to_csv(
    PHASH_CSV,
    index=False,
    encoding="utf-8"
)

print("Saved:", PHASH_CSV)
print("Rows :", len(phash_candidates_df))
print("Groups:", phash_candidates_df["group"].nunique())


COMPARE_DIR = (
    ROOT
    / "data"
    / "processed"
    / "phash_compare"
)

COMPARE_DIR.mkdir(
    parents=True,
    exist_ok=True
)


for group_id, (phash, items) in enumerate(
    phash_duplicate_groups.items(),
    start=1
):
    if len(items) != 2:
        continue

    left = items[0]
    right = items[1]

    left_path = ROOT / left["filepath"]
    right_path = ROOT / right["filepath"]

    with Image.open(left_path) as img:
        left_img = img.convert("RGB")

    with Image.open(right_path) as img:
        right_img = img.convert("RGB")

    fig, axes = plt.subplots(
        1,
        2,
        figsize=(10, 5)
    )

    axes[0].imshow(left_img)
    axes[0].set_title(
        f'{left["label"]}\n{Path(left["filepath"]).name}'
    )
    axes[0].axis("off")

    axes[1].imshow(right_img)
    axes[1].set_title(
        f'{right["label"]}\n{Path(right["filepath"]).name}'
    )
    axes[1].axis("off")

    fig.suptitle(
        f"pHash Group {group_id} | {phash}"
    )

    plt.tight_layout()

    output_path = (
        COMPARE_DIR
        / f"group_{group_id:03d}.jpg"
    )

    plt.savefig(
        output_path,
        dpi=150,
        bbox_inches="tight"
    )

    plt.close(fig)


print(
    "Comparison images saved to:",
    COMPARE_DIR
)


print()
print("=" * 64)
print("STEP 3.3B - APPLY VERIFIED pHASH DUPLICATE POLICY")
print("=" * 64)

# The 25 pHash groups were visually verified as exact duplicate images.
# Keep one deterministic representative per group; preserve raw files.
phash_exclude = set()
phash_audit_rows = []

for group_id, (phash, items) in enumerate(
    phash_duplicate_groups.items(),
    start=1
):
    items_sorted = sorted(items, key=lambda x: x["filepath"])
    kept = items_sorted[0]

    for item in items_sorted[1:]:
        phash_exclude.add(item["filepath"])
        phash_audit_rows.append({
            "group": group_id,
            "phash": phash,
            "kept_filepath": kept["filepath"],
            "excluded_filepath": item["filepath"],
            "label": item["label"],
            "source": item["source"],
            "reason": "perceptual_duplicate_phash_exact",
        })

cleaned_manifest = working_manifest[
    ~working_manifest["filepath"].isin(phash_exclude)
].copy()
cleaned_manifest.reset_index(drop=True, inplace=True)

CLEANED_MANIFEST_PATH = ROOT / "data" / "processed" / "cleaned_manifest.csv"
PHASH_AUDIT_PATH = ROOT / "data" / "processed" / "phash_exclusion_audit.csv"

cleaned_manifest.to_csv(CLEANED_MANIFEST_PATH, index=False, encoding="utf-8")
pd.DataFrame(phash_audit_rows).to_csv(
    PHASH_AUDIT_PATH,
    index=False,
    encoding="utf-8",
)

print("pHash files excluded:", len(phash_exclude))
print("Cleaned manifest rows:", len(cleaned_manifest))
print("Saved:", CLEANED_MANIFEST_PATH)
print("Saved:", PHASH_AUDIT_PATH)

print()
print("Class counts after pHash cleaning:")
print(cleaned_manifest["label"].value_counts())
print()
print("Source counts after pHash cleaning:")
print(cleaned_manifest["source"].value_counts())
