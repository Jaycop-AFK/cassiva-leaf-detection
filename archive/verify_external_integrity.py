from pathlib import Path
import hashlib

import pandas as pd
from PIL import Image
import imagehash


# ================================================================
# CONFIG
# ================================================================

ROOT = Path(__file__).resolve().parent.parent

CLEAN_MANIFEST = (
    ROOT
    / "data"
    / "processed"
    / "cleaned_manifest.csv"
)

EXTERNAL_ROOT = (
    ROOT
    / "external_test_leaf_only"
    / "labeled"
)

OUTPUT_DIR = (
    ROOT
    / "outputs"
    / "external_integrity"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

EXACT_OUTPUT = (
    OUTPUT_DIR
    / "external_exact_duplicates.csv"
)

PHASH_OUTPUT = (
    OUTPUT_DIR
    / "external_near_duplicates.csv"
)

SUMMARY_OUTPUT = (
    OUTPUT_DIR
    / "external_integrity_summary.csv"
)

IMAGE_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".webp",
    ".bmp",
}

PHASH_THRESHOLD = 4


# ================================================================
# SHA256
# ================================================================

def sha256_file(path):

    sha = hashlib.sha256()

    with open(
        path,
        "rb",
    ) as file:

        while True:

            chunk = file.read(
                1024 * 1024
            )

            if not chunk:
                break

            sha.update(
                chunk
            )

    return sha.hexdigest()


# ================================================================
# PHASH
# ================================================================

def get_phash(path):

    with Image.open(
        path
    ) as image:

        image = image.convert(
            "RGB"
        )

        return imagehash.phash(
            image,
            hash_size=8,
        )


# ================================================================
# RESOLVE TRAIN IMAGE
# ================================================================

def resolve_train_path(
    filepath,
):

    path = Path(
        str(filepath)
    )

    if path.is_absolute():

        return path

    return (
        ROOT
        / path
    )


# ================================================================
# GET EXTERNAL IMAGES
# ================================================================

def collect_external_images():

    rows = []

    for class_name in [
        "cbb",
        "cbs",
        "cmd",
        "healthy",
    ]:

        class_dir = (
            EXTERNAL_ROOT
            / class_name
        )

        if not class_dir.exists():
            continue

        for path in sorted(
            class_dir.rglob("*")
        ):

            if not path.is_file():
                continue

            if (
                path.suffix.lower()
                not in IMAGE_EXTENSIONS
            ):
                continue

            rows.append(
                {
                    "external_path":
                        path,

                    "external_label":
                        class_name,
                }
            )

    return rows


# ================================================================
# LOAD TRAIN MANIFEST
# ================================================================

def load_train_manifest():

    if not CLEAN_MANIFEST.exists():

        raise FileNotFoundError(
            f"Manifest not found:\n"
            f"{CLEAN_MANIFEST}"
        )

    df = pd.read_csv(
        CLEAN_MANIFEST
    )

    if "filepath" not in df.columns:

        raise ValueError(
            "cleaned_manifest.csv "
            "does not contain filepath column."
        )

    return df


# ================================================================
# BUILD TRAIN HASH INDEX
# ================================================================

def build_train_hash_index(
    train_df,
):

    print()
    print("=" * 72)
    print("BUILD TRAIN HASH INDEX")
    print("=" * 72)

    sha_index = {}

    phash_rows = []

    total = len(
        train_df
    )

    valid = 0
    missing = 0


    for idx, row in train_df.iterrows():

        path = resolve_train_path(
            row[
                "filepath"
            ]
        )

        if not path.exists():

            missing += 1
            continue


        try:

            sha = sha256_file(
                path
            )

            phash_value = get_phash(
                path
            )

        except Exception as exc:

            print(
                "ERROR:",
                path,
                exc,
            )

            continue


        sha_index.setdefault(
            sha,
            []
        ).append(
            {
                "filepath":
                    row[
                        "filepath"
                    ],

                "label":
                    row.get(
                        "label",
                        ""
                    ),

                "source":
                    row.get(
                        "source",
                        ""
                    ),
            }
        )


        phash_rows.append(
            {
                "filepath":
                    row[
                        "filepath"
                    ],

                "label":
                    row.get(
                        "label",
                        ""
                    ),

                "source":
                    row.get(
                        "source",
                        ""
                    ),

                "phash":
                    phash_value,
            }
        )


        valid += 1


        if (
            valid % 1000
            == 0
        ):

            print(
                f"Processed "
                f"{valid}/{total}"
            )


    print()
    print(
        "Valid training images:",
        valid
    )

    print(
        "Missing training images:",
        missing
    )


    return (
        sha_index,
        phash_rows,
    )


# ================================================================
# CHECK EXTERNAL
# ================================================================

def check_external(
    external_rows,
    sha_index,
    train_phash_rows,
):

    exact_matches = []

    near_matches = []

    summary_rows = []


    print()
    print("=" * 72)
    print("CHECK EXTERNAL IMAGES")
    print("=" * 72)


    for index, item in enumerate(
        external_rows,
        start=1,
    ):

        external_path = (
            item[
                "external_path"
            ]
        )

        external_label = (
            item[
                "external_label"
            ]
        )


        print(
            f"[{index}/{len(external_rows)}] "
            f"{external_path.name}"
        )


        try:

            external_sha = sha256_file(
                external_path
            )

            external_phash = get_phash(
                external_path
            )

        except Exception as exc:

            print(
                "ERROR:",
                exc,
            )

            continue


        # ========================================================
        # EXACT DUPLICATE
        # ========================================================

        exact = (
            external_sha
            in sha_index
        )


        if exact:

            for train_match in (
                sha_index[
                    external_sha
                ]
            ):

                exact_matches.append(
                    {
                        "external_file":
                            external_path.relative_to(
                                ROOT
                            ).as_posix(),

                        "external_label":
                            external_label,

                        "train_file":
                            train_match[
                                "filepath"
                            ],

                        "train_label":
                            train_match[
                                "label"
                            ],

                        "train_source":
                            train_match[
                                "source"
                            ],

                        "sha256":
                            external_sha,
                    }
                )


        # ========================================================
        # NEAR DUPLICATES
        # ========================================================

        best_distance = None

        best_matches = []


        for train_row in (
            train_phash_rows
        ):

            distance = (
                external_phash
                - train_row[
                    "phash"
                ]
            )


            if (
                best_distance is None
                or distance
                < best_distance
            ):

                best_distance = (
                    distance
                )

                best_matches = [
                    train_row
                ]


            elif (
                distance
                == best_distance
            ):

                best_matches.append(
                    train_row
                )


        near_duplicate = (
            best_distance is not None
            and best_distance
            <= PHASH_THRESHOLD
        )


        if near_duplicate:

            for train_match in (
                best_matches
            ):

                near_matches.append(
                    {
                        "external_file":
                            external_path.relative_to(
                                ROOT
                            ).as_posix(),

                        "external_label":
                            external_label,

                        "train_file":
                            train_match[
                                "filepath"
                            ],

                        "train_label":
                            train_match[
                                "label"
                            ],

                        "train_source":
                            train_match[
                                "source"
                            ],

                        "phash_distance":
                            best_distance,
                    }
                )


        # ========================================================
        # SUMMARY
        # ========================================================

        summary_rows.append(
            {
                "external_file":
                    external_path.relative_to(
                        ROOT
                    ).as_posix(),

                "external_label":
                    external_label,

                "exact_duplicate":
                    exact,

                "nearest_phash_distance":
                    best_distance,

                "near_duplicate":
                    near_duplicate,

                "leakage_status":
                    (
                        "EXACT_DUPLICATE"
                        if exact
                        else (
                            "REVIEW_NEAR_DUPLICATE"
                            if near_duplicate
                            else "CLEAR"
                        )
                    ),
            }
        )


    return (
        pd.DataFrame(
            exact_matches
        ),
        pd.DataFrame(
            near_matches
        ),
        pd.DataFrame(
            summary_rows
        ),
    )


# ================================================================
# MAIN
# ================================================================

def main():

    print("=" * 72)
    print("STEP 11A - EXTERNAL DATA INTEGRITY CHECK")
    print("=" * 72)

    print(
        "Training manifest:",
        CLEAN_MANIFEST
    )

    print(
        "External dataset:",
        EXTERNAL_ROOT
    )

    print(
        "pHash threshold:",
        PHASH_THRESHOLD
    )


    train_df = (
        load_train_manifest()
    )


    external_rows = (
        collect_external_images()
    )


    print()
    print(
        "Training rows:",
        len(train_df)
    )

    print(
        "External images:",
        len(external_rows)
    )


    if len(
        external_rows
    ) == 0:

        raise RuntimeError(
            "No external images found."
        )


    (
        sha_index,
        train_phash_rows,
    ) = build_train_hash_index(
        train_df
    )


    (
        exact_df,
        near_df,
        summary_df,
    ) = check_external(
        external_rows,
        sha_index,
        train_phash_rows,
    )


    # ============================================================
    # SAVE
    # ============================================================

    exact_df.to_csv(
        EXACT_OUTPUT,
        index=False,
    )

    near_df.to_csv(
        PHASH_OUTPUT,
        index=False,
    )

    summary_df.to_csv(
        SUMMARY_OUTPUT,
        index=False,
    )


    # ============================================================
    # RESULTS
    # ============================================================

    exact_count = int(
        summary_df[
            "exact_duplicate"
        ].sum()
    )

    near_count = int(
        summary_df[
            "near_duplicate"
        ].sum()
    )

    clear_count = int(
        (
            summary_df[
                "leakage_status"
            ]
            == "CLEAR"
        ).sum()
    )


    print()
    print("=" * 72)
    print("INTEGRITY RESULTS")
    print("=" * 72)

    print(
        "External images:",
        len(summary_df)
    )

    print(
        "Exact duplicates:",
        exact_count
    )

    print(
        "Near-duplicate candidates:",
        near_count
    )

    print(
        "Clear:",
        clear_count
    )


    print()
    print(
        "Status counts:"
    )

    print(
        summary_df[
            "leakage_status"
        ]
        .value_counts()
        .to_string()
    )


    if exact_count > 0:

        print()
        print("=" * 72)
        print("WARNING: EXACT DATA LEAKAGE FOUND")
        print("=" * 72)

        print(
            exact_df[
                [
                    "external_file",
                    "external_label",
                    "train_file",
                    "train_label",
                    "train_source",
                ]
            ].to_string(
                index=False
            )
        )


    if len(
        near_df
    ) > 0:

        print()
        print("=" * 72)
        print("NEAR-DUPLICATE CANDIDATES")
        print("=" * 72)

        print(
            near_df[
                [
                    "external_file",
                    "external_label",
                    "train_file",
                    "train_label",
                    "train_source",
                    "phash_distance",
                ]
            ].to_string(
                index=False
            )
        )


    print()
    print("=" * 72)
    print("STEP 11A COMPLETE")
    print("=" * 72)

    print(
        "Summary:",
        SUMMARY_OUTPUT
    )

    print(
        "Exact matches:",
        EXACT_OUTPUT
    )

    print(
        "Near matches:",
        PHASH_OUTPUT
    )


if __name__ == "__main__":
    main()
