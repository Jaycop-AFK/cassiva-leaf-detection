from pathlib import Path
from collections import Counter
import json


from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "data" / "raw"
IMG_EXT = {".jpg", ".jpeg", ".png"}


def is_img(p: Path) -> bool:
    return p.suffix.lower() in IMG_EXT


def peek(paths, n=3):
    """เปิดภาพตัวอย่างดูขนาดกับโหมดสี"""
    for p in paths[:n]:
        try:
            with Image.open(p) as im:
                print(f"       {p.name:<30s} {str(im.size):<14s} {im.mode}")
        except Exception as e:
            print(f"       {p.name:<30s} ERROR {e!r}")


print("=" * 64)
print("KAGGLE")
print("=" * 64)

kg = RAW / "kaggle"
if not kg.exists():
    print("!! ไม่เจอ data/raw/kaggle")
else:
    print("ไฟล์ชั้นบนสุด:", sorted(p.name for p in kg.iterdir()))

    lm = kg / "label_num_to_disease_map.json"
    if lm.exists():
        print("\nlabel map:")
        for k, v in json.loads(lm.read_text()).items():
            print(f"   {k} -> {v}")

    tc = kg / "train.csv"
    if tc.exists():
        rows = tc.read_text().strip().splitlines()
        print(f"\ntrain.csv: {len(rows) - 1} แถว")
        print("   header:", rows[0])
        print("   sample:", rows[1])
        cnt = Counter(r.split(",")[1] for r in rows[1:])
        print("   จำนวนต่อ label:", dict(sorted(cnt.items())))

    ti = kg / "train_images"
    if ti.exists():
        imgs = [p for p in ti.iterdir() if is_img(p)]
        print(f"\ntrain_images: {len(imgs)} ไฟล์")
        peek(imgs)


print()
print("=" * 64)
print("CCMT")
print("=" * 64)

cc = RAW / "ccmt" / "Cassava"

if not cc.exists():
    print("!! ไม่เจอ data/raw/ccmt/Cassava")
else:
    print(f"CCMT Cassava path: {cc}")

    for sub in sorted(d for d in cc.iterdir() if d.is_dir()):
        imgs = [p for p in sub.rglob("*") if is_img(p)]

        nested = sorted(
            d.name
            for d in sub.iterdir()
            if d.is_dir()
        )

        tag = f" [ซ้อน: {nested}]" if nested else ""

        print(f"  {sub.name:<28s} {len(imgs):>6d} ภาพ{tag}")

        peek(imgs, 2)
        
        

sizes = Counter()
modes = Counter()
broken = []

all_ccmt_imgs = [
    p
    for p in cc.rglob("*")
    if p.is_file() and is_img(p)
]

for img_path in all_ccmt_imgs:
    try:
        with Image.open(img_path) as img:
            sizes[img.size] += 1
            modes[img.mode] += 1
    except Exception as e:
        broken.append((img_path, str(e)))

print()
print("CCMT summary")
print(f"  total images : {len(all_ccmt_imgs)}")
print(f"  sizes        : {dict(sizes)}")
print(f"  modes        : {dict(modes)}")
print(f"  broken       : {len(broken)}")
