# Cassava Leaf Disease Classification

วิชา Applied Machine Learning — จำแนกโรคจากภาพใบมันสำปะหลัง

## ขอบเขต
Image classification 4 คลาส (ไม่ใช่ object detection — ไม่มี bounding box)
เลือกเฉพาะโรคที่เกิดจริงในประเทศไทย

| คลาส | ชื่อไทย | แหล่งข้อมูล |
|---|---|---|
| healthy | ต้นปกติ | Kaggle + CCMT |
| cmd | ใบด่าง (Cassava Mosaic Disease) | Kaggle + CCMT |
| cbb | ใบไหม้ (Cassava Bacterial Blight) | Kaggle + CCMT |
| bls | ใบจุดสีน้ำตาล (Brown Leaf Spot) | CCMT เท่านั้น |

**ตัดออก: CBSD และ CGM** — ไม่มีรายงานการระบาดในไทย
(CBSD พบเฉพาะแอฟริกา, Green Mottle พบเฉพาะหมู่เกาะโซโลมอน)

## Environment
- Windows, venv อยู่ที่ `venv/`
- PyTorch 2.11.0+cu128 (RTX 5060 Laptop, sm_120, VRAM 8GB)
- **สำคัญ**: ห้ามลง torch จาก PyPI ปกติ ต้องใช้ index `https://download.pytorch.org/whl/cu128`
- timm, scikit-learn, pandas, imagehash, seaborn, matplotlib, tqdm

## ข้อมูล
data/raw/kaggle/ — 21,397 ภาพ 800x600 RGB, train.csv (image_id,label)
  label map: 0=CBB 1=CBSD 2=CGM 3=CMD 4=Healthy
  counts: 0:1087  1:2189  2:2386  3:13158  4:2577
  → หลังตัด CBSD/CGM เหลือ 16,822 ภาพ

data/raw/ccmt/Dataset for Crop Pest and Disease Detection/ — CCMT (Mendeley bwh3zbpkpv)
  ใช้เฉพาะ `Raw Data/CCMT Dataset/Cassava/<class>/*.jpg` — 7,508 ภาพ 400x400 RGB, ไม่มีไฟล์เสีย
  counts: bacterial blight:2614  brown spot:1481  green mite:1015  healthy:1193  mosaic:1205
  map: bacterial blight→cbb, brown spot→bls, healthy→healthy, mosaic→cmd, green mite→ตัดออก
  → หลังตัด green mite เหลือ 6,493 ภาพ
  **ห้ามใช้ `CCMT Dataset-Augmented/`** — สร้างจาก raw (test_set มีไฟล์ byte-identical กับ raw) → data leakage
  พบไฟล์ซ้ำ byte-identical ใน raw เอง (brown spot: 1481 ไฟล์ แต่ md5 ไม่ซ้ำ 1467) → จัดการใน Step 3
  **ขนาดภาพต่างแหล่ง** (Kaggle 800x600 vs CCMT 400x400) = shortcut ของ source bias อีกทาง

## ปัญหาที่รู้แล้ว ต้องแก้
1. **Class imbalance หนัก** — CMD:CBB = 12.1:1
   trivial baseline (ตอบ CMD ตลอด) = accuracy 78.2% แต่ macro-F1 แค่ 0.293
   → ต้องใช้ class weighting หรือ focal loss
2. **Source bias** — bls มาจาก CCMT แหล่งเดียว
   โมเดลอาจเรียนรู้ "สไตล์ภาพ CCMT" แทนอาการโรค
   → ต้องดึง healthy/cmd/cbb จากทั้งสองแหล่งเพื่อไม่ให้ source กับ class สัมพันธ์กันสมบูรณ์
   → ต้องทำ Grad-CAM ยืนยันว่าโมเดลมองรอยโรคจริง

## เมตริก
**macro-F1 คือตัวหลัก** ไม่ใช่ accuracy
ต้องรายงาน per-class precision/recall/F1 + confusion matrix
คู่ที่คาดว่าจะสับสน: cbb กับ bls (จุดบนใบเหมือนกัน)

## แผน
- [x] Step 0 — ตั้ง environment, verify GPU
- [x] Step 1 — สำรวจข้อมูลดิบ (Kaggle + CCMT)
- [ ] Step 2 — สร้าง manifest รวมสองแหล่ง + ตัดคลาสที่ไม่ใช้
- [ ] Step 3 — clean: หาไฟล์เสีย, ภาพซ้ำด้วย perceptual hash
- [ ] Step 4 — stratified split (stratify ทั้ง class และ source)
- [ ] Step 5 — Dataset/DataLoader + augmentation
- [ ] Step 6 — train EfficientNet-B0 (timm) + AMP + class weights
- [ ] Step 7 — evaluate + confusion matrix + Grad-CAM
- [ ] Step 8 — ablation: มี/ไม่มี augmentation, มี/ไม่มี class weighting

## สไตล์การทำงานที่ต้องการ
สอนแบบ step by step ทีละสเต็ป อธิบายว่าทำไมถึงทำแบบนั้น
ไม่ต้องเขียนโค้ดรวดเดียวจบ ให้ทำทีละขั้นแล้วรอ verify ก่อนไปต่อ