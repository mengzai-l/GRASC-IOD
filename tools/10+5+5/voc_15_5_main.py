import os
import shutil
import random
import torch
import xml.etree.ElementTree as ET
from collections import Counter
from pathlib import Path
from tqdm import tqdm
from PIL import Image

from voc_15_5_LFF import LFFGenerator
from voc_15_5_DSA import DSAChecker
from voc_15_5_ASC import ASCCalibrator

# ==============================================================================
# 1. Configuration
# ==============================================================================
VOC_ROOT = Path("data/VOC2007")
AUG_ROOT = Path("data/VOC2007_15+5")  # Target directory for 15+5[cite: 10]

SD_PATH = "runwayml/stable-diffusion-inpainting"
CLIP_PATH = "openai/clip-vit-base-patch32"

# 15+5 specific model configuration[cite: 12]
CONFIG_FILE = 'configs/faster-rcnn-increase/voc/10-5-5/grasc-iod/10+5_GR.py'
CHECKPOINT_FILE = 'work_dir/voc/10-5-5/grasc-iod/10+5_GR/epoch_6.pth'
DEVICE = "cuda:0"

PREV5_CLASSES = ["diningtable", "dog", "horse", "motorbike", "person"]  # Classes to generate[cite: 10]
NEW5_CLASSES = ["pottedplant", "sheep", "sofa", "train", "tvmonitor"]  # Future classes[cite: 10]
ALL_CLASSES = PREV5_CLASSES + NEW5_CLASSES

RIGID_CLASSES = ["diningtable", "motorbike"]  # [cite: 12]
NON_RIGID_CLASSES = ["dog", "horse", "person"]  # [cite: 12]
BASE_CLASSES = RIGID_CLASSES + NON_RIGID_CLASSES

TARGET_PER_CLASS = 200  # [cite: 10]
PARAM_CONFIG = {"default": {"str": 0.80, "step": 30, "pad": 40}}

# ==============================================================================
# 2. Directory Initialization
# ==============================================================================
V4_IMG = AUG_ROOT / "JPEGImages"
V4_ANN = AUG_ROOT / "Annotations"
if V4_IMG.exists(): shutil.rmtree(V4_IMG)
if V4_ANN.exists(): shutil.rmtree(V4_ANN)
V4_IMG.mkdir(parents=True, exist_ok=True)
V4_ANN.mkdir(parents=True, exist_ok=True)

# 10+5+5 architecture logic[cite: 10]
SRC_LIST_FOR_GEN = VOC_ROOT / "10+5+5/task1_trainval.txt"
SRC_LIST_FOR_NEW = VOC_ROOT / "10+5+5/task2_trainval.txt"
LIST_FINAL_TASK3 = VOC_ROOT / "ImageSets/Main/final_merged_15_5.txt"

if __name__ == "__main__":
    print("=" * 60)
    print("🚀 GRASC-IOD: VOC 15+5 Main Pipeline Started")
    print("=" * 60)

    with open(SRC_LIST_FOR_GEN, 'r') as f:
        ALLOWED_IDS = set([line.strip() for line in f if line.strip()])

    # --- Phase 1: M1 & M2 ---
    lff = LFFGenerator(SD_PATH, DEVICE)
    dsa = DSAChecker(CLIP_PATH, ALL_CLASSES, DEVICE)

    tasks = []
    generated = Counter()

    for pid in list(ALLOWED_IDS):
        xml_file = VOC_ROOT / f"Annotations/{pid}.xml"
        if not xml_file.exists(): continue
        try:
            root = ET.parse(xml_file).getroot()
            for obj in root.findall("object"):
                name = obj.findtext("name")
                if name in PREV5_CLASSES:
                    bnd = obj.find("bndbox")
                    box = [int(float(bnd.findtext(k))) for k in ["xmin", "ymin", "xmax", "ymax"]]
                    tasks.append({"id": pid, "cls": name, "box": box})
        except:
            continue

    random.shuffle(tasks)
    pbar = tqdm(total=len(PREV5_CLASSES) * TARGET_PER_CLASS, desc="Generation & Filtering")

    for t in tasks:
        cls = t["cls"]
        if generated[cls] >= TARGET_PER_CLASS: continue
        try:
            pid = t["id"]
            img_p = VOC_ROOT / f"JPEGImages/{pid}.jpg"
            img = Image.open(img_p).convert("RGB")
            cfg = PARAM_CONFIG.get(cls, PARAM_CONFIG["default"])

            obj_crop, full_img = lff.generate_image(img, cls, t["box"], cfg)

            if dsa.check(obj_crop, cls):
                aug_id = f"{pid}_aug_{generated[cls]}"
                full_img.save(V4_IMG / f"{aug_id}.jpg")
                xml_tree = ET.parse(VOC_ROOT / f"Annotations/{pid}.xml")
                xml_tree.find("filename").text = f"{aug_id}.jpg"
                xml_tree.write(V4_ANN / f"{aug_id}.xml")
                generated[cls] += 1
                pbar.update(1)
        except Exception:
            continue

    pbar.close()
    del lff, dsa
    torch.cuda.empty_cache()

    # --- Phase 2: Copy Real Task 2 Whitelist Data ---
    print(f"\n🚚 [Step 2] Copying real data to v4...")
    src_img_dir = VOC_ROOT / "JPEGImages"
    src_ann_dir = VOC_ROOT / "Annotations"
    for pid in tqdm(sorted(list(ALLOWED_IDS)), desc="Copying Real Data"):
        src_xml = src_ann_dir / f"{pid}.xml"
        if src_xml.exists(): shutil.copy2(str(src_xml), str(V4_ANN / f"{pid}.xml"))
        src_jpg = src_img_dir / f"{pid}.jpg"
        if src_jpg.exists(): shutil.copy2(str(src_jpg), str(V4_IMG / f"{pid}.jpg"))

    # --- Phase 3: M3 (ASC Calibration) ---
    asc = ASCCalibrator(CONFIG_FILE, CHECKPOINT_FILE, RIGID_CLASSES, BASE_CLASSES, DEVICE)
    asc.process_folder(V4_ANN, V4_IMG)
    del asc
    torch.cuda.empty_cache()

    # --- Phase 4: Sync & Merge Lists ---
    print(f"\n🏗️  Synchronizing data to main directory...")
    main_img = VOC_ROOT / "JPEGImages"
    main_ann = VOC_ROOT / "Annotations"

    for xml in tqdm(list(V4_ANN.glob("*_aug_*.xml")), desc="Sync XML"):
        shutil.copy2(str(xml), str(main_ann / xml.name))
        shutil.copy2(str(xml), str(main_img / xml.name))
    for img in tqdm(list(V4_IMG.glob("*_aug_*.jpg")), desc="Sync JPG"):
        shutil.copy2(str(img), str(main_img / img.name))

    with open(SRC_LIST_FOR_NEW, 'r') as f:
        final_ids = set([line.strip() for line in f if line.strip()])
    gen_ids = [f[:-4] for f in os.listdir(VOC_ROOT / "JPEGImages") if '_aug_' in f and f.endswith('.jpg')]
    final_ids.update(gen_ids)

    FINAL_LIST_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(FINAL_LIST_PATH, 'w', encoding='utf-8') as f:
        f.write('\n'.join(sorted(list(final_ids))))

    print(f"\n🎉 GRASC-IOD 15+5 data preparation finished: {FINAL_LIST_PATH}")