import os
import shutil
import random
import torch
import xml.etree.ElementTree as ET
from collections import Counter
from pathlib import Path
from tqdm import tqdm
from PIL import Image

# Import the 3 core modules
from voc_10_10_LFF import LFFGenerator
from voc_10_10_DSA import DSAChecker
from voc_10_10_ASC import ASCCalibrator

# ==============================================================================
# 1. Paths and Basic Configuration
# ==============================================================================
VOC_ROOT = Path("/xxx/yyy/zzz/VOC2007")
AUG_ROOT = Path("data/VOC2007_base10")

SD_PATH = "runwayml/stable-diffusion-inpainting"
CLIP_PATH = "openai/clip-vit-base-patch32"
CONFIG_FILE = 'configs/faster-rcnn/voc/10/10.py'
CHECKPOINT_FILE = 'base/base10.pth'
DEVICE = "cuda:0"

RIGID_CLASSES = ["bicycle", "bottle", "bus", "car", "chair"]
NON_RIGID_CLASSES = ["aeroplane", "bird", "boat", "cat", "cow"]
BASE_CLASSES = RIGID_CLASSES + NON_RIGID_CLASSES
NEW_CLASSES = ["diningtable", "dog", "horse", "motorbike", "person", "pottedplant", "sheep", "sofa", "train",
               "tvmonitor"]
ALL_CLASSES = BASE_CLASSES + NEW_CLASSES

TARGET_PER_CLASS = 600
PARAM_CONFIG = {"default": {"str": 0.80, "step": 30, "pad": 40}}

# ==============================================================================
# 2. Directory Cleaning and Initialization
# ==============================================================================
V4_IMG = AUG_ROOT / "JPEGImages"
V4_ANN = AUG_ROOT / "Annotations"
if V4_IMG.exists(): shutil.rmtree(V4_IMG)
if V4_ANN.exists(): shutil.rmtree(V4_ANN)
V4_IMG.mkdir(parents=True, exist_ok=True)
V4_ANN.mkdir(parents=True, exist_ok=True)

TASK0_SRC_LIST = VOC_ROOT / "10+10/task0_trainval.txt"
TASK1_SRC_LIST = VOC_ROOT / "10+10/task1_trainval.txt"
LIST_TASK0_STRICT = AUG_ROOT / "train_task0_strict.txt"
FINAL_LIST_PATH = VOC_ROOT / "ImageSets/Main/final_merged_10_10.txt"

if __name__ == "__main__":
    print("=" * 60)
    print("GRASC-IOD: VOC 10+10 Main Pipeline Started")
    print("=" * 60)

    with open(TASK0_SRC_LIST, 'r') as f:
        ALLOWED_IDS_T0 = set([line.strip() for line in f if line.strip()])

    # ---------------------------------------------------------
    # Phase 1: Initialize M1 (Generation) & M2 (Semantic Alignment)
    # ---------------------------------------------------------
    lff = LFFGenerator(SD_PATH, DEVICE)
    dsa = DSAChecker(CLIP_PATH, ALL_CLASSES, DEVICE)

    tasks = []
    generated = Counter()

    for pid in list(ALLOWED_IDS_T0):
        xml_file = VOC_ROOT / f"Annotations/{pid}.xml"
        if not xml_file.exists(): continue
        try:
            root = ET.parse(xml_file).getroot()
            for obj in root.findall("object"):
                name = obj.findtext("name")
                if name in BASE_CLASSES:
                    bnd = obj.find("bndbox")
                    box = [int(float(bnd.findtext(k))) for k in ["xmin", "ymin", "xmax", "ymax"]]
                    tasks.append({"id": pid, "cls": name, "box": box})
        except:
            continue

    random.shuffle(tasks)
    pbar = tqdm(total=len(BASE_CLASSES) * TARGET_PER_CLASS, desc="Generation & Filtering Progress")

    for t in tasks:
        cls = t["cls"]
        if generated[cls] >= TARGET_PER_CLASS: continue
        try:
            pid = t["id"]
            img_p = VOC_ROOT / f"JPEGImages/{pid}.jpg"
            img = Image.open(img_p).convert("RGB")
            cfg = PARAM_CONFIG.get(cls, PARAM_CONFIG["default"])

            # 1. Call LFF for generation
            obj_crop, full_img = lff.generate_image(img, cls, t["box"], cfg)

            # 2. Call DSA for validation
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

    # Free VRAM, prepare for ASC calibration
    del lff
    del dsa
    torch.cuda.empty_cache()

    # ---------------------------------------------------------
    # Phase 2: Generate Task 0 clean list
    # ---------------------------------------------------------
    with open(LIST_TASK0_STRICT, 'w') as f:
        f.write('\n'.join(sorted(list(ALLOWED_IDS_T0))))

    # ---------------------------------------------------------
    # Phase 3: Initialize M3 (ASC Calibration) & Final Synchronization
    # ---------------------------------------------------------
    asc = ASCCalibrator(CONFIG_FILE, CHECKPOINT_FILE, RIGID_CLASSES, BASE_CLASSES, DEVICE)
    asc.process_folder(V4_ANN, V4_IMG)
    del asc
    torch.cuda.empty_cache()

    print(f"\nStart synchronizing clean data to the main directory...")
    main_img = VOC_ROOT / "JPEGImages"
    main_ann = VOC_ROOT / "Annotations"

    clean_xmls = list(V4_ANN.glob("*_aug_*.xml"))
    clean_imgs = list(V4_IMG.glob("*_aug_*.jpg"))

    for xml in tqdm(clean_xmls, desc="Sync XML"):
        shutil.copy2(str(xml), str(main_ann / xml.name))
        shutil.copy2(str(xml), str(main_img / xml.name))
    for img in tqdm(clean_imgs, desc="Sync JPG"):
        shutil.copy2(str(img), str(main_img / img.name))

    real_ids = [line.strip().replace('\n', '').replace('\\n', '') for line in open(TASK1_SRC_LIST) if line.strip()]
    aug_ids = [f.stem.strip() for f in clean_imgs if '_aug_' in f.stem]

    final_list = sorted(list(set(real_ids) | set(aug_ids)))
    valid_final = [pid for pid in final_list if
                   (main_img / f"{pid}.jpg").exists() and (main_ann / f"{pid}.xml").exists()]

    FINAL_LIST_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(FINAL_LIST_PATH, 'w', encoding='utf-8') as f:
        for pid in valid_final: f.write(f"{pid}\n")

    print(f"\nExecution completed! GRASC-IOD 10+10 data preparation finished: {FINAL_LIST_PATH}")