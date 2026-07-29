import os
import torch
import numpy as np
import xml.etree.ElementTree as ET
from tqdm import tqdm
from mmdet.apis import init_detector, inference_detector


class ASCCalibrator:
    def __init__(self, config_file, checkpoint_file, rigid_classes, base_classes, device="cuda:0"):
        print(f">> [ASC] Initializing spatial calibration module...")
        if not os.path.exists(checkpoint_file):
            raise FileNotFoundError(f"Weights not found: {checkpoint_file}")

        self.model = init_detector(config_file, checkpoint_file, device=device)
        self.rigid_classes = rigid_classes
        self.base_classes = base_classes

    def compute_iou(self, box1, box2):
        xx1, yy1 = np.maximum(box1[:2], box2[:2])
        xx2, yy2 = np.minimum(box1[2:], box2[2:])
        w, h = np.maximum(0, xx2 - xx1 + 1), np.maximum(0, yy2 - yy1 + 1)
        inter = w * h
        area1 = (box1[2] - box1[0] + 1) * (box1[3] - box1[1] + 1)
        area2 = (box2[2] - box2[0] + 1) * (box2[3] - box2[1] + 1)
        return inter / (area1 + area2 - inter + 1e-6)

    def process_folder(self, v4_ann_dir, v4_img_dir):
        xml_files = list(v4_ann_dir.glob("*_aug_*.xml"))
        stats = {"rigid_rectified": 0, "rigid_kept": 0, "soft_kept": 0, "deleted": 0}

        for xml_p in tqdm(xml_files, desc="ASC Calibration Progress"):
            img_p = v4_img_dir / xml_p.with_suffix(".jpg").name
            if not img_p.exists():
                os.remove(xml_p)
                continue

            try:
                result = inference_detector(self.model, str(img_p))
                tree = ET.parse(xml_p)
                root = tree.getroot()
                valid_objs = []
                has_mod = False

                for obj in root.findall("object"):
                    name = obj.findtext("name")
                    if name not in self.base_classes:
                        valid_objs.append(obj)
                        continue

                    cls_idx = self.model.dataset_meta['classes'].index(name) if hasattr(self.model,
                                                                                        'dataset_meta') else self.model.CLASSES.index(
                        name)
                    preds = result[cls_idx]
                    preds = preds[preds[:, 4] > 0.3]

                    if name in self.rigid_classes:
                        if len(preds) == 0:
                            has_mod = True
                            continue
                        bnd = obj.find("bndbox")
                        gt = np.array([float(bnd.findtext(k)) for k in ["xmin", "ymin", "xmax", "ymax"]])
                        ious = np.array([self.compute_iou(gt, p[:4]) for p in preds])
                        best_idx = np.argmax(ious)
                        max_iou = ious[best_idx]

                        if max_iou < 0.3:
                            has_mod = True
                            continue
                        elif max_iou < 0.85:
                            best_box = preds[best_idx][:4]
                            bnd.find("xmin").text = str(int(best_box[0]))
                            bnd.find("ymin").text = str(int(best_box[1]))
                            bnd.find("xmax").text = str(int(best_box[2]))
                            bnd.find("ymax").text = str(int(best_box[3]))
                            stats["rigid_rectified"] += 1
                            has_mod = True
                            valid_objs.append(obj)
                        else:
                            stats["rigid_kept"] += 1
                            valid_objs.append(obj)
                    else:
                        if len(preds) == 0:
                            has_mod = True
                            continue
                        stats["soft_kept"] += 1
                        valid_objs.append(obj)

                if not valid_objs:
                    os.remove(xml_p)
                    os.remove(img_p)
                    stats["deleted"] += 1
                elif has_mod:
                    for o in root.findall("object"): root.remove(o)
                    for o in valid_objs: root.append(o)
                    tree.write(xml_p)
            except Exception:
                continue

        print(f">> [ASC Report] Calibration completed: {stats}")