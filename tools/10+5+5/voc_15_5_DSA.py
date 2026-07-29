import torch
from transformers import CLIPProcessor, CLIPModel


class DSAChecker:
    def __init__(self, clip_path, all_classes, device="cuda:0"):
        print(f">> [DSA] Initializing semantic alignment module...")
        self.device = device
        self.clip_model = CLIPModel.from_pretrained(clip_path).to(device)
        self.clip_processor = CLIPProcessor.from_pretrained(clip_path)

        # 15+5 specific defense map[cite: 10]
        self.defense_map = {
            "motorbike": ["person"],
            "dog": ["sheep"],
            "horse": ["sheep"],
            "diningtable": ["sofa", "tvmonitor"]
        }
        self.defense_margin = 0.03
        self.clip_cfg = {"default": {"abs": 0.22, "mar": 0.05}}

        self.anchors = {}
        with torch.no_grad():
            inputs = self.clip_processor(text=[f"a photo of a {c}" for c in all_classes], return_tensors="pt",
                                         padding=True).to(device)
            emb = self.clip_model.get_text_features(**inputs)
            emb = emb / emb.norm(dim=-1, keepdim=True)
            for i, c in enumerate(all_classes):
                self.anchors[c] = emb[i:i + 1]

    def check(self, image, cls_name):
        with torch.no_grad():
            inputs = self.clip_processor(images=image, return_tensors="pt").to(self.device)
            img_emb = self.clip_model.get_image_features(**inputs)
            img_emb = img_emb / img_emb.norm(dim=-1, keepdim=True)

            sim_self = (img_emb @ self.anchors[cls_name].T).item()
            cfg = self.clip_cfg.get(cls_name, self.clip_cfg["default"])

            if sim_self < cfg["abs"]: return False

            if cls_name in self.defense_map:
                for enemy in self.defense_map[cls_name]:
                    sim_enemy = (img_emb @ self.anchors[enemy].T).item()
                    if sim_self < sim_enemy + self.defense_margin:
                        return False
        return True