import torch
import numpy as np
from PIL import Image, ImageDraw, ImageFilter
from diffusers import StableDiffusionInpaintPipeline


class LFFGenerator:
    def __init__(self, sd_path, device="cuda:0"):
        print(f">> [LFF] Initializing generation module...")
        self.pipe = StableDiffusionInpaintPipeline.from_pretrained(
            sd_path, torch_dtype=torch.float16, safety_checker=None
        ).to(device)
        self.pipe.set_progress_bar_config(disable=True)

    def color_match(self, orig, new):
        o, n = np.array(orig), np.array(new)
        res = np.zeros_like(n)
        for i in range(3):
            res[:, :, i] = (n[:, :, i] - n[:, :, i].mean()) * (o[:, :, i].std() / (n[:, :, i].std() + 1e-6)) + o[:, :,
                                                                                                               i].mean()
        return Image.fromarray(np.clip(res, 0, 255).astype(np.uint8))

    def expand_box(self, box, w, h, pad):
        return [max(0, box[0] - pad), max(0, box[1] - pad), min(w, box[2] + pad), min(h, box[3] + pad)]

    def make_mask(self, size, box_rel, blur):
        m = Image.new("L", size, 0)
        ImageDraw.Draw(m).rectangle(box_rel, fill=255)
        return m.filter(ImageFilter.GaussianBlur(blur))

    def generate_image(self, img, cls_name, box, cfg):
        W, H = img.size
        big_box = self.expand_box(box, W, H, cfg["pad"])
        patch_orig = img.crop(big_box)
        rel_box = [box[0] - big_box[0], box[1] - big_box[1], box[2] - big_box[0], box[3] - big_box[1]]
        mask = self.make_mask(patch_orig.size, rel_box, 9)

        prompt = f"a high resolution photo of a {cls_name}, realistic, 4k, detailed"
        neg = "cartoon, drawing, blurry, bad quality"

        # 15+5 specific negative prompts to protect NEW5_CLASSES
        if cls_name == "diningtable": neg += ", sofa, tvmonitor"
        if cls_name in ["dog", "horse"]: neg += ", sheep"

        res = self.pipe(
            prompt, negative_prompt=neg, image=patch_orig.resize((512, 512)),
            mask_image=mask.resize((512, 512)), strength=cfg["str"], num_inference_steps=cfg["step"]
        ).images[0]

        res = self.color_match(patch_orig, res.resize(patch_orig.size))
        final_patch = patch_orig.copy()
        final_patch.paste(res, (0, 0), mask)

        obj_crop = final_patch.crop(rel_box)
        full_img = img.copy()
        full_img.paste(final_patch, big_box)

        return obj_crop, full_img