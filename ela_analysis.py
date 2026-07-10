import os
import numpy as np
from PIL import Image, ImageChops, ImageEnhance

_ELA_QUALITY = 90    
_REAL_MEAN   = 5.0  
_FAKE_MEAN   = 20.0 


def perform_ela(img_path: str, quality: int = _ELA_QUALITY) -> Image.Image:
    
    original = Image.open(img_path).convert("RGB")
    temp_path = "_ela_tmp.jpg"
    original.save(temp_path, "JPEG", quality=quality)
    resaved  = Image.open(temp_path)
    ela_img  = ImageChops.difference(original, resaved)
    extrema  = ela_img.getextrema()
    max_diff = max([ex[1] for ex in extrema]) or 1
    ela_img  = ImageEnhance.Brightness(ela_img).enhance(255.0 / max_diff)
    os.remove(temp_path)
    return ela_img


def ela_tamper_score(img_path: str) -> dict:

    ela_img  = perform_ela(img_path)
    arr      = np.array(ela_img, dtype=np.float32)
    mean_diff = float(np.mean(arr))


    fake_prob = (mean_diff - _REAL_MEAN) / (_FAKE_MEAN - _REAL_MEAN)
    fake_prob = float(np.clip(fake_prob, 0.0, 1.0))

    return {
        "ela_image": ela_img,
        "mean_diff": round(mean_diff, 2),
        "fake_prob": round(fake_prob, 3),
    }
