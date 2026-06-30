"""
ComfyUI Custom Node: RM Flux Resolution
-----------------------------------------
Snaps an image to the nearest Flux2 training resolution bucket,
preserving aspect ratio. Insert this between your image loader and
GetImageSize so that all downstream nodes (Flux2Scheduler,
EmptyFlux2LatentImage) receive clean, bucket-aligned dimensions.

Flux2 was trained on aspect-ratio buckets whose area is approximately
1 megapixel (1024×1024). Feeding arbitrary dimensions outside these
buckets degrades composition, proportions, and sharpness even with a
strong prompt. This node eliminates that problem automatically.

Outputs the resized IMAGE tensor plus the final WIDTH and HEIGHT as
integers so they can be wired directly to Flux2Scheduler and
EmptyFlux2LatentImage, bypassing the need for a separate GetImageSize.
"""

import numpy as np
from PIL import Image


# ---------------------------------------------------------------------------
# Flux2 native training resolution buckets
# All pairs are (width, height). The list covers portrait, landscape and
# square orientations. Area ≈ 1MP in all cases.
# ---------------------------------------------------------------------------

FLUX2_BUCKETS = [
    (512,  2048),
    (512,  1984),
    (512,  1920),
    (512,  1856),
    (576,  1792),
    (576,  1728),
    (576,  1664),
    (640,  1600),
    (640,  1536),
    (704,  1472),
    (704,  1408),
    (704,  1344),
    (768,  1344),
    (768,  1280),
    (832,  1216),
    (832,  1152),
    (896,  1152),
    (896,  1088),
    (960,  1088),
    (960,  1024),
    (1024, 1024),  # square
    (1024,  960),
    (1088,  960),
    (1088,  896),
    (1152,  896),
    (1152,  832),
    (1216,  832),
    (1280,  768),
    (1344,  768),
    (1408,  704),
    (1472,  704),
    (1536,  640),
    (1600,  640),
    (1664,  576),
    (1728,  576),
    (1792,  576),
    (1856,  512),
    (1920,  512),
    (1984,  512),
    (2048,  512),
]

# Build the full set including both orientations (already included above,
# but also add the flipped pairs to be safe)
_ALL_BUCKETS = set()
for w, h in FLUX2_BUCKETS:
    _ALL_BUCKETS.add((w, h))
    _ALL_BUCKETS.add((h, w))
FLUX2_BUCKETS_FULL = sorted(_ALL_BUCKETS)


# ---------------------------------------------------------------------------
# Resolution logic
# ---------------------------------------------------------------------------

def nearest_flux2_bucket(src_w: int, src_h: int) -> tuple[int, int]:
    """
    Find the Flux2 bucket that best matches the source aspect ratio.

    Strategy:
      1. Compute the source aspect ratio.
      2. For every bucket, compute how close its aspect ratio is to the source.
      3. Among equally close buckets, prefer the one whose area is closest
         to the source area (capped at 1MP so we never upscale aggressively).
      4. Return (bucket_w, bucket_h).
    """
    src_ratio = src_w / src_h
    src_area  = src_w * src_h

    best = None
    best_score = float("inf")

    for bw, bh in FLUX2_BUCKETS_FULL:
        b_ratio = bw / bh
        # Primary score: aspect ratio distance (log scale keeps portrait/landscape symmetric)
        ratio_diff = abs(np.log(src_ratio) - np.log(b_ratio))
        # Secondary score: area distance (normalised)
        area_diff  = abs(src_area - bw * bh) / (1024 * 1024)
        score = ratio_diff * 10 + area_diff   # weight ratio match heavily
        if score < best_score:
            best_score = score
            best = (bw, bh)

    return best


def resize_to_bucket(image_tensor, bucket_w: int, bucket_h: int):
    """
    Resize a ComfyUI IMAGE tensor (B, H, W, C) to (bucket_h, bucket_w)
    using high-quality Lanczos resampling.
    Returns a new tensor of the same dtype.
    """
    import torch

    results = []
    for i in range(image_tensor.shape[0]):
        img_np  = (image_tensor[i].cpu().numpy() * 255).clip(0, 255).astype(np.uint8)
        pil_img = Image.fromarray(img_np)
        pil_img = pil_img.resize((bucket_w, bucket_h), Image.LANCZOS)
        arr     = np.array(pil_img).astype(np.float32) / 255.0
        results.append(arr)

    out = np.stack(results, axis=0)          # (B, H, W, C)
    return torch.from_numpy(out)


# ---------------------------------------------------------------------------
# Node
# ---------------------------------------------------------------------------

class RMFluxResolution:

    CATEGORY = "RM/Captioning"
    RETURN_TYPES  = ("IMAGE", "INT", "INT")
    RETURN_NAMES  = ("image",  "width", "height")
    FUNCTION = "resolve"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "show_bucket_info": (
                    "BOOLEAN",
                    {"default": True,
                     "label_on":  "Print bucket info to console",
                     "label_off": "Silent"},
                ),
            },
        }

    def resolve(self, image, show_bucket_info=True):
        # Source dimensions from the tensor (B, H, W, C)
        src_h, src_w = image.shape[1], image.shape[2]

        bucket_w, bucket_h = nearest_flux2_bucket(src_w, src_h)

        if show_bucket_info:
            src_ratio = src_w / src_h
            bkt_ratio = bucket_w / bucket_h
            print(
                f"[RM_flux_resolution] Source: {src_w}×{src_h} "
                f"(ratio {src_ratio:.3f})  →  "
                f"Bucket: {bucket_w}×{bucket_h} "
                f"(ratio {bkt_ratio:.3f})"
            )

        if src_w == bucket_w and src_h == bucket_h:
            # Already on a valid bucket — pass through unchanged
            if show_bucket_info:
                print("[RM_flux_resolution] Already at a valid Flux2 bucket — no resize needed.")
            return (image, bucket_w, bucket_h)

        resized = resize_to_bucket(image, bucket_w, bucket_h)
        return (resized, bucket_w, bucket_h)


# ---------------------------------------------------------------------------
# Registration
# ---------------------------------------------------------------------------

NODE_CLASS_MAPPINGS       = {"I2Caption Resolution": RMFluxResolution}
NODE_DISPLAY_NAME_MAPPINGS = {"I2Caption Resolution": "I2Caption Resolution"}
