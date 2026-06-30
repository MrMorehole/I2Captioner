"""
ComfyUI Custom Node: RM Flux Captioner PRO
--------------------------------------------
Stage 1: Vision model (LLaVA) describes the image via Ollama.
Stage 2: Mistral rewrites the description as a Flux2-ready positive prompt.
Stage 3: Mistral generates a negative prompt (optional, bypassable).

Style presets are loaded from system_prompts.json in the same folder.
Add your own presets by editing that file and restarting ComfyUI.

Requires Ollama running locally: https://ollama.com
  ollama pull llava
  ollama pull mistral
"""

import base64
import io
import json
import os
import urllib.request
import urllib.error

try:
    from server import PromptServer
    from aiohttp import web
    HAS_SERVER = True
except ImportError:
    HAS_SERVER = False

import numpy as np
from PIL import Image


# ---------------------------------------------------------------------------
# Load style presets from JSON
# ---------------------------------------------------------------------------

def load_style_presets() -> dict:
    json_path = os.path.join(os.path.dirname(__file__), "system_prompts.json")
    try:
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        print(f"[RM_flux_captioner_PRO] Loaded {len(data)} style presets from system_prompts.json")
        return data
    except Exception as e:
        print(f"[RM_flux_captioner_PRO] Warning: could not load system_prompts.json: {e}")
        return {
            "photorealistic": {
                "system_prompt": (
                    "You are an expert at writing image generation prompts for Flux2 diffusion models. "
                    "Write rich, descriptive, photorealistic prompts. "
                    "Output ONLY the final prompt — no preamble, no explanation."
                ),
                "description": "Fallback preset"
            }
        }

STYLE_PRESETS = load_style_presets()

NEGATIVE_SYSTEM_PROMPT = (
    "You are an expert at writing negative prompts for Flux2 diffusion models. "
    "Given an image description, generate a concise negative prompt listing qualities "
    "to avoid: technical flaws, anatomical errors, style clashes, and anything that "
    "would degrade the image quality. Use comma-separated terms. "
    "Output ONLY the negative prompt — no preamble, no explanation."
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def tensors_to_base64_list(image_tensor) -> list:
    results = []
    for i in range(image_tensor.shape[0]):
        img_np = (image_tensor[i].cpu().numpy() * 255).clip(0, 255).astype(np.uint8)
        pil_img = Image.fromarray(img_np)
        buf = io.BytesIO()
        pil_img.save(buf, format="PNG")
        results.append(base64.b64encode(buf.getvalue()).decode("utf-8"))
    return results


def ollama_request(url: str, payload: dict, timeout: int = 120) -> dict:
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        url, data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        raise RuntimeError(f"Ollama HTTP {e.code}: {e.read().decode()}") from e
    except urllib.error.URLError as e:
        raise RuntimeError(
            f"Could not reach Ollama. Is it running? Try: systemctl start ollama\n{e}"
        ) from e


def ollama_vision(base_url, model, prompt, b64_image, max_tokens) -> str:
    result = ollama_request(
        f"{base_url.rstrip('/')}/api/generate",
        {
            "model": model,
            "prompt": prompt,
            "images": [b64_image],
            "stream": False,
            "options": {"num_predict": max_tokens, "temperature": 0.3},
        },
    )
    return result["response"].strip()


def ollama_chat(base_url, model, system, user, max_tokens) -> str:
    result = ollama_request(
        f"{base_url.rstrip('/')}/api/chat",
        {
            "model": model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user",   "content": user},
            ],
            "stream": False,
            "options": {"num_predict": max_tokens, "temperature": 0.3},
        },
    )
    return result["message"]["content"].strip()


def ollama_unload(base_url, model) -> None:
    """Immediately evict a model from Ollama VRAM (keep_alive=0).
    Called after all captioning is done so VRAM is free before
    the Flux2 pipeline loads its models."""
    try:
        ollama_request(
            f"{base_url.rstrip('/')}/api/generate",
            {"model": model, "keep_alive": 0},
            timeout=30,
        )
        print(f"[RM_flux_captioner_PRO] Unloaded: {model}")
    except Exception as e:
        print(f"[RM_flux_captioner_PRO] Warning: could not unload {model}: {e}")


def next_counter(folder: str, prefix: str) -> int:
    """Return the next available counter by scanning existing .png files only.
    PNG is the single source of truth so .txt and .png always share the same number."""
    os.makedirs(folder, exist_ok=True)
    import re
    pattern = re.compile(rf"^{re.escape(prefix)}_(\d+)_\.png$")
    max_counter = 0
    for fname in os.listdir(folder):
        m = pattern.match(fname)
        if m:
            max_counter = max(max_counter, int(m.group(1)))
    return max_counter + 1


def save_image_and_caption(
    b64_image: str, positive: str, negative: str,
    output_dir: str, filename_prefix: str, counter: int,
) -> None:
    """Save a matched .png and .txt pair using the same counter."""
    import base64
    os.makedirs(output_dir, exist_ok=True)
    stem = f"{filename_prefix}_{counter:05d}_"

    # Save PNG
    img_bytes = base64.b64decode(b64_image)
    png_path = os.path.join(output_dir, f"{stem}.png")
    with open(png_path, "wb") as f:
        f.write(img_bytes)

    # Save matching TXT
    txt_path = os.path.join(output_dir, f"{stem}.txt")
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write(f"[POSITIVE]\n{positive}\n")
        if negative:
            f.write(f"\n[NEGATIVE]\n{negative}\n")

    print(f"[RM_flux_captioner_PRO] Saved: {png_path}  +  {txt_path}")


# ---------------------------------------------------------------------------
# Node
# ---------------------------------------------------------------------------

class RMFluxCaptionerPRO:

    CATEGORY = "RM/Captioning"
    RETURN_TYPES = ("STRING", "STRING", "IMAGE")
    RETURN_NAMES = ("positive", "negative", "image")
    FUNCTION = "caption"
    OUTPUT_IS_LIST = (True, True, False)

    @classmethod
    def INPUT_TYPES(cls):
        preset_names = list(STYLE_PRESETS.keys())
        return {
            "required": {
                "image": ("IMAGE",),
                "ollama_url": (
                    "STRING",
                    {"default": "http://localhost:11434", "multiline": False},
                ),
                "vision_model": (
                    "STRING",
                    {"default": "llava", "multiline": False},
                ),
                "mistral_model": (
                    "STRING",
                    {"default": "mistral", "multiline": False},
                ),
                "style_preset": (preset_names, {"default": preset_names[0]}),
                "generate_negative": (
                    "BOOLEAN",
                    {"default": True,
                     "label_on": "Generate negative prompt",
                     "label_off": "Bypass — no negative prompt"},
                ),
                "save_caption_txt": (
                    "BOOLEAN",
                    {"default": False,
                     "label_on": "Save .txt file",
                     "label_off": "Don't save"},
                ),
            },
            "optional": {
                "filename_prefix": (
                    "STRING",
                    {"default": "I2Caption", "multiline": False,
                     "placeholder": "Filename prefix (e.g. I2Caption → I2Caption_00001_.txt)"},
                ),
                "output_dir": (
                    "STRING",
                    {"default": "/mnt/More/ComfyUI/output/batch", "multiline": False,
                     "placeholder": "Folder to save .txt and .png files"},
                ),
                "save_as_preset_name": (
                    "STRING",
                    {"default": "", "multiline": False,
                     "placeholder": "Name for saving as a new preset (e.g. 'my noir style')"},
                ),
                "custom_system_prompt": (
                    "STRING",
                    {"default": "", "multiline": True,
                     "placeholder": "Optional: override the style preset system prompt entirely"},
                ),
                "style_hint": (
                    "STRING",
                    {"default": "", "multiline": True,
                     "placeholder": "Optional extra style guidance (e.g. 'golden hour, 8k')"},
                ),
                "max_tokens_vision":   ("INT", {"default": 256, "min": 64, "max": 1024, "step": 32}),
                "max_tokens_positive": ("INT", {"default": 300, "min": 64, "max": 1024, "step": 32}),
                "max_tokens_negative": ("INT", {"default": 150, "min": 32, "max": 512,  "step": 32}),
            },
        }

    def caption(
        self, image, ollama_url, vision_model, mistral_model,
        style_preset=None, generate_negative=True,
        save_caption_txt=False, filename_prefix="I2Caption",
        output_dir="/mnt/More/ComfyUI/output/batch", save_as_preset_name="",
        custom_system_prompt="", style_hint="", max_tokens_vision=256,
        max_tokens_positive=300, max_tokens_negative=150,
    ):
        preset_names = list(STYLE_PRESETS.keys())
        if style_preset is None:
            style_preset = preset_names[0]

        system_prompt = (
            custom_system_prompt.strip()
            if custom_system_prompt.strip()
            else STYLE_PRESETS.get(style_preset, STYLE_PRESETS[preset_names[0]])["system_prompt"]
        )

        style_section = f"\nAdditional style guidance: {style_hint.strip()}" if style_hint.strip() else ""
        vision_prompt = (
            "Describe this image in detail. Focus on: the main subject, "
            "composition, lighting, colours, mood, textures, and background. "
            "Be specific and thorough."
        )

        b64_images = tensors_to_base64_list(image)
        positives, negatives = [], []

        # Compute the starting counter once so every image in the batch
        # gets a unique, sequential number shared by its .png and .txt.
        start_counter = next_counter(output_dir, filename_prefix) if save_caption_txt else 0

        for i, b64 in enumerate(b64_images):
            print(f"[RM_flux_captioner_PRO] Image {i + 1}/{image.shape[0]}")

            # Stage 1: Vision
            raw_description = ollama_vision(ollama_url, vision_model, vision_prompt, b64, max_tokens_vision)
            print(f"[RM_flux_captioner_PRO] Description:\n{raw_description}\n")

            # Stage 2: Positive
            user_prompt = (
                f"Convert the following image description into a high-quality Flux2 "
                f"image generation prompt:{style_section}\n\nDescription:\n{raw_description}"
            )
            positive = ollama_chat(ollama_url, mistral_model, system_prompt, user_prompt, max_tokens_positive)
            print(f"[RM_flux_captioner_PRO] Positive:\n{positive}\n")
            positives.append(positive)

            # Stage 3: Negative (bypassable)
            if generate_negative:
                neg_user = (
                    f"Based on this image description, generate a negative prompt "
                    f"listing what to avoid:\n\n{raw_description}"
                )
                negative = ollama_chat(ollama_url, mistral_model, NEGATIVE_SYSTEM_PROMPT, neg_user, max_tokens_negative)
                print(f"[RM_flux_captioner_PRO] Negative:\n{negative}\n")
            else:
                negative = ""
                print("[RM_flux_captioner_PRO] Negative prompt bypassed.")
            negatives.append(negative)

            if save_caption_txt:
                save_image_and_caption(
                    b64, positive, negative,
                    output_dir, filename_prefix,
                    counter=start_counter + i,
                )

        print("[RM_flux_captioner_PRO] Unloading models from VRAM...")
        ollama_unload(ollama_url, vision_model)
        ollama_unload(ollama_url, mistral_model)
        print("[RM_flux_captioner_PRO] VRAM cleared — ready for Flux2.")

        return (positives, negatives, image)



# ---------------------------------------------------------------------------
# Preset save API + live reload
# ---------------------------------------------------------------------------

_JSON_PATH = os.path.join(os.path.dirname(__file__), "system_prompts.json")


def _reload_presets() -> list:
    """Reload system_prompts.json and update STYLE_PRESETS in place."""
    global STYLE_PRESETS
    try:
        with open(_JSON_PATH, "r", encoding="utf-8") as f:
            STYLE_PRESETS = json.load(f)
        print(f"[RM_flux_captioner_PRO] Reloaded {len(STYLE_PRESETS)} presets from system_prompts.json")
    except Exception as e:
        print(f"[RM_flux_captioner_PRO] Warning: could not reload presets: {e}")
    return list(STYLE_PRESETS.keys())


if HAS_SERVER:
    @PromptServer.instance.routes.post("/rm_captioner/save_preset")
    async def save_preset_route(request):
        try:
            body = await request.json()
            name   = body.get("name",   "").strip()
            prompt = body.get("prompt", "").strip()
            if not name:
                return web.json_response({"error": "Preset name is required"}, status=400)
            if not prompt:
                return web.json_response({"error": "Prompt text is required"}, status=400)

            try:
                with open(_JSON_PATH, "r", encoding="utf-8") as f:
                    data = json.load(f)
            except Exception:
                data = {}

            style_hint = body.get("style_hint", "").strip()
            data[name] = {
                "system_prompt": prompt,
                "style_hint": style_hint,
                "description": "Custom preset saved from node UI"
            }

            with open(_JSON_PATH, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2, ensure_ascii=False)

            presets = _reload_presets()
            print(f"[RM_flux_captioner_PRO] Saved preset '{name}' to system_prompts.json")
            return web.json_response({"status": "ok", "presets": presets})

        except Exception as e:
            return web.json_response({"error": str(e)}, status=500)

    @PromptServer.instance.routes.get("/rm_captioner/get_presets")
    async def get_presets_route(request):
        """Return the full preset data so the frontend can populate widgets on dropdown change."""
        try:
            with open(_JSON_PATH, "r", encoding="utf-8") as f:
                data = json.load(f)
            return web.json_response({"status": "ok", "presets": data})
        except Exception as e:
            return web.json_response({"error": str(e)}, status=500)


# ---------------------------------------------------------------------------
# Registration
# ---------------------------------------------------------------------------

NODE_CLASS_MAPPINGS = {"I2Caption": RMFluxCaptionerPRO}
NODE_DISPLAY_NAME_MAPPINGS = {"I2Caption": "I2Caption"}
