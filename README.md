# RM Flux Captioner PRO

A custom ComfyUI node suite that turns an input image into a ready-to-use Flux2 prompt using local LLMs via [Ollama](https://ollama.com).

## How it works

1. **Stage 1 — Vision description**: a vision model (LLaVA) describes the input image.
2. **Stage 2 — Prompt rewrite**: Mistral rewrites that description into a Flux2-ready positive prompt, using a configurable style preset.
3. **Stage 3 — Negative prompt** *(optional, bypassable)*: Mistral generates a matching negative prompt.

Style presets are defined in `system_prompts.json` — add your own by editing that file and restarting ComfyUI.

## Nodes included

- Image captioner / prompt builder (`I2Caption.py`)
- Resolution helper (`I2Caption_resolution.py`)
- Latent helper (`I2Caption_latent.py`)
- Model loader (`I2Caption_loader.py`)

## Requirements

- [ComfyUI](https://github.com/comfyanonymous/ComfyUI)
- [Ollama](https://ollama.com) running locally, with the following models pulled:
  ```bash
  ollama pull llava
  ollama pull mistral
  ```

## Installation

Clone this repo into your `ComfyUI/custom_nodes` directory:

```bash
cd ComfyUI/custom_nodes
git clone https://github.com/MrMorehole/rm-flux-captioner-pro.git
```

Restart ComfyUI. The nodes will appear under their display names in the node menu.

## Example workflow

![Example workflow](workflow.png)

## License

This project is licensed under the **GNU Affero General Public License v3.0 (AGPL-3.0)**. See the [LICENSE](LICENSE) file for the full text.

In short: you're free to use, modify, and distribute this software, but if you modify it and run it as a network service (e.g. host it for others to use), you must make the source code of your modified version available to those users.
