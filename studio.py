# ============================================================
#  Molab Image Studio — quality-ceiling build
#  Engine (prompt brain): Gemma 4 12B abliterated (huihui-ai), BF16
#  Image: FLUX.1-dev + Flux-Uncensored-V2
#         (merged bundle first; dev + LoRA as fallback)
#  UI: Gradio — describe in Persian or English, the engine writes
#      the FLUX prompt, you can edit it, then generate.
# ============================================================

import subprocess, sys

def _pip(*pkgs):
    for cmd in ([sys.executable, "-m", "pip", "install", "-q", "-U", *pkgs],
                ["uv", "pip", "install", "--system", "-q", *pkgs]):
        try:
            r = subprocess.run(cmd, capture_output=True, text=True, timeout=1200)
            if r.returncode == 0:
                print("installed:", ", ".join(pkgs))
                return
            print("install attempt failed:", (r.stderr or "")[-300:])
        except Exception as e:
            print("install attempt error:", e)
    print("WARNING: could not install", pkgs)

_pip("diffusers", "transformers", "accelerate", "gradio", "safetensors", "sentencepiece", "protobuf")

import os, random, traceback
import torch

print("=" * 60)
if torch.cuda.is_available():
    _p = torch.cuda.get_device_properties(0)
    print(f"GPU: {_p.name} — {_p.total_memory / 2**30:.1f} GB VRAM")
else:
    print("WARNING: no GPU visible!")

def vram():
    if not torch.cuda.is_available():
        return "no GPU"
    a = torch.cuda.memory_allocated() / 2**30
    r = torch.cuda.memory_reserved() / 2**30
    return f"VRAM allocated {a:.1f} GB / reserved {r:.1f} GB"

# ------------------------------------------------------------
# 1) Engine — Gemma 4 12B abliterated (uncensored prompt brain)
# ------------------------------------------------------------
GEMMA_REPO = "huihui-ai/Huihui-gemma-4-12B-it-abliterated"

from transformers import AutoTokenizer
try:
    from transformers import AutoModelForImageTextToText as _AutoLLM
except Exception:
    from transformers import AutoModelForCausalLM as _AutoLLM

print(f"loading engine: {GEMMA_REPO} ...")
tok = AutoTokenizer.from_pretrained(GEMMA_REPO)
llm = _AutoLLM.from_pretrained(GEMMA_REPO, dtype=torch.bfloat16, device_map="cuda")
llm.eval()
print("engine loaded —", vram())

BRAIN_SYSTEM = (
    "You are the prompt engine of a photorealistic image studio. "
    "The user describes an image in Persian or English. "
    "Reply with ONE detailed English prompt for the FLUX.1 image model and nothing else: "
    "no explanations, no quotes, no markdown. "
    "Pack in: subject and appearance, clothing, pose/action, setting and background, "
    "lighting, camera and lens (e.g. shot on Canon EOS R5, 85mm, f/1.8), "
    "skin texture and photorealism cues. Keep it under 180 words."
)

def make_prompt(description: str) -> str:
    msgs = [
        {"role": "system", "content": BRAIN_SYSTEM},
        {"role": "user", "content": description.strip()},
    ]
    try:
        ids = tok.apply_chat_template(msgs, add_generation_prompt=True,
                                      tokenize=True, return_tensors="pt").to("cuda")
        attn = None
    except Exception:
        text = BRAIN_SYSTEM + "\n\nUser: " + description.strip() + "\nPrompt:"
        enc = tok(text, return_tensors="pt").to("cuda")
        ids, attn = enc["input_ids"], enc.get("attention_mask")
    with torch.no_grad():
        out = llm.generate(ids, attention_mask=attn, max_new_tokens=320,
                           do_sample=False, pad_token_id=tok.eos_token_id)
    gen = out[0][ids.shape[1]:]
    text = tok.decode(gen, skip_special_tokens=True).strip()
    # strip any stray block markers the model may add
    for marker in ("POSITIVE:", "Prompt:", "prompt:"):
        if marker in text:
            text = text.split(marker)[-1].strip()
    return text

# ------------------------------------------------------------
# 2) Image — FLUX.1-dev + Uncensored-V2
# ------------------------------------------------------------
from diffusers import FluxPipeline

BACKEND = None
pipe = None

def _load_flux():
    global pipe, BACKEND
    # Path A: one-download merged bundle (FLUX.1-dev with the uncensored LoRA baked in)
    try:
        print("loading image model: shauray/FLUX-UNCENSORED-merged ...")
        p = FluxPipeline.from_pretrained("shauray/FLUX-UNCENSORED-merged",
                                         dtype=torch.bfloat16)
        p = p.to("cuda")
        BACKEND = "FLUX.1-dev + Uncensored (merged bundle)"
        return p
    except Exception as e:
        print("merged bundle failed, falling back to dev + LoRA:", repr(e)[:300])
    # Path B: stock FLUX.1-dev + the Uncensored-V2 LoRA (needs HF token if the repo is gated)
    p = FluxPipeline.from_pretrained("black-forest-labs/FLUX.1-dev",
                                     dtype=torch.bfloat16)
    p.load_lora_weights("enhanceaiteam/Flux-Uncensored-V2")
    p = p.to("cuda")
    BACKEND = "FLUX.1-dev + Flux-Uncensored-V2 LoRA"
    return p

pipe = _load_flux()
print("image model loaded —", BACKEND, "—", vram())

# ------------------------------------------------------------
# 3) Generation
# ------------------------------------------------------------
SIZES = {
    "1024 × 1024 (square)": (1024, 1024),
    "832 × 1216 (portrait)": (832, 1216),
    "1216 × 832 (landscape)": (1216, 832),
    "1344 × 768 (wide)": (1344, 768),
    "768 × 1344 (tall)": (768, 1344),
}

def generate(prompt, size_label, steps, guidance, lora_strength, count, seed):
    if not prompt or not prompt.strip():
        return [], "اول یه پرامپت بساز یا بنویس."
    w, h = SIZES[size_label]
    if BACKEND and BACKEND.endswith("LoRA"):
        try:
            pipe.set_adapters(["default_0"], adapter_weights=[float(lora_strength)])
        except Exception as e:
            print("set_adapters failed:", e)
    imgs = []
    base_seed = int(seed)
    for i in range(int(count)):
        s = base_seed if base_seed >= 0 else random.randint(0, 2**31 - 1)
        g = torch.Generator("cuda").manual_seed(s + i)
        img = pipe(prompt=prompt.strip(), width=w, height=h,
                   num_inference_steps=int(steps),
                   guidance_scale=float(guidance),
                   generator=g).images[0]
        imgs.append(img)
    return imgs, f"ساخته شد — seed پایه: {base_seed} — {vram()}"

# ------------------------------------------------------------
# 4) UI
# ------------------------------------------------------------
import gradio as gr

STATUS = f"موتور: Gemma 4 12B abliterated — عکس: {BACKEND} — {vram()}"

with gr.Blocks(title="Molab Image Studio") as demo:
    gr.Markdown("# 🎨 Molab Image Studio\n"
                "توضیحت رو فارسی یا انگلیسی بنویس؛ موتور Gemma پرامپت حرفه‌ای FLUX رو می‌سازه، "
                "می‌تونی ادیتش کنی، بعد عکس ساخته می‌شه.")
    gr.Textbox(value=STATUS, label="وضعیت مدل‌ها", interactive=False)
    desc = gr.Textbox(label="توضیح صحنه (فارسی یا انگلیسی)", lines=3,
                      placeholder="مثلاً: یه زن جوون با موهای مشکی توی یه کافه‌ی دنج، نور عصر، لبخند ملایم…")
    btn_prompt = gr.Button("🧠 ساخت پرامپت با موتور", variant="primary")
    prompt_box = gr.Textbox(label="پرامپت FLUX (قابل ادیت)", lines=5)
    btn_prompt.click(make_prompt, inputs=desc, outputs=prompt_box)
    with gr.Row():
        size = gr.Dropdown(list(SIZES.keys()), value="1024 × 1024 (square)", label="ابعاد")
        count = gr.Slider(1, 4, value=1, step=1, label="تعداد عکس")
        seed = gr.Number(value=-1, label="Seed ‏(-1 = تصادفی)", precision=0)
    with gr.Row():
        steps = gr.Slider(10, 50, value=28, step=1, label="Steps")
        guidance = gr.Slider(1.0, 6.0, value=3.5, step=0.1, label="Guidance")
        lora = gr.Slider(0.0, 1.5, value=1.0, step=0.05,
                         label="قدرت LoRA ‏(فقط مسیر dev+LoRA)")
    btn_gen = gr.Button("📸 ساخت عکس", variant="primary")
    gallery = gr.Gallery(label="خروجی", columns=2)
    note = gr.Textbox(label="یادداشت", interactive=False)
    btn_gen.click(generate,
                  inputs=[prompt_box, size, steps, guidance, lora, count, seed],
                  outputs=[gallery, note])

demo.queue()
demo.launch(share=True)
