# ============================================================
#  Molab Image Studio — quality-ceiling build (v2, full logging)
#  Engine (prompt brain): Gemma 4 12B abliterated (huihui-ai), BF16
#  Image: FLUX.1-dev + Flux-Uncensored-V2
#         (merged bundle first; dev + LoRA as fallback)
#  UI: Gradio — describe in Persian or English, the engine writes
#      the FLUX prompt, you can edit it, then generate.
#  Everything logs live under the cell: installs, downloads
#  (a watcher prints HF-cache GB + GPU mem every 30 s), loads,
#  and generation steps.
# ============================================================

import sys
try:
    sys.stdout.reconfigure(line_buffering=True)
except Exception:
    pass

def log(*a):
    print(*a, flush=True)

import subprocess

def _pip(*pkgs):
    for cmd in ([sys.executable, "-m", "pip", "install", "-U", *pkgs],
                ["uv", "pip", "install", "--system", *pkgs]):
        log(">>> running:", " ".join(cmd))
        try:
            r = subprocess.run(cmd, timeout=1800)  # streams live to the cell
            if r.returncode == 0:
                log("installed:", ", ".join(pkgs))
                return
            log("install attempt failed with code", r.returncode)
        except Exception as e:
            log("install attempt error:", e)
    log("WARNING: could not install", pkgs)

_pip("diffusers", "transformers", "accelerate", "gradio", "safetensors", "sentencepiece", "protobuf")

import os, random, threading, time
import torch

log("=" * 60)
if torch.cuda.is_available():
    _p = torch.cuda.get_device_properties(0)
    log(f"GPU: {_p.name} — {_p.total_memory / 2**30:.1f} GB VRAM")
else:
    log("WARNING: no GPU visible!")

def vram():
    if not torch.cuda.is_available():
        return "no GPU"
    a = torch.cuda.memory_allocated() / 2**30
    r = torch.cuda.memory_reserved() / 2**30
    return f"VRAM allocated {a:.1f} GB / reserved {r:.1f} GB"

# --- progress watcher: HF cache size + GPU mem every 30 s ---
_stop_watch = threading.Event()

def _hf_cache_bytes():
    total = 0
    for root, dirs, files in os.walk(os.path.expanduser("~/.cache/huggingface")):
        for f in files:
            try:
                total += os.path.getsize(os.path.join(root, f))
            except OSError:
                pass
    return total

def _watch():
    while not _stop_watch.is_set():
        gpu = ""
        if torch.cuda.is_available():
            gpu = f" — GPU mem {torch.cuda.memory_allocated() / 2**30:.1f} GB"
        log(f"  … progress: HF cache {_hf_cache_bytes() / 2**30:.1f} GB{gpu}")
        _stop_watch.wait(30)

threading.Thread(target=_watch, daemon=True).start()

# ------------------------------------------------------------
# 1) Engine — Gemma 4 12B abliterated (uncensored prompt brain)
# ------------------------------------------------------------
GEMMA_REPO = "huihui-ai/Huihui-gemma-4-12B-it-abliterated"

from transformers import AutoTokenizer
try:
    from transformers import AutoModelForImageTextToText as _AutoLLM
except Exception:
    from transformers import AutoModelForCausalLM as _AutoLLM

log(f"⬇ engine: downloading/loading {GEMMA_REPO} (one-time, ~24 GB) …")
_t0 = time.time()
tok = AutoTokenizer.from_pretrained(GEMMA_REPO)
llm = _AutoLLM.from_pretrained(GEMMA_REPO, dtype=torch.bfloat16, device_map="cuda")
llm.eval()
log(f"✔ engine loaded in {time.time() - _t0:.0f}s —", vram())

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
    # Path A: merged bundle, loaded component-by-component straight onto the
    # GPU. (A plain FluxPipeline.from_pretrained + .to("cuda") dies on this
    # bundle with "Cannot copy out of meta tensor" under current diffusers —
    # the bundle itself is a standard FLUX layout, verified file by file.)
    try:
        log("⬇ image model: downloading shauray/FLUX-UNCENSORED-merged (one-time, ~32 GB) …")
        _t = time.time()
        from huggingface_hub import snapshot_download
        from diffusers import (FluxTransformer2DModel, AutoencoderKL,
                               FlowMatchEulerDiscreteScheduler)
        from transformers import (CLIPTextModel, CLIPTokenizer,
                                  T5EncoderModel, T5TokenizerFast)
        snap = snapshot_download("shauray/FLUX-UNCENSORED-merged")
        log("snapshot:", snap)

        def comp(cls, sub, dtype=torch.bfloat16):
            m = cls.from_pretrained(os.path.join(snap, sub), dtype=dtype,
                                    low_cpu_mem_usage=False)
            m = m.to("cuda")
            log(f"  ✔ {sub} on GPU — {vram()}")
            return m

        transformer = comp(FluxTransformer2DModel, "transformer")
        text_encoder = comp(CLIPTextModel, "text_encoder")
        text_encoder_2 = comp(T5EncoderModel, "text_encoder_2")
        vae = comp(AutoencoderKL, "vae", dtype=torch.float32)
        scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(snap, subfolder="scheduler")
        tokenizer = CLIPTokenizer.from_pretrained(snap, subfolder="tokenizer")
        tokenizer_2 = T5TokenizerFast.from_pretrained(snap, subfolder="tokenizer_2")
        p = FluxPipeline(transformer=transformer, text_encoder=text_encoder,
                         text_encoder_2=text_encoder_2, vae=vae,
                         scheduler=scheduler, tokenizer=tokenizer,
                         tokenizer_2=tokenizer_2)
        BACKEND = "FLUX.1-dev + Uncensored (merged bundle)"
        log(f"✔ merged bundle loaded in {time.time() - _t:.0f}s")
        return p
    except Exception as e:
        log("merged bundle failed, falling back to dev + LoRA:", repr(e)[:300])
    # Path B: stock FLUX.1-dev + the Uncensored-V2 LoRA (needs HF token if the repo is gated)
    log("⬇ image model: downloading/loading black-forest-labs/FLUX.1-dev + Uncensored-V2 LoRA …")
    p = FluxPipeline.from_pretrained("black-forest-labs/FLUX.1-dev",
                                     dtype=torch.bfloat16)
    p.load_lora_weights("enhanceaiteam/Flux-Uncensored-V2")
    p = p.to("cuda")
    BACKEND = "FLUX.1-dev + Flux-Uncensored-V2 LoRA"
    return p

pipe = _load_flux()
log("✔ image model loaded —", BACKEND, "—", vram())

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

def _step_log(pipe_, step, timestep, kwargs):
    if step % 5 == 0:
        log(f"  generating… step {step}")
    return kwargs

def generate(prompt, size_label, steps, guidance, lora_strength, count, seed):
    if not prompt or not prompt.strip():
        return [], "اول یه پرامپت بساز یا بنویس."
    w, h = SIZES[size_label]
    if BACKEND and BACKEND.endswith("LoRA"):
        try:
            pipe.set_adapters(["default_0"], adapter_weights=[float(lora_strength)])
        except Exception as e:
            log("set_adapters failed:", e)
    imgs = []
    base_seed = int(seed)
    for i in range(int(count)):
        s = base_seed if base_seed >= 0 else random.randint(0, 2**31 - 1)
        g = torch.Generator("cuda").manual_seed(s + i)
        _t = time.time()
        img = pipe(prompt=prompt.strip(), width=w, height=h,
                   num_inference_steps=int(steps),
                   guidance_scale=float(guidance),
                   generator=g, callback_on_step_end=_step_log).images[0]
        log(f"✔ image {i + 1} done in {time.time() - _t:.0f}s (seed {s + i})")
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

_stop_watch.set()
log("✔ all models ready — launching Gradio …")
demo.queue()
demo.launch(share=True)
