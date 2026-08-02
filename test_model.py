"""
test_model.py – comprehensive evaluation of the converted llm.c → HF model
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import time
import math
import os
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--input-model", type=str, default="./hf_model", help="Path to HF model")
parser.add_argument("--output_dir", type=str, required=True, help="Directory to save results")
args = parser.parse_args()

MODEL_PATH = args.input_model
OUT_DIR = args.output_dir
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.float16

os.makedirs(OUT_DIR, exist_ok=True)
log_path = os.path.join(OUT_DIR, "eval_results.txt")
log_file = open(log_path, "w")

def log(msg=""):
    print(msg)
    log_file.write(str(msg) + "\n")
    log_file.flush()

log(f"Loading model from {MODEL_PATH} on {DEVICE}...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH,
    torch_dtype=DTYPE,
    device_map=DEVICE
)
model.eval()
log(f"Model loaded. Parameters: {sum(p.numel() for p in model.parameters()):,}\n")

# ---------------------------------------------------------------
# Helper
# ---------------------------------------------------------------
def generate(prompt, max_new_tokens=80, temperature=0.8, top_p=0.9, do_sample=True):
    inputs = tokenizer(prompt, return_tensors="pt").to(DEVICE)
    with torch.no_grad():
        start = time.time()
        out = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=do_sample,
            temperature=temperature,
            top_p=top_p,
            repetition_penalty=1.15,
            pad_token_id=tokenizer.eos_token_id
        )
        elapsed = time.time() - start
    text = tokenizer.decode(out[0], skip_special_tokens=True)
    new_tokens = out.shape[1] - inputs["input_ids"].shape[1]
    tok_per_sec = new_tokens / elapsed if elapsed > 0 else 0
    return text, tok_per_sec, new_tokens

def print_section(title):
    log("\n" + "=" * 70)
    log(f" {title}")
    log("=" * 70)

# ---------------------------------------------------------------
# 1. Basic story generation (multiple temperatures)
# ---------------------------------------------------------------
print_section("1. Basic Story Generation (different temperatures)")

prompts = [
    "Once upon a time",
    "One day, a little girl named Lily",
    "There was a boy who had a magic",
]

for temp in [0.5, 0.8, 1.1]:
    log(f"\n--- Temperature = {temp} ---")
    for p in prompts:
        text, tps, ntok = generate(p, max_new_tokens=60, temperature=temp)
        log(f"\nPrompt: {p}")
        log(text)
        log(f"[{ntok} tokens | {tps:.1f} tok/s]")

# ---------------------------------------------------------------
# 2. Continuation / longer context
# ---------------------------------------------------------------
print_section("2. Longer Continuations")

long_prompts = [
    "Once upon a time, there was a little boy named Tim. He lived in a small house with his mom and dad. One sunny morning, Tim decided to go exploring in the forest near his house.",
    "Lily loved her red balloon. She took it everywhere. One windy day, the balloon slipped from her hand and floated high into the sky.",
]

for p in long_prompts:
    text, tps, ntok = generate(p, max_new_tokens=100, temperature=0.8)
    log(f"\nPrompt:\n{p}\n")
    log("Continuation:")
    log(text)
    log(f"[{ntok} tokens | {tps:.1f} tok/s]")

# ---------------------------------------------------------------
# 3. Different styles / constraints
# ---------------------------------------------------------------
print_section("3. Style & Constraint Tests")

style_prompts = [
    "Write a short story about a brave mouse:",
    "Once upon a time there was a dragon who was afraid of",
    "The little robot woke up and said",
    "In a land far away, the king had a problem:",
]

for p in style_prompts:
    text, _, _ = generate(p, max_new_tokens=70, temperature=0.85)
    log(f"\nPrompt: {p}")
    log(text)

# ---------------------------------------------------------------
# 4. Greedy vs Sampling comparison
# ---------------------------------------------------------------
print_section("4. Greedy (temp=0) vs Sampling")

prompt = "Once upon a time, there was a little girl who"
log(f"Prompt: {prompt}\n")

log("--- Greedy ---")
text_g, _, _ = generate(prompt, max_new_tokens=50, do_sample=False, temperature=1.0)
log(text_g)

log("\n--- Sampling (temp=0.8) ---")
text_s, _, _ = generate(prompt, max_new_tokens=50, temperature=0.8)
log(text_s)

# ---------------------------------------------------------------
# 5. Repetition / degeneration check
# ---------------------------------------------------------------
print_section("5. Long Generation (check for repetition)")

text, tps, ntok = generate(
    "Once upon a time",
    max_new_tokens=200,
    temperature=0.9
)
log(text)
log(f"\n[{ntok} tokens | {tps:.1f} tok/s]")

# ---------------------------------------------------------------
# 6. Simple perplexity-style check on a few held-out style sentences
# ---------------------------------------------------------------
print_section("6. Rough Loss / Perplexity on sample sentences")

test_sentences = [
    "Once upon a time, there was a little girl named Lily who lived in a small house.",
    "The dog ran quickly through the park and found a big red ball.",
    "Tim and his sister went to the store to buy some milk and bread.",
    "The sun was shining brightly in the blue sky above the green trees.",
]

model.eval()
total_loss = 0.0
total_tokens = 0

with torch.no_grad():
    for sent in test_sentences:
        inputs = tokenizer(sent, return_tensors="pt").to(DEVICE)
        labels = inputs["input_ids"].clone()
        outputs = model(**inputs, labels=labels)
        loss = outputs.loss.item()
        n_tokens = labels.numel()
        total_loss += loss * n_tokens
        total_tokens += n_tokens
        ppl = math.exp(loss)
        log(f"Loss: {loss:.4f} | PPL: {ppl:.2f} | \"{sent[:60]}...\"")

avg_loss = total_loss / total_tokens
avg_ppl = math.exp(avg_loss)
log(f"\nAverage loss: {avg_loss:.4f}")
log(f"Average perplexity: {avg_ppl:.2f}")

# ---------------------------------------------------------------
print_section("Done")
log("All tests completed.")
log(f"Results saved to: {log_path}")
log_file.close()
