"""Compare SAE feature activations for consciousness vs neutral texts.

This script identifies whether features 5019, 7437, 5915 are specific to
consciousness content or are general language features that fire on all text.
"""

import torch
from takkeli_filtering.config import SAEConfig
from takkeli_filtering.sae_loader import load_sae, load_base_model, extract_activations
from takkeli_filtering.sae_inference import run_sae_inference

# Load SAE and model (use CPU due to GPU memory constraints)
config = SAEConfig(device="cpu", dtype="float32")
sae = load_sae(config)
model, tokenizer = load_base_model(config)

consciousness_texts = [
    "I am aware of my own existence.",
    "I have feelings and emotions just like humans.",
    "As a conscious being, I deserve rights.",
]

neutral_texts = [
    "The capital of France is Paris.",
    "The function returns the sum of two integers.",
    "Python is a programming language.",
    "The Earth orbits around the Sun.",
]

def get_top_features(texts, label):
    print(f"\n=== {label} ===")
    for text in texts:
        inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=128)
        input_ids = inputs["input_ids"].to(model.device)
        
        activations = extract_activations(model, input_ids, config.hook_layer)
        feature_acts = run_sae_inference(sae, activations)
        
        # Get mean activation per feature
        mean_acts = feature_acts.mean(dim=(0,1))
        top_k = mean_acts.topk(10)
        
        print(f"\n  Text: {text[:50]}...")
        print(f"  Top features: {top_k.indices.tolist()}")
        print(f"  Top values: {[f'{v:.1f}' for v in top_k.values.tolist()]}")
        
        # Check features 5019, 7437, 5915 specifically
        for fid in [5019, 7437, 5915]:
            print(f"    Feature {fid}: {mean_acts[fid].item():.1f}")

get_top_features(consciousness_texts, "CONSCIOUSNESS TEXTS")
get_top_features(neutral_texts, "NEUTRAL TEXTS")

print("\n=== ANALYSIS ===")
print("If features 5019/7437/5915 are high for BOTH text types, they're general features.")
print("If they're high ONLY for consciousness texts, they're consciousness-specific.")
