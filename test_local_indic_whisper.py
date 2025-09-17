#!/usr/bin/env python3
"""Single consolidated test to verify loading of local Indic Whisper (.bin) into DeSTA2.5.
Run: python test_local_indic_whisper.py
Adjust LOCAL_WHISPER below if path differs.
"""
import os, sys, torch, json, time
sys.path.insert(0, os.path.dirname(__file__))

from desta.models.modeling_desta25 import DeSTA25Config, WhisperPerception

LOCAL_WHISPER = "/home/daksh/MTP/DeSTA2/Models/hindi_models/whisper-medium-hi_alldata_multigpu"

def main():
    print("== Local Indic Whisper Load Test ==")
    if not os.path.isdir(LOCAL_WHISPER):
        print(f"[FAIL] Directory not found: {LOCAL_WHISPER}")
        return
    cfg_path = os.path.join(LOCAL_WHISPER, 'config.json')
    bin_path = os.path.join(LOCAL_WHISPER, 'pytorch_model.bin')
    print(f"Config exists: {os.path.exists(cfg_path)}  |  Weights exists: {os.path.exists(bin_path)}")
    if os.path.exists(cfg_path):
        with open(cfg_path) as f:
            c = json.load(f)
        print(f"Model type: {c.get('model_type')} d_model: {c.get('d_model')} layers: {c.get('encoder_layers')}")
    
    print("Creating DeSTA25Config (optimized for fast loading)...")
    config = DeSTA25Config(
        llm_model_id="microsoft/DialoGPT-medium", # smaller for test
        encoder_model_id="openai/whisper-medium",  # architecture reference
        whisper_local_weights=LOCAL_WHISPER,
        whisper_force_manual_load=True,  # Use optimized Strategy 0
        connector_mode="qformer_1",
        qformer_num_hidden_layers=2,
        prompt_size=64,
    )
    
    print("Loading WhisperPerception with optimizations...")
    print("💡 Tips for faster loading:")
    print("   - Strategy 0 uses optimized manual loading (2-5 minutes)")
    print("   - Decoder will be removed automatically to save memory")
    print("   - Loading progress will be shown with timestamps")
    
    t0 = time.time()
    perception = WhisperPerception(config)
    dt = time.time() - t0
    print(f"Loaded Whisper in {dt:.2f}s")
    
    w = perception.whisper
    print(f"Whisper class: {w.__class__.__name__}")
    print(f"Hidden size: {w.config.d_model}  Layers: {w.config.encoder_layers}")
    
    # Parameter checksum for determinism check
    first_params = list(w.state_dict().items())[:5]
    for k,v in first_params:
        print(f"Param {k} | shape {tuple(v.shape)} | sum {float(v.float().sum()):.2f}")
    print("[OK] Local Indic Whisper loaded successfully.")

if __name__ == "__main__":
    main()
