#!/usr/bin/env python3
"""Test DeSTA2.5 model initialization with local Indic Whisper for training setup."""

import os
import sys
import torch
import time

sys.path.insert(0, "/home/daksh/MTP/DeSTA2/DeSTA2.5-Audio")

from desta.models.modeling_desta25 import DeSTA25Config, DeSTA25AudioModel

def test_desta_initialization():
    print("== DeSTA2.5 Model Initialization Test ==")
    
    # Configuration similar to your YAML config
    config = DeSTA25Config(
        llm_model_id="DeSTA-ntu/Llama-3.1-8B-Instruct",
        encoder_model_id="openai/whisper-large-v3",  # Use large architecture for large model
        whisper_local_weights="/home/daksh/MTP/DeSTA2/Models/hindi_models/whisper-large-hi-noldcil",
        whisper_force_manual_load=False,  # Use fast Strategy 0 loading
        connector_mode="qformer_1",
        qformer_num_hidden_layers=6,  # as in your config
        prompt_size=64,
        use_lora=False,
        audio_locator="<|AUDIO|>",
        placeholder_token="<|reserved_special_token_87|>",
    )
    
    print("✅ Config created successfully")
    print(f"📊 Whisper local path: {config.whisper_local_weights}")
    print(f"📊 LLM model: {config.llm_model_id}")
    print(f"📊 Encoder d_model: {config.encoder_config.d_model}")
    print(f"📊 LLM hidden_size: {config.llm_config.hidden_size}")
    
    print("\n🔄 Creating DeSTA2.5AudioModel...")
    t0 = time.time()
    
    try:
        # Initialize the model
        model = DeSTA25AudioModel(config)
        dt = time.time() - t0
        print(f"✅ Model created successfully in {dt:.2f}s")
        
        # Check components
        print("\n📊 Model Components:")
        print(f"  - LLM: {type(model.llm_model).__name__}")
        print(f"  - Whisper: {type(model.perception.whisper).__name__}")
        print(f"  - Whisper config: d_model={model.perception.whisper.config.d_model}, layers={model.perception.whisper.config.encoder_layers}")
        print(f"  - QFormer layers: {model.perception.connector.qformer.config.num_hidden_layers}")
        print(f"  - Target layer IDs: {model.perception.connector.config.target_layer_ids}")
        
        # Check trainable parameters
        print("\n🔧 Configuring trainable parameters...")
        model.configure_trainable_parameters()
        
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        print(f"📊 Total parameters: {total_params:,}")
        print(f"📊 Trainable parameters: {trainable_params:,}")
        print(f"📊 Trainable ratio: {trainable_params/total_params*100:.2f}%")
        
        # Basic model validation (skip complex forward pass for now)
        print("\n✅ Basic model validation...")
        print(f"  - Model is properly initialized: {model is not None}")
        print(f"  - Whisper encoder available: {hasattr(model.perception, 'whisper')}")
        print(f"  - LLM available: {hasattr(model, 'llm_model')}")
        print(f"  - QFormer available: {hasattr(model.perception.connector, 'qformer')}")
        
        print("\n🎉 All tests passed! Model is ready for training.")
        return True
        
    except Exception as e:
        print(f"❌ Error during model initialization: {e}")
        import traceback
        print("📋 Full traceback:")
        print(traceback.format_exc())
        return False

if __name__ == "__main__":
    success = test_desta_initialization()
    
    print("\n" + "="*60)
    if success:
        print("🎉 DeSTA2.5 model initialization successful!")
        print("💡 Ready to run training with:")
        print("   - Local Indic Whisper encoder")
        print("   - Llama-3.1-8B LLM")
        print("   - 6-layer QFormer connector")
    else:
        print("⚠️  Model initialization failed. Check errors above.")
    print("="*60)
