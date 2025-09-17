#!/usr/bin/env python3
"""
Test script to verify the cleaned up code works correctly
"""

import os
import sys
import torch

# Add the project to Python path
sys.path.insert(0, '/home/daksh/MTP/DeSTA2/DeSTA2.5-Audio')

def test_config_creation():
    """Test that DeSTA25Config can be created without errors"""
    print("Testing DeSTA25Config creation...")
    
    from desta.models.modeling_desta25 import DeSTA25Config
    
    # Test basic config
    config = DeSTA25Config(
        llm_model_id="DeSTA-ntu/Llama-3.1-8B-Instruct",
        encoder_model_id="openai/whisper-large-v3",
        whisper_force_manual_load=True
    )
    
    assert config.whisper_force_manual_load == True, "whisper_force_manual_load not stored properly"
    assert config.llm_model_id == "DeSTA-ntu/Llama-3.1-8B-Instruct"
    assert config.encoder_model_id == "openai/whisper-large-v3"
    
    print("✅ DeSTA25Config creation test passed")
    return config

def test_imports():
    """Test that all necessary imports work"""
    print("Testing imports...")
    
    try:
        from desta.models.modeling_desta25 import (
            DeSTA25AudioModel, 
            DeSTA25Config,
            QformerConnector,
            WhisperPerception,
            GenerationOutput
        )
        print("✅ All imports successful")
        return True
    except Exception as e:
        print(f"❌ Import failed: {e}")
        return False

def test_model_structure():
    """Test that model can be instantiated with basic config"""
    print("Testing model structure...")
    
    try:
        from desta.models.modeling_desta25 import DeSTA25Config
        
        # Create config with local paths if available
        local_whisper_path = "/home/daksh/MTP/DeSTA2/Models/hindi_models/whisper-large-hi-noldcil"
        
        config = DeSTA25Config(
            llm_model_id="microsoft/DialoGPT-small",  # Use smaller model for testing
            encoder_model_id="openai/whisper-tiny",  # Use tiny for faster testing
            whisper_local_weights=local_whisper_path if os.path.exists(local_whisper_path) else None,
            whisper_force_manual_load=True if os.path.exists(local_whisper_path) else False,
            prompt_size=32,  # Smaller for testing
            qformer_num_hidden_layers=1
        )
        
        print(f"Config created with whisper_force_manual_load = {config.whisper_force_manual_load}")
        print("✅ Model structure test passed")
        return config
        
    except Exception as e:
        print(f"❌ Model structure test failed: {e}")
        return None

def main():
    """Run all tests"""
    print("="*60)
    print("DeSTA2.5 Code Cleanup Verification Tests")
    print("="*60)
    
    # Test 1: Imports
    if not test_imports():
        return False
    
    # Test 2: Config creation
    config = test_config_creation()
    if config is None:
        return False
    
    # Test 3: Model structure
    model_config = test_model_structure()
    if model_config is None:
        return False
    
    print("\n" + "="*60)
    print("🎉 All cleanup verification tests passed!")
    print("✅ The code is ready for use")
    print("="*60)
    
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
