
import os
import torch
import torch.nn as nn
import gc
import time
from collections import OrderedDict
import logging

from dataclasses import dataclass
from desta.utils.audio import AudioSegment
from transformers import AutoTokenizer, AutoProcessor
from transformers import PretrainedConfig, PreTrainedModel, AutoModelForCausalLM, AutoConfig
from transformers.models.bert.modeling_bert import BertEncoder
from transformers import WhisperForConditionalGeneration, BertConfig
from transformers.models.whisper import WhisperConfig
from safetensors.torch import load_file


def _prepare_audio_context_and_start_positions(
                                             token_list,
                                             audio_locator,
                                             audio_size_list,
                                             transcription_size_list,
                                             placeholder_token
        ):
        assert len(audio_size_list) == len(transcription_size_list), f"audio_size_list and transcription_size_list must have the same length, audio_size_list: {audio_size_list}, transcription_size_list: {transcription_size_list}"

        result = []
        start_positions = []
        for x in token_list:
            if x == audio_locator:
                # start_positions.append(len(result))
                transcription_size = transcription_size_list.pop(0)
                audio_size = audio_size_list.pop(0)

                # result.extend(transcription)
                start_positions.append(len(result))
                result.extend([placeholder_token] * (audio_size))
                result.extend([placeholder_token] * (transcription_size))
            else:
                result.append(x)
                
        return result, start_positions


class QformerConnector(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config

        # Handle different Whisper model architectures
        # For OpenAI Whisper models
        if self.config.encoder_model_id == "openai/whisper-medium":
            self.config.target_layer_ids = [5, 11, 17, 23]
        elif self.config.encoder_model_id == "openai/whisper-small":
            self.config.target_layer_ids = [2, 5, 8, 11]
        elif self.config.encoder_model_id == "openai/whisper-tiny":
            self.config.target_layer_ids = [0, 1, 2, 3]
        elif self.config.encoder_model_id == "openai/whisper-large-v3":
            self.config.target_layer_ids = [7, 15, 23, 31]
        # For custom/local Whisper models (including AI4Bharat Indic Whisper)
        elif hasattr(self.config, 'whisper_local_weights') and self.config.whisper_local_weights:
            # Determine layer configuration based on model path
            model_path_lower = str(self.config.whisper_local_weights).lower()
            if "medium" in model_path_lower:
                self.config.target_layer_ids = [5, 11, 17, 23]  # Same as openai/whisper-medium
            elif "small" in model_path_lower:
                self.config.target_layer_ids = [2, 5, 8, 11]  # Same as openai/whisper-small
            elif "large" in model_path_lower:
                self.config.target_layer_ids = [7, 15, 23, 31]  # Same as openai/whisper-large
            else:
                # Default to medium configuration for unknown custom models
                self.config.target_layer_ids = [5, 11, 17, 23]
            print(f"🔧 Custom Whisper model: {len(self.config.target_layer_ids)} target layers configured")
        else:
            raise NotImplementedError(f"model_id {self.config.encoder_model_id} not implemented")


        self.layer_prompts = nn.ParameterList([
            nn.Parameter(torch.randn(1, self.config.prompt_size, self.config.encoder_config.d_model)) for _ in range(len(self.config.target_layer_ids))]
        )

        self.layer_weights = nn.Parameter(torch.zeros(self.config.prompt_size, len(self.config.target_layer_ids), dtype=torch.float))

        if self.config.connector_mode == "qformer_1":
            # init Qformerblock
            qformer_config = BertConfig()
            qformer_config.num_hidden_layers = self.config.qformer_num_hidden_layers
            qformer_config.num_attention_heads = self.config.encoder_config.encoder_attention_heads
            qformer_config.hidden_size = self.config.encoder_config.d_model
            qformer_config.add_cross_attention = True
            qformer_config.is_decoder = True
            qformer_config._attn_implementation = "eager"

            self.qformer = BertEncoder(qformer_config)
            self.proj = nn.Sequential(
                    nn.LayerNorm(self.config.encoder_config.d_model),
                    nn.Linear(self.config.encoder_config.d_model, self.config.llm_config.hidden_size) # project to llm hidden size
                )
                
            # Load QFormer weights if available
            qformer_local = getattr(self.config, "qformer_local_weights", None)
            if qformer_local and os.path.exists(qformer_local):
                print(f"🔧 Loading QFormer weights from {qformer_local}")
                try:
                    if qformer_local.endswith('.safetensors'):
                        state = load_file(qformer_local)
                    else:
                        state = torch.load(qformer_local, map_location="cpu")
                    missing, unexpected = self.qformer.load_state_dict(state, strict=False)
                    if missing or unexpected:
                        print(f"   QFormer loading: {len(missing)} missing, {len(unexpected)} unexpected keys")
                    else:
                        print(f"   ✅ QFormer weights loaded successfully")
                except Exception as e:
                    print(f"   ❌ Failed to load QFormer weights: {e}")
        else:
            raise NotImplementedError(f"connector_mode {self.config.connector_mode} not implemented")
        

    def forward(self, encoder_hidden_states):
        """
        input: 
            encoder_hidden_states: layerwise hidden states from the encoder
        """
        layer_prompt_outputs = []
        for idx, encoder_hidden_state in enumerate(encoder_hidden_states):
            if idx in self.config.target_layer_ids:
                layer_prompt = self.layer_prompts[self.config.target_layer_ids.index(idx)].expand(encoder_hidden_state.size(0), -1, -1)
                qformer_output = self.qformer(
                    hidden_states=layer_prompt,
                    encoder_hidden_states=encoder_hidden_state,
                )
                layer_prompt_output = qformer_output.last_hidden_state
                layer_prompt_outputs.append(layer_prompt_output)
        
        layer_prompt_outputs = torch.stack(layer_prompt_outputs, dim=0)
        layer_prompt_outputs = layer_prompt_outputs.permute(1, 2, 0, 3)
        self.norm_weights = torch.nn.functional.softmax(self.layer_weights, dim=-1).unsqueeze(-1)
        output = (layer_prompt_outputs * self.norm_weights).sum(dim=2) # (b, prompt_size, d_llm)
        output = self.proj(output)
        
        return output

@dataclass
class GenerationOutput():
    audios: list[str]
    generated_ids: list[torch.Tensor]
    text: list[str]

class WhisperPerception(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        whisper_local = getattr(self.config, "whisper_local_weights", None)
        if whisper_local and os.path.isdir(whisper_local):
            print(f"  🔧 Loading local Whisper model from: {whisper_local}")
            self.whisper = self._load_local_whisper_model(whisper_local)
        elif whisper_local and os.path.isfile(whisper_local):
            print(f"  🔧 Loading Whisper weights from file: {whisper_local}")
            self.whisper = self._load_whisper_from_file(whisper_local)
        else:
            print(f"  🔧 Loading Whisper from HuggingFace: {self.config.encoder_model_id}")
            self.whisper = WhisperForConditionalGeneration.from_pretrained(
                self.config.encoder_model_id, cache_dir=os.getenv("HF_HOME")
            )
        self.connector = QformerConnector(config)

    def _sanitize_state_dict(self, state_dict):
        if 'state_dict' in state_dict and isinstance(state_dict['state_dict'], dict):
            state_dict = state_dict['state_dict']
        if 'model' in state_dict and isinstance(state_dict['model'], dict):
            state_dict = state_dict['model']
        
        new_state = {}
        sample_keys = list(state_dict.keys())[:3]
        print(f"    🔧 Checkpoint sample keys: {sample_keys}")
        
        # Check if keys need "model." prefix added or removed
        has_model_prefix = any(k.startswith('model.') for k in sample_keys)
        has_encoder_decoder_only = any(k.startswith('encoder.') or k.startswith('decoder.') for k in sample_keys)
        
        print(f"    🔧 has_model_prefix: {has_model_prefix}, has_encoder_decoder_only: {has_encoder_decoder_only}")
        
        for k, v in state_dict.items():
            nk = k
            
            # Remove common prefixes first
            if nk.startswith('module.'):
                nk = nk[len('module.') :]
            
            # Handle model prefix logic
            if has_model_prefix:
                # Keep model. prefix (checkpoint already has it - don't remove!)
                # This is the case for our Indic Whisper: model.encoder.*
                pass  # Keep the key as is
            elif has_encoder_decoder_only:
                # Add model. prefix (checkpoint has encoder.* but model expects model.encoder.*)
                if nk.startswith('encoder.') or nk.startswith('decoder.'):
                    nk = 'model.' + nk
            else:
                # Fallback: remove model. prefix (original behavior for other cases)
                if nk.startswith('model.'):
                    nk = nk[len('model.') :]
            
            new_state[nk] = v
            
        print(f"    🔧 After sanitization sample: {list(new_state.keys())[:3]}")
        return new_state

    def _ensure_safetensors(self, local_path):
        """Convert pytorch_model.bin to model.safetensors for faster loading (always recreate for local models)"""
        bin_path = os.path.join(local_path, 'pytorch_model.bin')
        safe_path = os.path.join(local_path, 'model.safetensors')

        if not os.path.exists(bin_path):
            return  # nothing to do if no pytorch_model.bin

        if not getattr(self.config, 'whisper_autoconvert_to_safetensors', True):
            return

        # Delete existing safetensors file to ensure clean replacement
        if os.path.exists(safe_path):
            print('  🗑️ Removing existing model.safetensors to ensure clean replacement...')
            os.remove(safe_path)
        
        # Create new safetensors file with full model (including decoder) for inference
        print('  🔄 Creating model.safetensors with full model weights (including decoder)...')
        try:
            from safetensors.torch import save_file
            
            raw_state = torch.load(bin_path, map_location='cpu', weights_only=True)
            clean_state = self._sanitize_state_dict(raw_state)

            # Keep ALL weights including decoder (needed for inference)
            # Log some key information about what we're saving
            encoder_keys = [k for k in clean_state.keys() if 'encoder' in k]
            decoder_keys = [k for k in clean_state.keys() if 'decoder' in k]
            print(f"    📊 Saving {len(encoder_keys)} encoder keys, {len(decoder_keys)} decoder keys")
            
            save_file(clean_state, safe_path)
            del raw_state, clean_state
            gc.collect()
            
            # Verify the conversion worked
            safe_size = os.path.getsize(safe_path) / (1024**3)
            print(f'  ✅ Created model.safetensors (full model with encoder+decoder, {safe_size:.2f} GB)')
            
        except ImportError:
            print('  ⚠️ safetensors not installed, keeping pytorch_model.bin')
        except Exception as e:
            print(f'  ⚠️ safetensors conversion failed: {e}')

    def _load_local_whisper_model(self, local_path):
        has_pytorch_model = os.path.exists(os.path.join(local_path, 'pytorch_model.bin'))
        has_config = os.path.exists(os.path.join(local_path, 'config.json'))
        has_safetensors = os.path.exists(os.path.join(local_path, 'model.safetensors'))
        pth_files = [f for f in os.listdir(local_path) if f.endswith('.pth') and not f.startswith('rng_state')]
        force_manual = getattr(self.config, 'whisper_force_manual_load', False)
        
        # OLD CODE: Always ensure fresh safetensors for local models (to include full model with decoder)
        # Commented out: Now we only create safetensors when using Strategy 1
        # if has_config and has_pytorch_model:
        #     try:
        #         self._ensure_safetensors(local_path)
        #         has_safetensors = os.path.exists(os.path.join(local_path, 'model.safetensors'))  # Recheck after conversion
        #         print(f'  ✅ Safetensors conversion completed, preferring Strategy 1')
        #     except Exception as e:
        #         print(f'  ⚠️ safetensors conversion skipped: {e}')
        
        print(f'  🔧 Loading preferences: force_manual={force_manual}')
        print('Local Whisper directory contents:')
        print(f"    - config.json: {'✅' if has_config else '❌'}")
        print(f"    - pytorch_model.bin: {'✅' if has_pytorch_model else '❌'}")
        print(f"    - model.safetensors: {'✅' if has_safetensors else '❌'}")
        print(f"    - .pth files: {len(pth_files)}")
        
        if has_pytorch_model:
            model_size = os.path.getsize(os.path.join(local_path, 'pytorch_model.bin')) / (1024**3)
            print(f"    - Model size: {model_size:.2f} GB")

        # Strategy 0: Fast manual loading (RECOMMENDED for large files)
        if has_config and has_pytorch_model and force_manual:
            try:
                print('  Strategy 0: Fast manual loading (optimized for large files)')
                t0 = time.time()
                
                try:
                    local_cfg = WhisperConfig.from_pretrained(local_path)
                except Exception:
                    from transformers.models.whisper.configuration_whisper import WhisperConfig as _WC
                    local_cfg = _WC.from_json_file(os.path.join(local_path, 'config.json'))
                
                print(f"    Config loaded in {time.time() - t0:.2f}s")
                
                model = WhisperForConditionalGeneration(local_cfg)
                print(f"    Architecture created in {time.time() - t0:.2f}s")
                
                print('    Loading weights...')
                raw_state = torch.load(
                    os.path.join(local_path, 'pytorch_model.bin'), 
                    map_location='cpu',
                    weights_only=True 
                )
                print(f"    Weights loaded from disk in {time.time() - t0:.2f}s")
                
                clean_state = self._sanitize_state_dict(raw_state)
                missing, unexpected = model.load_state_dict(clean_state, strict=False)
                print(f"    Strategy 0 completed in {time.time() - t0:.2f}s")
                print(f"    Missing: {len(missing)}, Unexpected: {len(unexpected)}")
                
                if missing or unexpected:
                    print(f"    🔍 Weight loading details:")
                    if missing:
                        print(f"      Missing keys (first 5): {missing[:5]}")
                    if unexpected:
                        print(f"      Unexpected keys (first 5): {unexpected[:5]}")
                
                # Verify weights are loaded
                conv1_weight = model.model.encoder.conv1.weight
                conv1_bias = getattr(model.model.encoder.conv1, 'bias', None)
                embed_pos = model.model.encoder.embed_positions.weight
                print(f"    ✅ Weight verification:")
                print(f"      conv1.weight: shape={conv1_weight.shape}, sum={float(conv1_weight.sum()):.2f}")  # type: ignore
                if conv1_bias is not None:
                    print(f"      conv1.bias: shape={conv1_bias.shape}, sum={float(conv1_bias.sum()):.2f}")  # type: ignore
                print(f"      embed_positions: shape={embed_pos.shape}, sum={float(embed_pos.sum()):.2f}")  # type: ignore
                
                del raw_state, clean_state
                gc.collect()
                
                return model
            except Exception as e:
                print(f"  ❌ Strategy 0 failed: {e}")

        # Strategy 1: Optimized from_pretrained (FAST + STABLE with safetensors)
        if has_config and (has_safetensors or has_pytorch_model) and not force_manual:
            try:
                # Create or refresh safetensors only for Strategy 1
                if not has_safetensors and has_pytorch_model:
                    print('  🔧 Strategy 1: Creating safetensors for optimized loading...')
                    try:
                        self._ensure_safetensors(local_path)
                        has_safetensors = os.path.exists(os.path.join(local_path, 'model.safetensors'))
                    except Exception as e:
                        print(f'  ⚠️ Safetensors creation failed: {e}, using pytorch_model.bin')
                
                use_safetensors = has_safetensors
                strategy_desc = f"{'safetensors' if use_safetensors else 'pytorch_model.bin'}"
                print(f'  Strategy 1: Optimized from_pretrained (using {strategy_desc})')
                t0 = time.time()
                
                model = WhisperForConditionalGeneration.from_pretrained(
                    local_path,
                    cache_dir=os.getenv('HF_HOME'),
                    local_files_only=True,
                    low_cpu_mem_usage=True,
                    device_map=None,  
                    use_safetensors=use_safetensors, 
                    trust_remote_code=False,
                )
                
                # Optimize attention implementation for faster inference
                try:
                    model.config._attn_implementation = "eager"
                except Exception:
                    pass
                
                print(f"    ✅ Strategy 1 completed in {time.time() - t0:.2f}s")
                
                # Quick verification with NaN/inf handling
                conv1_weight = model.model.encoder.conv1.weight
                embed_pos = model.model.encoder.embed_positions.weight
                
                conv1_sum = float(conv1_weight.sum())  # type: ignore
                embed_sum = float(embed_pos.sum())  # type: ignore
                
                # Check for inf/nan values and provide safer reporting
                conv1_status = "OK" if torch.isfinite(conv1_weight).all() else "HAS_INF/NAN"  # type: ignore
                embed_status = "OK" if torch.isfinite(embed_pos).all() else "HAS_INF/NAN"  # type: ignore
                
                print(f"    🔍 Quick verification: conv1_weight sum={conv1_sum:.2f} ({conv1_status}), embed_pos sum={embed_sum:.2f} ({embed_status})")
                
                return model
            except Exception as e:
                print(f'  ❌ Strategy 1 failed: {e}')

        # Strategy 2: Base + manual (backup)
        if has_pytorch_model:
            try:
                print('  Strategy 2: Base architecture + manual loading')
                base_model_id = self.config.encoder_model_id
                base = WhisperForConditionalGeneration.from_pretrained(
                    base_model_id, 
                    cache_dir=os.getenv('HF_HOME'), 
                    low_cpu_mem_usage=True
                )
                
                print('    Loading custom weights...')
                raw_state = torch.load(
                    os.path.join(local_path, 'pytorch_model.bin'), 
                    map_location='cpu',
                    weights_only=True
                )
                clean_state = self._sanitize_state_dict(raw_state)
                missing, unexpected = base.load_state_dict(clean_state, strict=False)
                print(f'    Done Strategy 2 completed (missing={len(missing)}, unexpected={len(unexpected)})')
                
                del raw_state, clean_state
                gc.collect()
                
                return base
            except Exception as e:
                print(f'  Error Strategy 2 failed: {e}')

        # Strategy 3: .pth fallback
        if pth_files:
            try:
                print('  Strategy 3: .pth file fallback')
                base = WhisperForConditionalGeneration.from_pretrained(
                    self.config.encoder_model_id, 
                    cache_dir=os.getenv('HF_HOME'), 
                    low_cpu_mem_usage=True
                )
                pth_files.sort(key=lambda f: os.path.getsize(os.path.join(local_path, f)), reverse=True)
                raw_state = torch.load(os.path.join(local_path, pth_files[0]), map_location='cpu')
                clean_state = self._sanitize_state_dict(raw_state)
                missing, unexpected = base.load_state_dict(clean_state, strict=False)
                print(f'    Strategy 3 completed (missing={len(missing)}, unexpected={len(unexpected)})')
                return base
            except Exception as e:
                print(f'  ❌ Strategy 3 failed: {e}')

        print('  ⚠️  All strategies failed, falling back to base HF model')
        return WhisperForConditionalGeneration.from_pretrained(
            self.config.encoder_model_id, 
            cache_dir=os.getenv('HF_HOME'),
            low_cpu_mem_usage=True
        )

    def _load_whisper_from_file(self, file_path):
        try:
            base = WhisperForConditionalGeneration.from_pretrained(self.config.encoder_model_id, cache_dir=os.getenv('HF_HOME'))
            raw_state = torch.load(file_path, map_location='cpu')
            clean_state = self._sanitize_state_dict(raw_state)
            missing, unexpected = base.load_state_dict(clean_state, strict=False)
            print(f'  ✅ Single file load completed (missing={len(missing)}, unexpected={len(unexpected)})')
            return base
        except Exception as e:
            print(f'  ❌ Single file load failed: {e}; reverting to HF model')
            return WhisperForConditionalGeneration.from_pretrained(self.config.encoder_model_id, cache_dir=os.getenv('HF_HOME'))

    def forward(self, input_features, attention_mask=None, transcription_embeddings_list=None, **kwargs):
        bs = input_features.size(0)

        audio_features = self.forward_whisper(input_features=input_features, transcription_embeddings_list=transcription_embeddings_list)
        speech_feature_lengths = [self.config.prompt_size] * audio_features.size(0) # (b, )
        
        return audio_features, speech_feature_lengths


    def forward_whisper(self, input_features, attention_mask=None, transcription_embeddings_list=None, **kwargs):
        """
        
        """
        bs = input_features.size(0)
        
        expected_seq_length = self.whisper.model.encoder.config.max_source_positions * self.whisper.model.encoder.conv1.stride[0] * self.whisper.model.encoder.conv2.stride[0]

        if input_features.shape[-1] != expected_seq_length:
            raise ValueError(
                f"Whisper expects the mel input features to be of length {expected_seq_length}, but found {input_features.shape[-1]}. Make sure to pad the input mel features to {expected_seq_length}."
            )
        
        # Ensure input dtype matches model weights (important for mixed precision)
        model_dtype = next(self.whisper.model.encoder.parameters()).dtype
        if input_features.dtype != model_dtype:
            input_features = input_features.to(model_dtype)

        inputs_embeds = nn.functional.gelu(self.whisper.model.encoder.conv1(input_features))
        inputs_embeds = nn.functional.gelu(self.whisper.model.encoder.conv2(inputs_embeds))

        inputs_embeds = inputs_embeds.permute(0, 2, 1)
        embed_pos = self.whisper.model.encoder.embed_positions.weight[:self.whisper.model.encoder.config.max_source_positions, :] # @kehan

        hidden_states = inputs_embeds + embed_pos
        features_length = hidden_states.size(1)

        if self.config.connector_mode == "qformer_1":
            layer_prompt_outputs = []
            for idx, encoder_layer in enumerate(self.whisper.model.encoder.layers):
                
                layer_outputs = encoder_layer(
                    hidden_states,
                    attention_mask=None,
                    layer_head_mask=None,
                    output_attentions=None,
                )
                hidden_states = layer_outputs[0]

                if idx in self.connector.config.target_layer_ids:
                    # use different prompt for different layers
                    layer_prompt = self.connector.layer_prompts[self.connector.config.target_layer_ids.index(idx)].expand(bs, -1, -1)
                    
                    # Qformer is a BERTEncoder(but set to decoder) from huggingface Transformers
                    qformer_output = self.connector.qformer(
                        layer_prompt,
                        encoder_hidden_states=hidden_states,
                    )
                    
                    layer_prompt_output = qformer_output.last_hidden_state[:, :self.config.prompt_size, :] # (b, prompt_size, d_model)
                    layer_prompt_outputs.append(layer_prompt_output) # list of (b, prompt_size, d_model)

            layer_prompt_outputs = torch.stack(layer_prompt_outputs, dim=0) # (layer, b, prompt_size, d_model)
            layer_prompt_outputs = layer_prompt_outputs.permute(1, 2, 0, 3) # (b, prompt_size, layer, d_model)
            
            self.norm_weights = torch.nn.functional.softmax(self.connector.layer_weights, dim=-1).unsqueeze(-1) # (prompt_size, layer, 1)
            prompt_output = (layer_prompt_outputs * self.norm_weights).sum(dim=2) # (b, prompt_size, d_model)
            assert prompt_output.size(1) == self.config.prompt_size, prompt_output.size()
            prompt_output = self.connector.proj(prompt_output)
            
            return prompt_output

        else:
            raise NotImplementedError(f"mode {self.config.connector_mode} not implemented")
    
    



class DeSTA25Config(PretrainedConfig):
    model_type = "desta25"

    def __init__(self, 
                 llm_model_id="DeSTA-ntu/Llama-3.1-8B-Instruct",
                 encoder_model_id="openai/whisper-large-v3",
                 connector_mode="qformer_1", 
                 qformer_num_hidden_layers=2, 
                 prompt_size=64, 
                 use_lora=False,
                 audio_locator="<|AUDIO|>",
                 placeholder_token="<|reserved_special_token_87|>",
                 whisper_local_weights=None,
                 llm_local_weights=None,
                 qformer_local_weights=None,
                 whisper_force_manual_load=False,
                 whisper_autoconvert_to_safetensors=True,
                 **kwargs):
        
        super().__init__(**kwargs)

        self.llm_model_id = llm_model_id
        self.encoder_model_id = encoder_model_id
        self.connector_mode = connector_mode
        self.qformer_num_hidden_layers = qformer_num_hidden_layers
        self.prompt_size = prompt_size

        self.audio_locator = audio_locator
        self.placeholder_token = placeholder_token

        # Add local weights paths
        self.whisper_local_weights = whisper_local_weights
        self.llm_local_weights = llm_local_weights  
        self.qformer_local_weights = qformer_local_weights
        self.whisper_force_manual_load = whisper_force_manual_load
        self.whisper_autoconvert_to_safetensors = whisper_autoconvert_to_safetensors

        # Load configs (prioritize local paths if available)
        if self.llm_local_weights and os.path.isdir(self.llm_local_weights):
            self.llm_config = AutoConfig.from_pretrained(self.llm_local_weights)
        else:
            self.llm_config = AutoConfig.from_pretrained(self.llm_model_id)
            
        if self.whisper_local_weights and os.path.isdir(self.whisper_local_weights):
            self.encoder_config = AutoConfig.from_pretrained(self.whisper_local_weights)
        else:
            self.encoder_config = AutoConfig.from_pretrained(self.encoder_model_id)

        self.use_lora = use_lora

        self.info = "Ｄｅｓｔａ２。５ Ａｕｄｉｏ"



class DeSTA25AudioModel(PreTrainedModel):
    config_class = DeSTA25Config

    def __init__(self, config, cache_dir=None, token=None, **kwargs):
        super().__init__(config, **kwargs)

        self.config = config

        token = token if token else os.getenv("HF_TOKEN")
        cache_dir = cache_dir if cache_dir else os.getenv("HF_HOME")

        self.audio_locator = config.audio_locator
        self.placeholder_token = config.placeholder_token

        # Load LLM model (prioritize local if available)
        llm_local = getattr(self.config, "llm_local_weights", None)
        if llm_local:
            if os.path.isdir(llm_local):
                print(f"🔧 Loading local LLM from: {llm_local}")
                self.llm_model = AutoModelForCausalLM.from_pretrained(
                    llm_local,
                    dtype=torch.bfloat16,
                    cache_dir=cache_dir,
                    token=token,
                )
            elif llm_local.endswith((".pth", ".pt")):
                print(f"🔧 Loading LLM weights from file: {llm_local}")
                self.llm_model = AutoModelForCausalLM.from_config(self.config.llm_config)
                state = torch.load(llm_local, map_location="cpu")
                if "model" in state:
                    state = state["model"]
                elif "state_dict" in state:
                    state = state["state_dict"]
                missing, unexpected = self.llm_model.load_state_dict(state, strict=False)
                self.llm_model.to(torch.bfloat16)
                if missing or unexpected:
                    print(f"   LLM loading: {len(missing)} missing, {len(unexpected)} unexpected keys")
            else:
                raise ValueError("llm_local_weights must be a directory or a .pth/.pt file")
        else:
            print(f"🔧 Loading LLM from HuggingFace: {self.config.llm_model_id}")
            self.llm_model = AutoModelForCausalLM.from_pretrained(
                self.config.llm_model_id,
                dtype=torch.bfloat16,
                cache_dir=cache_dir,
                token=token,
            )

        if self.config.use_lora:
            from peft import LoraConfig, get_peft_model

            lora_config = LoraConfig(
                r=16,
                lora_alpha=16,
                lora_dropout=0.1,
                target_modules=["q_proj", "k_proj", "v_proj"],
            )
            self.llm_model = get_peft_model(self.llm_model, lora_config).base_model.model
        
        print(f"🔧 Loading Audio encoder from {self.config.encoder_model_id}")
        self.perception = WhisperPerception(self.config)

        self.configure_trainable_parameters()

    def _device(self):
        """Get device of the model parameters"""
        return next(self.parameters()).device

    @property
    def device(self):
        """Get device of the model parameters"""
        return next(self.parameters()).device

    def forward(self, input_ids,
                attention_mask, 
                batch_features, 
                batch_transcription_ids,
                batch_start_positions,
                labels=None,
                **kwargs):
        
        inputs_embeds = self._prepare_inputs_for_llm(
            input_ids=input_ids, 
            attention_mask=attention_mask, 
            batch_features=batch_features,
            batch_transcription_ids=batch_transcription_ids, 
            batch_start_positions=batch_start_positions
        )


        outputs = self.llm_model(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            labels=labels,
        )
        return outputs 

    def _prepare_inputs_for_llm(self, 
                               input_ids,
                               attention_mask,
                               batch_features,
                               batch_transcription_ids,
                               batch_start_positions
        ):
        """
        Prepare the embeddings input for the LLM.
        Batch_features: list of audio features
        Batch_transcription_ids: list of transcription ids
        Batch_start_positions: list of start positions
        """

        N_audio = len(batch_start_positions)
        
        # Get list of transcription embeddings
        transcription_embeddings_list = []
        with torch.no_grad():
            for audio_batch_idx in range(N_audio):
                transcription_embeddings = self.llm_model.model.embed_tokens(
                    batch_transcription_ids[audio_batch_idx].squeeze(0)
                ) # (length, dim)
                transcription_embeddings_list.append(transcription_embeddings)

        # Forward speech encoder and connector
        # Get audio features from Qformer
        batch_audio_features, batch_audio_feature_lengths = self.perception(
            input_features=batch_features, transcription_embeddings_list=transcription_embeddings_list
        )

        assert len(batch_start_positions) == len(batch_transcription_ids) == batch_audio_features.size(0) == len(batch_audio_feature_lengths), "batch_start_positions, batch_transcription_ids, audio_features, speech_feature_lengths must have the same length."


        # [---- Other text embeddings ----][---- placeholder embeddings ----][---- Other text embeddings ----]
        inputs_embeds = self.llm_model.model.embed_tokens(input_ids)
        
        
        for audio_batch_idx in range(N_audio):
            start_position = batch_start_positions[audio_batch_idx] # tuple (text_idx, audio_start_position)
            text_batch_idx = start_position[0]
            audio_start_position = start_position[1]

            # get the speech features   
            audio_features = batch_audio_features[audio_batch_idx]
            speech_feature_length = batch_audio_feature_lengths[audio_batch_idx]

            # get transcription embeddings
            transcription_embeddings = transcription_embeddings_list[audio_batch_idx] # (length, dim)

            # # concat the speech features and transcription embeddings
            audio_embeddings = torch.cat([audio_features, transcription_embeddings], dim=0)

            assert audio_embeddings.size(0) == (speech_feature_length + transcription_embeddings.size(0))

            # # replace the input_embeds with the audio features
            # # [---- Other text embeddings ----][---- audio features + transcription embeddings ----][---- Other text embeddings ----]
            target_slice = slice(audio_start_position, audio_start_position + audio_embeddings.size(0))
            inputs_embeds[text_batch_idx, target_slice] = audio_embeddings
            


            if input_ids[text_batch_idx, audio_start_position-1] == 128096:
                # for debugging
                logging.warning(input_ids[text_batch_idx, audio_start_position-1: audio_start_position + audio_embeddings.size(0)+1])

            # clean GPU memory
            del audio_features, speech_feature_length, transcription_embeddings, audio_embeddings

        return inputs_embeds
        
    def state_dict(self, destination=None, prefix: str = '', keep_vars: bool = False):  # type: ignore[override]
        full = super().state_dict(destination=destination, prefix=prefix, keep_vars=keep_vars)  # type: ignore[arg-type]
        trainable = OrderedDict()
        named = dict(self.named_parameters())
        for k, v in full.items():
            base_name = k[len(prefix):] if prefix and k.startswith(prefix) else k
            param = named.get(base_name, None)
            if param is not None and param.requires_grad:
                trainable[k] = v
        return trainable


    def _generate_step(self, inputs, pad_token_id, temperature=0.7, top_p=0.9, max_new_tokens=512, do_sample=True):
        input_ids = inputs["context_input_ids"] # only context inputs
        attention_mask = inputs["context_attention_mask"] # only context attention mask
        batch_start_positions = inputs["context_batch_start_positions"]

        batch_transcription_ids = inputs["batch_transcription_ids"]
        # batch_audio_features, batch_audio_feature_lengths = self.perception()

        # get the generated text
        inputs_embeds = self._prepare_inputs_for_llm(
            input_ids=input_ids, 
            attention_mask=attention_mask, 
            batch_features=inputs["batch_features"],
            batch_transcription_ids=batch_transcription_ids, 
            batch_start_positions=batch_start_positions
        )

        if do_sample is False:
            top_p = None
            temperature = None
        
        generated_ids = self.llm_model.generate(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            pad_token_id=pad_token_id,
            temperature=temperature,
            top_p=top_p,
            max_new_tokens=max_new_tokens,
            do_sample=do_sample
        )

        return generated_ids


    def configure_trainable_parameters(self):
        """
        for training, log the trainable parameters
        """

        known_parameters = []
        # Freeze LLM parameters
        for name, params in self.llm_model.named_parameters():
            params.requires_grad = False
            known_parameters.append(f"llm_model.{name}")

        # Freeze encoder parameters
        for name, params in self.perception.whisper.named_parameters():
            params.requires_grad = False
            known_parameters.append(f"perception.whisper.{name}")


        # Make other parameters or lora parameters trainable
        self.trainable_parameter_names = []
        trainable_parameters = []
        for name, params in self.named_parameters():
            if name not in known_parameters or "lora" in name:
                params.requires_grad = True
                self.trainable_parameter_names.append(name)
                trainable_parameters.append(params)



    def _setup_generation(self):

        self.tokenizer = AutoTokenizer.from_pretrained(self.config.llm_model_id, cache_dir=os.getenv("HF_HOME"))
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
        self.tokenizer.padding_side = "left"
        
        self.tokenizer.add_tokens([self.audio_locator])
        self.processor = AutoProcessor.from_pretrained(self.config.encoder_model_id, cache_dir=os.getenv("HF_HOME"))

        assert len(self.tokenizer.tokenize(self.audio_locator)) == 1, "audio_locator must be a single token"
        assert len(self.tokenizer.tokenize(self.placeholder_token)) == 1, "placeholder_token must be a single token in the tokenizer"

        # VAD
        self.vad_model, utils = torch.hub.load(repo_or_dir='snakers4/silero-vad', model='silero_vad')
        (self.get_speech_timestamps, _, _, _, _) = utils


    def generate(self, messages,
        # LLM generation args
        temperature=0.7,
        top_p=0.9,
        do_sample=True,
        max_new_tokens=512,
        ):
        """
        messages = [
            {
                "role": "system",
                "content": "Focus on the audio clips and instructions.",
            },
            {
                "role": "user",
                "content": "Hello! this is my audio <|AUDIO|>. Help me transcribe."
                "audios": [
                    "audio": "/path/to/filepath", # path to audio file
                    "text": None # Optional, None or provide text
                ]
            },
        ]
        """
        if not hasattr(self, "tokenizer"):
            self._setup_generation()

        if isinstance(messages, list):
            if isinstance(messages[0], dict):
                messages_list = [messages]
            else: 
                messages_list = messages
        else:
            raise ValueError("messages should be a list of dictionaries or a list of lists.")

        all_audios = []
        all_transcriptions = []
        for messages in messages_list:
            for message in messages:
                content = message["content"]
                audios = message.get("audios", [])
                assert len(audios) == content.count(self.audio_locator), "audio count does not match (<|AUDIO|>) count"

                for audio in audios:
                    all_audios.append(audio["audio"])
                    all_transcriptions.append(audio.get("text"))

        if len(all_audios) > 0:
            """
            If audios are provided, run:
            1. get features and transcription
            2. prepare LLM inputs
            3. run generation
            """

            batch_features = []
            asr_features = []
            asr_indices = []
            for i, (audio, trans) in enumerate(zip(all_audios, all_transcriptions)):
                if not os.path.exists(audio):
                    raise ValueError(f"Audio file {audio} does not exist.")

                # Extract audio features
                feature = AudioSegment.from_file(
                    audio,
                    target_sr=16000,
                    channel_selector="average"
                ).samples

                batch_features.append(feature)

                # Run VAD detect if there is speech in the audio
                is_speech = self.get_speech_timestamps(feature, self.vad_model)
                if is_speech and trans is None:
                    asr_features.append(feature)
                    asr_indices.append(i)
                if not is_speech:
                    all_transcriptions[i] = " "
            
            batch_features = self.processor(batch_features, sampling_rate=16000, return_tensors="pt").input_features
            batch_features = batch_features.to(self.device)
            audio_size_list = [self.config.prompt_size] * len(batch_features)


            # RUN ASR
            if asr_features:
                asr_features = self.processor(asr_features, sampling_rate=16000, return_tensors="pt").input_features
                asr_features = asr_features.to(self.device)

                transcriptions = self.perception.whisper.generate(
                    input_features=asr_features,
                    attention_mask=None,
                    max_new_tokens=128
                )
                transcriptions = self.processor.batch_decode(
                    transcriptions,
                    skip_special_tokens=True,
                )
            else:
                # no audio needs ASR result
                transcriptions = []

            
            for i, transcription in zip(asr_indices, transcriptions):
                all_transcriptions[i] = transcription.strip()
                    
            transcription_size_list = [
                len(self.tokenizer.tokenize(text, add_special_tokens=False)) for text in all_transcriptions
            ]


            audio_context_list = []
            start_positions_list = []
            for messages in messages_list:
                audio_context = self.tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True,
                )

                # <start_audio><|AUDIO|><end_audio> is a indicator used in the training stage
                # We replace <|AUDIO|> with <start_audio><|AUDIO|><end_audio> here
                audio_context = audio_context.replace(self.audio_locator, f"<start_audio>{self.audio_locator}<end_audio>")

                audio_context, start_positions = _prepare_audio_context_and_start_positions(
                        token_list=self.tokenizer.tokenize(audio_context), 
                        audio_locator=self.audio_locator,
                        audio_size_list=audio_size_list,
                        transcription_size_list=transcription_size_list,
                        placeholder_token=self.placeholder_token
                    )


                audio_context = self.tokenizer.convert_tokens_to_string(audio_context)
                audio_context_list.append(audio_context)

                start_positions_list.append(start_positions)


            audio_context_inputs = self.tokenizer(
                audio_context_list,
                truncation=True,
                padding="longest",
                return_tensors="pt",
                return_length=True,
                add_special_tokens=False,
            )

            audio_context_batch_start_positions = []
            for i in range(audio_context_inputs["length"].size(0)):
                total_length = audio_context_inputs["length"][i]
                pad_length = total_length - audio_context_inputs["attention_mask"][i].sum()

                for start_position in start_positions_list[i]:
                    audio_context_batch_start_positions.append((i, start_position + pad_length))

            batch_transcription_ids = []
            for transcription in all_transcriptions:
                batch_transcription_ids.append(
                    self.tokenizer.encode(transcription, add_special_tokens=False, return_tensors="pt").long().to(self.device)
                )

            inputs = {
                "batch_features": batch_features,
                "batch_transcription_ids": batch_transcription_ids,

                "context_input_ids": audio_context_inputs["input_ids"],
                "context_attention_mask": audio_context_inputs['attention_mask'],
                "context_batch_start_positions": audio_context_batch_start_positions,
            }
            inputs = {
                k: v.to(self.device) if isinstance(v, torch.Tensor) else v
                for k, v in inputs.items()
            }

            generated_ids = self._generate_step(
                inputs, 
                pad_token_id=self.tokenizer.pad_token_id,
                temperature=temperature,
                top_p=top_p,
                max_new_tokens=max_new_tokens,
                do_sample=do_sample)

            return GenerationOutput(
                text=self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True),
                audios=[(a, t) for a,t in zip(all_audios, all_transcriptions)],
                generated_ids=generated_ids.tolist()
            )

        else:
            """
            if no audios are provided, it's identical to the original LLM generation
            """

            inputs = self.tokenizer.apply_chat_template(
                messages_list,
                tokenize=False,
                add_generation_prompt=True,
            )
            inputs = self.tokenizer(inputs, return_tensors="pt", padding=True).to(self.device)

            terminators = [
                self.tokenizer.eos_token_id,
                self.tokenizer.convert_tokens_to_ids("<|eot_id|>")
            ]

            generated_ids = self.llm_model.generate(
                inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                eos_token_id=terminators,
                temperature=temperature,
                top_p=top_p,
                max_new_tokens=max_new_tokens,
                do_sample=do_sample
            )

            generated_ids_list = []
            for i in range(len(generated_ids)):
                generated_ids_list.append(generated_ids[i][inputs["input_ids"].shape[1]:].tolist())

            return GenerationOutput(
                text=self.tokenizer.batch_decode(generated_ids_list, skip_special_tokens=True),
                audios=[],
                generated_ids=generated_ids_list
            )

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, *args, **kwargs):
        """
        Custom from_pretrained method to load pretrained LLM and Whisper model.
        model.safetensors only contains trainable parameters from DeSTA2.5-Audio.
        """
        
        cache_dir = kwargs.get("cache_dir", os.getenv("HF_HOME"))

        config = cls.config_class.from_pretrained(pretrained_model_name_or_path, cache_dir=cache_dir)
        model = cls(config)
        
        if os.path.isdir(pretrained_model_name_or_path):
            model.load_state_dict(
                load_file(os.path.join(pretrained_model_name_or_path, "model.safetensors")), strict=False
            )
        else:
            from huggingface_hub import hf_hub_download
            path = hf_hub_download(repo_id=pretrained_model_name_or_path, filename="model.safetensors", cache_dir=cache_dir)
            model.load_state_dict(
                load_file(path), strict=False
            )

        return model