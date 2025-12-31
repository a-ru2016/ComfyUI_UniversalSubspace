import torch
import torch.nn.functional as F
import folder_paths
import comfy.utils
import comfy.sd
from safetensors.torch import load_file
import re
import os

# --- Helper Functions ---

def convert_alpha_key_to_basis_base(key):
    """Alpha key (te1.text.model...) -> Basis base name"""
    return key.replace("_", ".")

def convert_basis_base_to_comfy_key(basis_base_key):
    """Basis base name (te1.text.model...) -> ComfyUI LoRA key (lora_te1_text_model...)"""
    if basis_base_key.startswith("unet."):
        new_key = "lora_unet_" + basis_base_key[5:]
    elif basis_base_key.startswith("te1."):
        new_key = "lora_te1_" + basis_base_key[4:]
    elif basis_base_key.startswith("te2."):
        new_key = "lora_te2_" + basis_base_key[4:]
    elif basis_base_key.startswith("te."):
        new_key = "lora_te_" + basis_base_key[3:]
    else:
        new_key = "lora_unet_" + basis_base_key # Fallback
    return new_key.replace(".", "_")

def map_model_keys_to_lora_keys(model, clip):
    """Scan all layers in the model and create a map to LoRA keys"""
    shape_map = {}
    
    # --- 1. UNet Mapping ---
    unet_root = model.model.diffusion_model
    for name, module in unet_root.named_modules():
        if module.__class__.__name__ in ["Linear", "Conv2d", "LoRACompatibleLinear", "LoRACompatibleConv"]:
            lora_key_base = "lora_unet_" + name.replace(".", "_")
            if hasattr(module, "weight"):
                shape_map[lora_key_base] = module.weight.shape

    # --- 2. CLIP Mapping ---
    def scan_te(te_model, prefix):
        for name, module in te_model.named_modules():
            if module.__class__.__name__ in ["Linear", "Conv2d", "LoRACompatibleLinear", "LoRACompatibleConv"]:
                if hasattr(module, "weight"):
                    clean_name = name
                    if clean_name.startswith("transformer."):
                        clean_name = clean_name.replace("transformer.", "")
                    lora_key_base = prefix + clean_name.replace(".", "_")
                    shape_map[lora_key_base] = module.weight.shape
                    
                    raw_key_base = prefix + name.replace(".", "_")
                    if raw_key_base not in shape_map:
                        shape_map[raw_key_base] = module.weight.shape

    if hasattr(clip.cond_stage_model, "clip_l"): 
        scan_te(clip.cond_stage_model.clip_l, "lora_te1_")
    if hasattr(clip.cond_stage_model, "clip_g"): 
        scan_te(clip.cond_stage_model.clip_g, "lora_te2_")
    if hasattr(clip.cond_stage_model, "transformer"):
        scan_te(clip.cond_stage_model.transformer, "lora_te_")
        
    return shape_map

def resize_lora_weight(weight, expected_param_count, is_up=False):
    """
    Adjusts the LoRA weight rank to match the Subspace Basis using Slicing.
    """
    current_count = weight.numel()
    if current_count == expected_param_count:
        return weight

    shape = weight.shape
    
    # Identify Rank Dimension and Target Rank
    if len(shape) == 2: # Linear
        if not is_up: # Down: (Rank, In)
            rank_dim = 0
            other_dim = shape[1]
            current_rank = shape[0]
            required_rank = expected_param_count // other_dim
        else: # Up: (Out, Rank)
            rank_dim = 1
            other_dim = shape[0]
            current_rank = shape[1]
            required_rank = expected_param_count // other_dim
    elif len(shape) == 4: # Conv
        if not is_up: # Down: (Rank, In, K, K)
            rank_dim = 0
            other_dims = shape[1] * shape[2] * shape[3]
            current_rank = shape[0]
            required_rank = expected_param_count // other_dims
        else: # Up: (Out, Rank, 1, 1)
            rank_dim = 1
            other_dims = shape[0] * shape[2] * shape[3]
            current_rank = shape[1]
            required_rank = expected_param_count // other_dims
    else:
        return weight # Unknown shape

    if required_rank == current_rank:
        return weight

    # Case 1: Rank Insufficient (Padding)
    if required_rank > current_rank:
        pad_size = required_rank - current_rank
        pad_shape = list(shape)
        pad_shape[rank_dim] = pad_size
        padding = torch.zeros(pad_shape, dtype=weight.dtype, device=weight.device)
        
        if rank_dim == 0:
            new_weight = torch.cat([weight, padding], dim=0)
        else:
            new_weight = torch.cat([weight, padding], dim=1)
            
    # Case 2: Rank Excess (Slicing)
    else:
        if rank_dim == 0:
            new_weight = weight[:required_rank, ...]
        else:
            # Assuming rank is dim 1
            if len(shape) == 2:
                new_weight = weight[:, :required_rank]
            elif len(shape) == 4:
                new_weight = weight[:, :required_rank, :, :]
            else:
                new_weight = weight

    return new_weight

# --- Class Definitions ---

class UniversalLoRALoader:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {"universal_lora_name": (folder_paths.get_filename_list("loras"), )}}
    
    RETURN_TYPES = ("UNIVERSAL_SUBSPACE",)
    FUNCTION = "load_subspace"
    CATEGORY = "UniversalSubspace"

    def load_subspace(self, universal_lora_name):
        lora_path = folder_paths.get_full_path("loras", universal_lora_name)
        state_dict = load_file(lora_path)
        print(f"DEBUG: Loaded Basis/LoRA. Count: {len(state_dict)}")
        return (state_dict,)

class LoadUniversalWeights:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {"weight_file": (folder_paths.get_filename_list("loras"), )}}
    
    RETURN_TYPES = ("UNIVERSAL_WEIGHTS",)
    FUNCTION = "load_weights"
    CATEGORY = "UniversalSubspace"

    def load_weights(self, weight_file):
        path = folder_paths.get_full_path("loras", weight_file)
        weights = load_file(path)
        print(f"DEBUG: Loaded Weights. Count: {len(weights)}")
        return (weights,)

class ApplyUniversalWeights:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model": ("MODEL",),
                "clip": ("CLIP",),
                "subspace": ("UNIVERSAL_SUBSPACE",),
                "universal_weights": ("UNIVERSAL_WEIGHTS",),
                "strength_model": ("FLOAT", {"default": 1.0, "min": -10.0, "max": 10.0, "step": 0.01}),
                "strength_clip": ("FLOAT", {"default": 1.0, "min": -10.0, "max": 10.0, "step": 0.01}),
            }
        }
    
    RETURN_TYPES = ("MODEL", "CLIP")
    FUNCTION = "apply_weights"
    CATEGORY = "UniversalSubspace"

    def apply_weights(self, model, clip, subspace, universal_weights, strength_model, strength_clip):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print("DEBUG: Mapping model keys for Universal Weights...")
        shape_map = map_model_keys_to_lora_keys(model, clip)
        
        reconstructed_lora = {}
        count = 0
        
        for key, alpha in universal_weights.items():
            if not key.endswith(".alpha"): continue
            
            alpha_base = key.replace(".alpha", "")
            basis_base = convert_alpha_key_to_basis_base(alpha_base)
            down_prefix = basis_base + ".lora_down.weight"
            up_prefix = basis_base + ".lora_up.weight"
            
            if (down_prefix + ".mean") not in subspace or (up_prefix + ".mean") not in subspace: 
                continue

            try:
                a = alpha.to(device, dtype=torch.float32)
                
                # --- Down Reconstruction ---
                mu_d = subspace[down_prefix + ".mean"].to(device, dtype=torch.float32)
                V_d = subspace[down_prefix + ".basis"].to(device, dtype=torch.float32)
                w_down_flat = mu_d + torch.matmul(V_d, a)

                # --- Up Reconstruction ---
                mu_u = subspace[up_prefix + ".mean"].to(device, dtype=torch.float32)
                V_u = subspace[up_prefix + ".basis"].to(device, dtype=torch.float32)
                w_up_flat = mu_u + torch.matmul(V_u, a)

                # --- Shape Matching ---
                comfy_key_base = convert_basis_base_to_comfy_key(basis_base)
                target_shape = shape_map.get(comfy_key_base, None)
                final_key_name = comfy_key_base

                if target_shape is None:
                    if "_text_model_" in comfy_key_base:
                        alt_key = comfy_key_base.replace("_text_model_", "_")
                        target_shape = shape_map.get(alt_key, None)
                        if target_shape is not None: final_key_name = alt_key

                if target_shape is None: continue

                if len(target_shape) == 2:
                    out_dim, in_dim = target_shape
                    if w_down_flat.numel() % in_dim != 0: continue
                    rank = w_down_flat.numel() // in_dim
                    w_down = w_down_flat.reshape(rank, in_dim)
                    w_up = w_up_flat.reshape(out_dim, rank)
                elif len(target_shape) == 4:
                    out_dim, in_dim, kh, kw = target_shape
                    ks = kh * kw
                    if w_down_flat.numel() % (in_dim * ks) != 0: continue
                    rank = w_down_flat.numel() // (in_dim * ks)
                    w_down = w_down_flat.reshape(rank, in_dim, kh, kw)
                    w_up = w_up_flat.reshape(out_dim, rank, 1, 1)
                else: continue

                reconstructed_lora[final_key_name + ".lora_down.weight"] = w_down.cpu()
                reconstructed_lora[final_key_name + ".lora_up.weight"] = w_up.cpu()
                count += 1

            except Exception as e:
                print(f"Error reconstructing {basis_base}: {e}")
        
        print(f"DEBUG: Universal LoRA Applied Layers: {count}")
        if count == 0: return (model, clip)

        model_lora, clip_lora = comfy.sd.load_lora_for_models(model, clip, reconstructed_lora, strength_model, strength_clip)
        return (model_lora, clip_lora)

class ApplyUniversalLoRA:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model": ("MODEL",),
                "clip": ("CLIP",),
                "subspace": ("UNIVERSAL_SUBSPACE",),
                "target_lora_name": (folder_paths.get_filename_list("loras"), ),
                "strength_model": ("FLOAT", {"default": 1.0, "min": -10.0, "max": 10.0, "step": 0.01}),
                "strength_clip": ("FLOAT", {"default": 1.0, "min": -10.0, "max": 10.0, "step": 0.01}),
            }
        }
    RETURN_TYPES = ("MODEL", "CLIP")
    FUNCTION = "apply_universal"
    CATEGORY = "UniversalSubspace"

    def apply_universal(self, model, clip, subspace, target_lora_name, strength_model, strength_clip):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # 1. Load Target LoRA
        target_path = folder_paths.get_full_path("loras", target_lora_name)
        target_lora = load_file(target_path)
        print(f"DEBUG: Loaded Target LoRA: {target_lora_name} ({len(target_lora)} keys)")

        final_lora = {}

        # 2. Identify Basis Keys
        basis_keys = set()
        for key in subspace.keys():
            if key.endswith(".lora_down.weight.basis"):
                base = key.replace(".lora_down.weight.basis", "")
                basis_keys.add(base)
        
        count_approx = 0
        
        # 3. Process each basis block
        for basis_base in basis_keys:
            comfy_key_base = convert_basis_base_to_comfy_key(basis_base)
            down_key = comfy_key_base + ".lora_down.weight"
            up_key = comfy_key_base + ".lora_up.weight"
            
            # Fallback search for key names
            if down_key not in target_lora:
                if "_text_model_" in comfy_key_base:
                    alt = comfy_key_base.replace("_text_model_", "_")
                    if (alt + ".lora_down.weight") in target_lora:
                        down_key = alt + ".lora_down.weight"
                        up_key = alt + ".lora_up.weight"

            if down_key in target_lora and up_key in target_lora:
                try:
                    down_prefix = basis_base + ".lora_down.weight"
                    up_prefix = basis_base + ".lora_up.weight"
                    
                    if (down_prefix + ".mean") in subspace and (up_prefix + ".mean") in subspace:
                        # --- Prepare Data (Joint Projection) ---
                        mu_d = subspace[down_prefix + ".mean"].to(device, dtype=torch.float32)
                        V_d = subspace[down_prefix + ".basis"].to(device, dtype=torch.float32)
                        mu_u = subspace[up_prefix + ".mean"].to(device, dtype=torch.float32)
                        V_u = subspace[up_prefix + ".basis"].to(device, dtype=torch.float32)
                        
                        w_down_raw = target_lora[down_key].to(device, dtype=torch.float32)
                        w_up_raw = target_lora[up_key].to(device, dtype=torch.float32)
                        
                        w_down = resize_lora_weight(w_down_raw, mu_d.shape[0], is_up=False)
                        w_up = resize_lora_weight(w_up_raw, mu_u.shape[0], is_up=True)
                        
                        # Joint System for Least Squares
                        target_vec_d = w_down.flatten() - mu_d
                        target_vec_u = w_up.flatten() - mu_u
                        
                        joint_target = torch.cat([target_vec_d, target_vec_u], dim=0)
                        joint_basis = torch.cat([V_d, V_u], dim=0)
                        
                        result = torch.linalg.lstsq(joint_basis, joint_target)
                        alpha = result.solution
                        
                        w_down_recon_flat = mu_d + torch.matmul(V_d, alpha)
                        w_up_recon_flat = mu_u + torch.matmul(V_u, alpha)
                        
                        w_down_recon = w_down_recon_flat.reshape(w_down.shape)
                        w_up_recon = w_up_recon_flat.reshape(w_up.shape)
                        
                        final_lora[down_key] = w_down_recon.cpu()
                        final_lora[up_key] = w_up_recon.cpu()
                        
                        count_approx += 1
                except Exception as e:
                    print(f"Warning: Failed to process Joint {basis_base}: {e}")
        
        print(f"DEBUG: Joint Subspace Application - Total Reconstructed Pairs: {count_approx}")

        model_lora, clip_lora = comfy.sd.load_lora_for_models(model, clip, final_lora, strength_model, strength_clip)
        return (model_lora, clip_lora)

class ApplyNullLaLoRA:
    """
    Node to apply Null-LoRA / LaLoRA weights trained with network_null_lalora_bata.py.
    Fully supports:
    - Null Space (Frozen) + Universal Basis (Trainable)
    - Standard Scaling (s)
    - DoRA Magnitude Scaling (m)
    """
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model": ("MODEL",),
                "clip": ("CLIP",),
                "subspace": ("UNIVERSAL_SUBSPACE",),
                "null_lalora_weights": ("UNIVERSAL_WEIGHTS",),
                "strength_model": ("FLOAT", {"default": 1.0, "min": -10.0, "max": 10.0, "step": 0.01}),
                "strength_clip": ("FLOAT", {"default": 1.0, "min": -10.0, "max": 10.0, "step": 0.01}),
            }
        }
    
    RETURN_TYPES = ("MODEL", "CLIP")
    FUNCTION = "apply_null_lalora"
    CATEGORY = "UniversalSubspace"

    def apply_null_lalora(self, model, clip, subspace, null_lalora_weights, strength_model, strength_clip):
        device = torch.device("cpu") # Process weights on CPU to save VRAM
        print("DEBUG: Mapping model keys for Null-LaLoRA (DoRA/Standard)...")
        shape_map = map_model_keys_to_lora_keys(model, clip)
        
        # --- Pre-calculate Subspace Key Map ---
        subspace_lookup = {}
        for k in subspace.keys():
            if k.endswith(".lora_down.weight.mean"):
                basis_base = k.replace(".lora_down.weight.mean", "")
                safe_key = basis_base.replace(".", "_")
                subspace_lookup[safe_key] = basis_base
                
        reconstructed_lora = {}
        count = 0
        
        # Identify layer base names from the loaded weights
        layer_bases = set()
        for key in null_lalora_weights.keys():
            if key.endswith(".alpha_down"):
                layer_bases.add(key.replace(".alpha_down", ""))
            elif key.endswith(".alpha") and not key.endswith(".alpha_up"):
                layer_bases.add(key.replace(".alpha", ""))
        
        for base_name in layer_bases:
            # 1. Resolve Keys
            matched_subspace_base = subspace_lookup.get(base_name, None)
            if matched_subspace_base is None:
                continue

            comfy_key_base = convert_basis_base_to_comfy_key(matched_subspace_base)
            
            # Check target shape
            target_shape = shape_map.get(comfy_key_base, None)
            if target_shape is None:
                 if "_text_model_" in comfy_key_base:
                     alt_key = comfy_key_base.replace("_text_model_", "_")
                     if alt_key in shape_map:
                         comfy_key_base = alt_key
                         target_shape = shape_map[alt_key]

            if target_shape is None:
                continue

            try:
                # --- Retrieve Basis ---
                mu_d = subspace[matched_subspace_base + ".lora_down.weight.mean"].to(device, dtype=torch.float32)
                V_d = subspace[matched_subspace_base + ".lora_down.weight.basis"].to(device, dtype=torch.float32)
                mu_u = subspace[matched_subspace_base + ".lora_up.weight.mean"].to(device, dtype=torch.float32)
                V_u = subspace[matched_subspace_base + ".lora_up.weight.basis"].to(device, dtype=torch.float32)
                
                # --- Retrieve Alpha & Reconstruct Trainable ---
                w_down_train_flat = None
                w_up_train_flat = None
                half_rank = 0

                if (base_name + ".alpha_down") in null_lalora_weights:
                    alpha_d = null_lalora_weights[base_name + ".alpha_down"].to(device, dtype=torch.float32)
                    alpha_u = null_lalora_weights[base_name + ".alpha_up"].to(device, dtype=torch.float32)
                    
                    half_rank = alpha_d.shape[0]
                    # Resize basis if rank mismatches
                    if V_d.shape[1] > half_rank: V_d = V_d[:, :half_rank]
                    if V_u.shape[1] > half_rank: V_u = V_u[:, :half_rank]

                    # Transpose logic matches network_null_lalora_bata.py
                    # w_down = (alpha_d @ basis_d.T) + mean
                    w_down_train_flat = (alpha_d @ V_d.t()) + mu_d.view(1, -1)
                    # w_up = (basis_u @ alpha_u) + mean
                    w_up_train_flat = (V_u @ alpha_u) + mu_u.view(-1, 1)

                elif (base_name + ".alpha") in null_lalora_weights:
                     # Legacy scalar alpha support
                     alpha = null_lalora_weights[base_name + ".alpha"].to(device, dtype=torch.float32)
                     half_rank = alpha.shape[0]
                     if V_d.shape[1] > half_rank: V_d = V_d[:, :half_rank]
                     if V_u.shape[1] > half_rank: V_u = V_u[:, :half_rank]
                     
                     w_down_train_flat = mu_d + torch.matmul(V_d, alpha)
                     w_up_train_flat = mu_u + torch.matmul(V_u, alpha)
                else:
                    continue 

                # --- Retrieve Frozen Null Parts ---
                # Priority: 1. Weights File (if saved), 2. Subspace File
                frozen_down = None
                frozen_up = None

                # Note on naming:
                # In training script: null_down=(Out, Rank), null_up=(Rank, In)
                # In forward pass: frozen_down=null_up(Rank, In), frozen_up=null_down(Out, Rank)
                if (base_name + ".null_up") in null_lalora_weights:
                    frozen_down = null_lalora_weights[base_name + ".null_up"].to(device, dtype=torch.float32)
                    frozen_up = null_lalora_weights[base_name + ".null_down"].to(device, dtype=torch.float32)
                elif (matched_subspace_base + ".lora_up.weight.null") in subspace:
                     frozen_down = subspace[matched_subspace_base + ".lora_up.weight.null"].to(device, dtype=torch.float32)
                     frozen_up = subspace[matched_subspace_base + ".lora_down.weight.null"].to(device, dtype=torch.float32)
                
                if frozen_down is None or frozen_up is None:
                    continue

                # Ensure Frozen rank matches half_rank (Slice if necessary)
                if frozen_down.shape[0] > half_rank: 
                    frozen_down = frozen_down[:half_rank, ...]
                
                if frozen_up.shape[1] > half_rank and len(frozen_up.shape) == 2:
                    frozen_up = frozen_up[:, :half_rank]
                elif len(frozen_up.shape) == 4 and frozen_up.shape[1] > half_rank:
                     frozen_up = frozen_up[:, :half_rank, :, :]

                # --- Determine Mode: Standard (s) or DoRA (m) ---
                use_dora = False
                if (base_name + ".m") in null_lalora_weights:
                    use_dora = True
                    scale_vec = null_lalora_weights[base_name + ".m"].to(device, dtype=torch.float32)
                elif (base_name + ".s") in null_lalora_weights:
                    use_dora = False
                    scale_vec = null_lalora_weights[base_name + ".s"].to(device, dtype=torch.float32)
                    s1 = scale_vec[:half_rank]
                    s2 = scale_vec[half_rank:]
                else:
                    # Fallback default
                    scale_vec = torch.ones(target_shape[0], device=device, dtype=torch.float32)
                    use_dora = True # Treat as identity DoRA
                    
                # --- Reshape & Combine ---
                if len(target_shape) == 2:
                    # Linear: (Out, In)
                    out_dim, in_dim = target_shape
                    w_down_train = w_down_train_flat.reshape(half_rank, in_dim)
                    w_up_train = w_up_train_flat.reshape(out_dim, half_rank)
                    
                    f_down = frozen_down.reshape(half_rank, in_dim)
                    f_up = frozen_up.reshape(out_dim, half_rank)

                    if use_dora:
                        # DoRA Logic: Concatenate then Apply Magnitude
                        final_down = torch.cat([f_down, w_down_train], dim=0) # (Rank*2, In)
                        final_up = torch.cat([w_up_train, f_up], dim=1)       # (Out, Rank*2)
                        final_up = final_up * scale_vec.view(-1, 1)
                    else:
                        # Standard Logic: Apply scales at bottleneck
                        w_down_scaled = w_down_train * s2.view(-1, 1)
                        final_down = torch.cat([f_down, w_down_scaled], dim=0)
                        
                        w_up_scaled = w_up_train * s1.view(1, -1)
                        final_up = torch.cat([w_up_scaled, f_up], dim=1)

                elif len(target_shape) == 4:
                    # Conv2d: (Out, In, kh, kw)
                    out_dim, in_dim, kh, kw = target_shape
                    w_down_train = w_down_train_flat.reshape(half_rank, in_dim, kh, kw)
                    w_up_train = w_up_train_flat.reshape(out_dim, half_rank, 1, 1)
                    
                    f_down = frozen_down.reshape(half_rank, in_dim, kh, kw)
                    f_up = frozen_up.reshape(out_dim, half_rank, 1, 1)
                    
                    if use_dora:
                        final_down = torch.cat([f_down, w_down_train], dim=0)
                        final_up = torch.cat([w_up_train, f_up], dim=1)
                        final_up = final_up * scale_vec.view(-1, 1, 1, 1)
                    else:
                        w_down_scaled = w_down_train * s2.view(-1, 1, 1, 1)
                        final_down = torch.cat([f_down, w_down_scaled], dim=0)
                        
                        w_up_scaled = w_up_train * s1.view(1, -1, 1, 1)
                        final_up = torch.cat([w_up_scaled, f_up], dim=1)
                    
                else:
                    continue

                reconstructed_lora[comfy_key_base + ".lora_down.weight"] = final_down.cpu()
                reconstructed_lora[comfy_key_base + ".lora_up.weight"] = final_up.cpu()
                count += 1

            except Exception as e:
                print(f"Error reconstructing Null-LaLoRA {base_name}: {e}")

        print(f"DEBUG: Null-LaLoRA Applied Layers: {count} (DoRA: {use_dora})")
        if count == 0: return (model, clip)

        model_lora, clip_lora = comfy.sd.load_lora_for_models(model, clip, reconstructed_lora, strength_model, strength_clip)
        return (model_lora, clip_lora)

NODE_CLASS_MAPPINGS = {
    "UniversalLoRALoader": UniversalLoRALoader,
    "LoadUniversalWeights": LoadUniversalWeights,
    "ApplyUniversalWeights": ApplyUniversalWeights,
    "ApplyUniversalLoRA": ApplyUniversalLoRA,
    "ApplyNullLaLoRA": ApplyNullLaLoRA
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "UniversalLoRALoader": "Load Universal Subspace (Basis)",
    "LoadUniversalWeights": "Load Universal Weights (Alpha)",
    "ApplyUniversalWeights": "Apply Universal Weights",
    "ApplyUniversalLoRA": "Approximating LoRA (Project Target)",
    "ApplyNullLaLoRA": "Apply Null-LaLoRA (Universal + Null)"
}