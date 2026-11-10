#!/usr/local/bin/python3.11
# ====================================================================
# Copyright (C) 2026 
# Licensed under GNU GPLv3 / Text concepts under CC BY-SA 4.0
# Original source post on Threads. References required.
#// ====================================================================

import numpy as np
import sys
import os
import json
from gguf import GGUFReader

def find_ollama_blob_by_name(model_name, base_dir="/mnt/btrfs/@ollama"):
    """ Zoekt de exacte SHA256-hash van de GGUF-modelblob in het manifest. """
    if ":" in model_name:
        name_part, tag = model_name.split(":", 1)
    else:
        name_part, tag = model_name, "latest"
        
    possible_paths = [
        os.path.join(base_dir, "manifests", "registry.ollama.ai", "library", name_part, tag),
        os.path.join(base_dir, "manifests", "hf.co", name_part, tag),
        os.path.join(base_dir, "manifests", name_part, tag)
    ]
    
    manifest_found = None
    for path in possible_paths:
        if os.path.exists(path):
            manifest_found = path
            break
            
    if not manifest_found:
        print(f"[-] Manifest voor model '{model_name}' niet gevonden.")
        return None

    with open(manifest_found, "r") as f:
        try:
            manifest_data = json.load(f)
        except Exception as e:
            print(f"[-] Error bij parsen manifest: {e}")
            return None

    model_digest = None
    for layer in manifest_data.get("layers", []):
        media_type = layer.get("mediaType", "")
        if "gguf" in media_type or "model" in media_type:
            model_digest = layer.get("digest", "")
            break
            
    if not model_digest and manifest_data.get("layers"):
        largest_layer = max(manifest_data["layers"], key=lambda x: x.get("size", 0))
        model_digest = largest_layer.get("digest", "")

    if not model_digest:
        return None

    blob_filename = model_digest.replace("sha256:", "sha256-")
    return os.path.join(base_dir, "blobs", blob_filename)

def process_tensor_to_quaternions(weights_raw):
    """ Verwerkt een 2D tensor matrix direct via Numpy vectorisatie """
    # Altijd casten naar float32 om scalar overflows te vermijden!
    W = weights_raw.astype(np.float32)
    
    rows, cols = W.shape
    pad_r = (4 - (rows % 4)) % 4
    pad_c = (4 - (cols % 4)) % 4
    if pad_r > 0 or pad_c > 0:
        W = np.pad(W, ((0, pad_r), (0, pad_c)), mode='constant')
        
    h_rows, h_cols = W.shape
    
    # Hervorm de matrix naar 4x4 sub-grids voor vector-processing
    grid = W.reshape(h_rows // 4, 4, h_cols // 4, 4).swapaxes(1, 2)
    
    # Pas de Clifford-Hamilton transformatie parallel toe op alle 4x4 blokken
    a = (grid[:, :, 0, 0] + grid[:, :, 1, 1] + grid[:, :, 2, 2] + grid[:, :, 3, 3]) / 4.0
    b = (-grid[:, :, 0, 1] + grid[:, :, 1, 0] - grid[:, :, 2, 3] + grid[:, :, 3, 2]) / 4.0
    c = (-grid[:, :, 0, 2] + grid[:, :, 1, 3] + grid[:, :, 2, 0] - grid[:, :, 3, 1]) / 4.0
    d = (-grid[:, :, 0, 3] - grid[:, :, 1, 2] + grid[:, :, 2, 1] + grid[:, :, 3, 0]) / 4.0
    
    # Stapel de quaternion-componenten achter elkaar [N, M, 4]
    quaternions = np.stack([a, b, c, d], axis=-1)
    return quaternions.flatten()

def compile_ollama_to_fpga(model_name):
    gguf_path = find_ollama_blob_by_name(model_name)
    if not gguf_path or not os.path.exists(gguf_path):
        print("[-] Kon de actieve model blob niet localiseren.")
        return
        
    print(f"[+] Blob gevonden: {gguf_path}")
    reader = GGUFReader(gguf_path)
    output_bin_path = "fpga_hbm2_stream.bin"
    total_bytes = 0

    with open(output_bin_path, "wb") as out_file:
        for tensor in reader.tensors:
            name = tensor.name
            weights = tensor.data
            
            # Verwerk alleen de structurele 2D gewichten-matrices
            if len(weights.shape) == 2:
                print(f" -> Mapping layer: {name:<50} ... ", end="", flush=True)
                
                # 1. Hypercomplexe transformatie
                q_stream = process_tensor_to_quaternions(weights)
                
                # 2. INT8 Kwantisatie
                max_val = np.max(np.abs(q_stream))
                if max_val > 0:
                    scale = 127.0 / max_val
                    int8_stream = np.clip(np.round(q_stream * scale), -128, 127).astype(np.int8)
                else:
                    int8_stream = np.zeros(q_stream.size, dtype=np.int8)
                    
                # 3. Write naar binaire FPGA stream
                out_file.write(int8_stream.tobytes())
                total_bytes += int8_stream.nbytes
                print(f"[SUCCESS] {int8_stream.nbytes:,} bytes.")

    print(f"\n[!] COMPILATIE COMPLEET: '{output_bin_path}' succesvol gegenereerd ({total_bytes / (1024*1024):.2f} MB).")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Gebruik: ./ollama_blob_compiler.py <modelnaam:tag>")
        sys.exit(1)
    compile_ollama_to_fpga(sys.argv[1])
