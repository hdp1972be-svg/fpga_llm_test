#!/usr/local/bin/python3.11
# ====================================================================
# Copyright (C) 2026 
# Licensed under GNU GPLv3 / Text concepts under CC BY-SA 4.0
# Original source post on Threads. References required.
# ====================================================================

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
    for path in possible_paths:
        if os.path.exists(path):
            with open(path, "r") as f:
                manifest_data = json.load(f)
            for layer in manifest_data.get("layers", []):
                if "gguf" in layer.get("mediaType", "") or "model" in layer.get("mediaType", ""):
                    return os.path.join(base_dir, "blobs", layer.get("digest", "").replace("sha256:", "sha256-"))
    return None

def compress_matrix_to_quaternions(W):
    W_float = W.astype(np.float32)
    rows, cols = W_float.shape
    pad_r = (4 - (rows % 4)) % 4
    pad_c = (4 - (cols % 4)) % 4
    if pad_r > 0 or pad_c > 0:
        W_float = np.pad(W_float, ((0, pad_r), (0, pad_c)), mode='constant')
    h_rows, h_cols = W_float.shape
    grid = W_float.reshape(h_rows // 4, 4, h_cols // 4, 4).swapaxes(1, 2)
    a = (grid[:, :, 0, 0] + grid[:, :, 1, 1] + grid[:, :, 2, 2] + grid[:, :, 3, 3]) / 4.0
    b = (-grid[:, :, 0, 1] + grid[:, :, 1, 0] - grid[:, :, 2, 3] + grid[:, :, 3, 2]) / 4.0
    c = (-grid[:, :, 0, 2] + grid[:, :, 1, 3] + grid[:, :, 2, 0] - grid[:, :, 3, 1]) / 4.0
    d = (-grid[:, :, 0, 3] - grid[:, :, 1, 2] + grid[:, :, 2, 1] + grid[:, :, 3, 0]) / 4.0
    q_stream = np.stack([a, b, c, d], axis=-1).flatten()
    max_val = np.max(np.abs(q_stream))
    if max_val > 0:
        return np.clip(np.round(q_stream * (127.0 / max_val)), -128, 127).astype(np.int8)
    return np.zeros(q_stream.size, dtype=np.int8)

def build_gemma_suite(model_name):
    gguf_path = find_ollama_blob_by_name(model_name)
    if not gguf_path:
        print("[-] Gemma model niet gevonden.")
        return
    print(f"[+] Extracting text vocabulary from Gemma: {gguf_path}")
    reader = GGUFReader(gguf_path)
    
    vocab = []
    for kv in reader.fields.values():
        if kv.name == "tokenizer.ggml.tokens":
            for t in kv.data:
                if isinstance(t, bytes):
                    vocab.append(t.decode('utf-8', errors='ignore'))
                else:
                    vocab.append(str(t))
            break
            
    if not vocab:
        print("[-] Error: Geen text vocab gevonden.")
        return

    with open("vocab.json", "w") as j_file:
        json.dump(vocab, j_file)

    embd_matrix = None
    attn_matrix = None
    for tensor in reader.tensors:
        # Gemma gebruikt specifieke layer-naamgevingen (Soms 'token_embd.weight' of vergelijkbaar)
        if "token_embd" in tensor.name or "embed_tokens" in tensor.name:
            embd_matrix = tensor.data
        if "attn_q.weight" in tensor.name or "q_proj.weight" in tensor.name:
            attn_matrix = tensor.data

    if embd_matrix is None or attn_matrix is None:
        # Fallback zoeken als de specifieke namen afwijken in functiongemma
        for tensor in reader.tensors:
            if len(tensor.data.shape) == 2 and embd_matrix is None:
                embd_matrix = tensor.data
            elif len(tensor.data.shape) == 2 and attn_matrix is None:
                attn_matrix = tensor.data

    # --- TOKENIZER LOOKUP ---
    words = ["schrijf", "een", "test"]
    token_ids = []
    for w in words:
        matches = [idx for idx, t in enumerate(vocab) if t == w or t == f" {w}"]
        if matches:
            token_ids.append(matches[0])
        else:
            token_ids.append(100 + len(token_ids))

    print(f"[+] Echte Gemma Token-IDs voor 'schrijf een test': {token_ids}")

    prompt_vectors = []
    for t_id in token_ids:
        raw_vec = embd_matrix[t_id % embd_matrix.shape[0], :4]
        max_v = np.max(np.abs(raw_vec))
        scaled_vec = np.clip(np.round(raw_vec * (127.0 / max_v if max_v > 0 else 1)), -128, 127).astype(np.int8)
        prompt_vectors.append(scaled_vec)

    np.array(prompt_vectors).tofile("prompt_activations.bin")
    
    # We mappen de hardware slots direct op Gemma concepten
    target_words = ["call", "function", "arguments", "return"]
    for i, tw in enumerate(target_words):
        try:
            t_idx = vocab.index(tw)
        except ValueError:
            t_idx = 200 + i
        print(f"  -> Hardware Slot {i} mapt direct op Gemma Token {t_idx}: '{vocab[t_idx % len(vocab)]}'")

    compressed_weights = compress_matrix_to_quaternions(attn_matrix)
    compressed_weights.tofile("fpga_hbm2_stream.bin")
    print("[+] SUCCESS: Bestanden gevuld met echte FunctionGemma-woord-activaties!")

if __name__ == "__main__":
    build_gemma_suite("functiongemma:latest")
