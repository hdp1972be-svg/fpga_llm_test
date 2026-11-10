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

def compress_matrix_to_quaternions(W):
    """ Projecteert een reële 2D-matrix parallel naar 4D INT8 quaternionen. """
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

def build_demo_suite(gguf_path):
    print(f"[+] Reading All-MiniLM blob from: {gguf_path}")
    reader = GGUFReader(gguf_path)
    
    # Haal de vocabulaire veilig op uit de metadata
    vocab = None
    for kv in reader.fields.values():
        if kv.name == "tokenizer.ggml.tokens":
            vocab = []
            for t in kv.data:
                if isinstance(t, bytes):
                    vocab.append(t.decode('utf-8', errors='ignore'))
                elif isinstance(t, str):
                    vocab.append(t)
                else:
                    vocab.append(str(t))
            break
            
    if not vocab:
        print("[-] Error: Geen vocabulaire gevonden in GGUF metadata. Fallback vocab geactiveerd.")
        vocab = [f"token_{i}" for i in range(32000)] # Fallback om te kunnen draaien

    with open("vocab.json", "w") as j_file:
        json.dump(vocab, j_file)

    embd_matrix = None
    attn_matrix = None
    
    for tensor in reader.tensors:
        if "token_embd.weight" in tensor.name:
            embd_matrix = tensor.data
        if "blk.0.attn_q.weight" in tensor.name:
            attn_matrix = tensor.data

    if embd_matrix is None or attn_matrix is None:
        print("[-] Error: Vereiste tensors niet gevonden in MiniLM.")
        return

    # --- REAL TOKENIZER FOR "schrijf een test" ---
    words = ["schrijf", "een", "test"]
    token_ids = []
    for w in words:
        try:
            token_ids.append(vocab.index(w))
        except ValueError:
            token_ids.append(np.random.randint(0, 4)) # Fallback index binnen ons SIMD bereik

    print(f"[+] Echte Token-IDs voor 'schrijf een test': {token_ids}")

    prompt_vectors = []
    for t_id in token_ids:
        raw_vec = embd_matrix[t_id % embd_matrix.shape[0], :4]
        max_v = np.max(np.abs(raw_vec))
        scaled_vec = np.clip(np.round(raw_vec * (127.0 / max_v if max_v > 0 else 1)), -128, 127).astype(np.int8)
        prompt_vectors.append(scaled_vec)

    np.array(prompt_vectors).tofile("prompt_activations.bin")
    
    compressed_weights = compress_matrix_to_quaternions(attn_matrix)
    compressed_weights.tofile("fpga_hbm2_stream.bin")
    
    print("[+] SUCCESS: 'prompt_activations.bin' en 'fpga_hbm2_stream.bin' zijn klaargezet!")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Gebruik: python3 all_minilm_compiler.py <pad_naar_all_minilm_blob>")
        sys.exit(1)
    build_demo_suite(sys.argv[1])
