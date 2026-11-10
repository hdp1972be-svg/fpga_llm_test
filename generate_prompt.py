#!/usr/bin/env python3
# ====================================================================
# Copyright (C) 2026 
# Licensed under GNU GPLv3 / Text concepts under CC BY-SA 4.0
# Original source post on Threads. References required.
# ====================================================================

import numpy as np

def generate_text_prompt():
    print("=== PACKING REAL TEXT PROMPT: 'schrijf een test' ===")
    
    # We definiëren de 4D spatial embeddings voor de drie opeenvolgende woorden
    # NumPy zet deze gewone integers om naar keiharde signed 8-bit bytes (INT8)
    prompt_quaternions = np.array([
        [25,   10,  -5,   0],  # Token 1: "schrijf"
        [5,    30,  20, -10],  # Token 2: "een"
        [-15,   0,  45,  15]   # Token 3: "test"
    ], dtype=np.int8)

    output_filename = "prompt_activations.bin"
    with open(output_filename, "wb") as f:
        f.write(prompt_quaternions.tobytes())
        
    print(f"[+] SUCCESS: '{output_filename}' weggeschreven naar de workspace.")
    print("    Bevat de 4D spatial embeddings voor je driewoords-prompt.")

if __name__ == "__main__":
    generate_text_prompt()
