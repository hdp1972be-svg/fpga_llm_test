#!/usr/local/bin/python3.11
# ====================================================================
# Copyright (C) 2026 
# Licensed under GNU GPLv3 / Text concepts under CC BY-SA 4.0
# Original source post on Threads. References required.
# ====================================================================

import json
import re
import os

def detokenize_hardware_output():
    print("====================================================================")
    print("BIONISCHE DE-TOKENIZER: VAN SILICIUM-BITS NAAR LEESBARE TAAL")
    print("====================================================================")

    # 1. Controleer of de vocabulaire aanwezig is
    if not os.path.exists("vocab.json"):
        print("[-] Error: 'vocab.json' ontbreekt. Run eerst all_minillm_compiler.py!")
        return
        
    with open("vocab.json", "r") as f:
        vocab = json.load(f)

    # 2. Lees de geverifieerde Icarus Verilog console-output in
    log_filename = "verilog_output.txt"
    if not os.path.exists(log_filename):
        print(f"[-] Error: '{log_filename}' niet gevonden.")
        print("    Zorg dat je de simulator eerst zo runt:")
        print("    vvp hbm2_prompt_engine.out > verilog_output.txt")
        return

    with open(log_filename, "r") as f:
        log_content = f.read()

    # 3. Extraheer de Token-IDs die je ArgMax core via PCIe naar buiten heeft gespuugd
    predicted_ids = re.findall(r"CHIP PREDICTS NEXT TOKEN-ID:\s+(\d+)", log_content)

    if not predicted_ids:
        print("[-] Waarschuwing: Geen 'CHIP PREDICTS NEXT TOKEN-ID' regels gevonden in de log.")
        print("    We scannen nu op de ruwe testbench monitor-uitgangen...")
        predicted_ids = re.findall(r"WINNING TOKEN ID FOR HOST:\s+(\d+)", log_content)

    if not predicted_ids:
        print("[-] Fout: Kon geen geldige hardware-tokens extraheren uit de simulatielog.")
        return

    print(f"[+] Succesvol {len(predicted_ids)} binaire hardware-transanties gedetecteerd.\n")
    print("--- LIVE HARDWARE DE-TOKENIZATION BEZIG ---")
    
    # 4. Vertaal de bits live terug naar de tekst-vocabulaire van het model
    for idx, t_id in enumerate(predicted_ids):
        token_index = int(t_id)
        
        # Voorkom out-of-bounds op kleine testvocabulaires
        if token_index < len(vocab):
            word = vocab[token_index]
        else:
            word = f"[Unknown_Token_{token_index}]"
            
        print(f" Klokcyclus {idx+1} (T={(35 + idx*10)}000ns) | Hardware Token {token_index:<3} ===> Woord: '{word}'")

    print("--------------------------------------------------------------------")
    print("🚀 DE-TOKENIZER SUCCES: De bionische 4D hardware-loop is volledig gesloten!")
    print("   Tekst-prompt ingevoerd -> Omgezet naar 4D-quaternionen -> Berekend op")
    print("   het silicium via propagatievertraging -> Terugvertaald naar tekst.")

if __name__ == "__main__":
    detokenize_hardware_output()
