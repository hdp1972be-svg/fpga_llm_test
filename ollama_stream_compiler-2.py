#!/usr/local/bin/python3.11
import json
import os
import sys

def find_ollama_blob_by_name(model_name, base_dir="/mnt/btrfs/@ollama"):
    """
    Scant de Ollama manifest-mappen en zoekt de exacte SHA256-hash van de GGUF-modelblob.
    Accepteert formaten zoals 'qwen2.5-coder:1.5b' of 'hf.co/Maincode/Maincoder-1B-GGUF:Q4_K_M'.
    """
    manifest_found = None
    
    # 1. Splits de modelnaam en tag (default naar 'latest' als er geen tag is)
    if ":" in model_name:
        name_part, tag = model_name.split(":", 1)
    else:
        name_part, tag = model_name, "latest"
        
    # 2. Bepaal de mogelijke locaties van het manifest-bestand
    possible_paths = [
        os.path.join(base_dir, "manifests", "registry.ollama.ai", "library", name_part, tag),
        os.path.join(base_dir, "manifests", name_part, tag) # Voor hf.co / custom mappen
    ]
    
    for path in possible_paths:
        if os.path.exists(path):
            manifest_found = path
            break
            
    if not manifest_found:
        print(f"[-] Error: Manifest voor model '{model_name}' niet gevonden.")
        print("    Gecontroleerde paden:")
        for p in possible_paths: print(f"      - {p}")
        return None

    # 3. Lees het JSON manifest uit
    with open(manifest_found, "r") as f:
        try:
            manifest_data = json.load(f)
        except Exception as e:
            print(f"[-] Error tijdens parsen van manifest JSON: {e}")
            return None

    # 4. Zoek naar de hoofd-model-laag (meestal de grootste GGUF laag)
    model_digest = None
    for layer in manifest_data.get("layers", []):
        media_type = layer.get("mediaType", "")
        # Ollama specificeert model-blobs met deze specifieke media-types
        if "gguf" in media_type or "model" in media_type:
            model_digest = layer.get("digest", "")
            break
            
    # Fallback: als mediaType niet matcht, pak de grootste laag qua omvang
    if not model_digest and manifest_data.get("layers"):
        largest_layer = max(manifest_data["layers"], key=lambda x: x.get("size", 0))
        model_digest = largest_layer.get("digest", "")

    if not model_digest:
        print(f"[-] Error: Kon geen model-digest vinden in het manifest van {model_name}.")
        return None

    # Ollama slaat hashes op als 'sha256:hash_waarde', we hebben de bestandsnaam nodig
    blob_filename = model_digest.replace("sha256:", "sha256-")
    full_blob_path = os.path.join(base_dir, "blobs", blob_filename)
    
    if os.path.exists(full_blob_path):
        print(f"[+] Model '{model_name}' succesvol gemapped naar:")
        print(f"    {full_blob_path} ({os.path.getsize(full_blob_path) / (1024*1024):.2f} MB)")
        return full_blob_path
    else:
        print(f"[-] Error: Manifest verwijst naar blob {blob_filename}, maar bestand ontbreekt in blobs map.")
        return None

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Gebruik: ./ollama_blob_compiler.py <modelnaam:tag>")
        print("Voorbeeld: ./ollama_blob_compiler.py qwen2.5-coder:1.5b")
        sys.exit(1)
        
    model_input = sys.argv[1]
    blob_path = find_ollama_blob_by_name(model_input)
    
    if blob_path:
        print("\n[!] Je kunt nu de stream-compiler starten met dit geverifieerde pad!")
        # Hier kun je direct je 'compile_ollama_blob(blob_path, "fpga_hbm2_stream.bin")' aanroepen!
