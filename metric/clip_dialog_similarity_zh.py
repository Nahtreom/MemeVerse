import os
import json
from tqdm import tqdm
from PIL import Image
import torch
import torch.nn.functional as F
from transformers import ChineseCLIPProcessor, ChineseCLIPModel
import glob

os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

# Path configuration
IMAGE_ROOT = r"../Meme Warehouse/EmojoPackage_processed"
INPUT_DIR = "../Dialogs_with_meme"

def clean_prefix(s):
    """Remove A:/B: prefix and extra spaces"""
    if s.startswith("A:") or s.startswith("B:"):
        return s[2:].strip()
    return s.strip()

def is_image_file(s):
    return s.lower().endswith(('.jpg', '.png'))

def calculate_similarity(model, processor, image_path, text, device):
    """Get image-text cosine similarity"""
    image = Image.open(image_path).convert('RGB')
    inputs = processor(images=image, text=text, return_tensors="pt", padding=True)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.no_grad():
        image_embeds = model.get_image_features(pixel_values=inputs["pixel_values"])
        text_embeds = model.get_text_features(input_ids=inputs["input_ids"], attention_mask=inputs["attention_mask"])
    image_embeds = F.normalize(image_embeds, p=2, dim=-1)
    text_embeds = F.normalize(text_embeds, p=2, dim=-1)
    similarity = F.cosine_similarity(image_embeds, text_embeds).item()
    return similarity

# Resume from breakpoint: extract finished dialog_id from written json

def load_finished_dialog_ids(output_path):
    if not os.path.exists(output_path):
        return set(), []
    with open(output_path, 'r', encoding='utf-8') as f:
        try:
            data = json.load(f)
        except Exception:
            return set(), []
    finished_ids = set()
    for entry in data:
        finished_ids.add(entry.get("dialog_id"))
    return finished_ids, data

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🚀 Current device: {device}")
    if device.type == 'cuda':
        print(f"🧠 Using GPU: {torch.cuda.get_device_name(0)}")

    model = ChineseCLIPModel.from_pretrained("OFA-Sys/chinese-clip-vit-base-patch16").to(device)
    processor = ChineseCLIPProcessor.from_pretrained("OFA-Sys/chinese-clip-vit-base-patch16")

    json_files = glob.glob(os.path.join(INPUT_DIR, '*.json'))
    print(f"Found {len(json_files)} json files to process.")
    for input_json in json_files:
        file_name = os.path.basename(input_json)
        output_json = file_name.replace('.json', '_result.json')
        print(f"\nProcessing: {file_name}")
        with open(input_json, 'r', encoding='utf-8') as f:
            dialogs = json.load(f)
        finished_ids, finished_data = load_finished_dialog_ids(output_json)
        results = finished_data.copy() if finished_data else []
        for dialog in tqdm(dialogs, desc=f"Processing dialog {file_name}"):
            dialog_id = dialog.get('dialog_id', '')
            if dialog_id in finished_ids:
                continue
            full_dialog = [clean_prefix(x) for x in dialog.get('full_dialog', [])]
            dialog_results = []
            i = 0
            while i < len(full_dialog) - 1:
                cur = full_dialog[i]
                nxt = full_dialog[i + 1]
                if is_image_file(cur) and not is_image_file(nxt):
                    image_path = os.path.join(IMAGE_ROOT, cur)
                    text = nxt
                    if os.path.exists(image_path):
                        try:
                            sim = calculate_similarity(model, processor, image_path, text, device)
                        except Exception as e:
                            sim = None
                            print(f"[Error] Similarity calculation failed: {image_path}, {text}, {e}")
                    else:
                        sim = None
                        print(f"[Warning] Image does not exist: {image_path}")
                    dialog_results.append({
                        "image": cur,
                        "text": text,
                        "similarity": sim
                    })
                i += 1
            valid_sims = [x["similarity"] for x in dialog_results if x["similarity"] is not None]
            avg_similarity = sum(valid_sims) / len(valid_sims) if valid_sims else None
            result_entry = {
                "dialog_id": dialog_id,
                "image_text_pairs": dialog_results,
                "average_similarity": avg_similarity
            }
            results.append(result_entry)
            # Save in real time to prevent data loss due to interruption
            with open(output_json, 'w', encoding='utf-8') as f:
                json.dump(results, f, ensure_ascii=False, indent=2)
        print(f"✅ {file_name} processed and written to {output_json}")
    print(f"\n🎉 All files processed!")

if __name__ == "__main__":
    main()
