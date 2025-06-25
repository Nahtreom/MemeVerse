import numpy as np
import json
import os
from pathlib import Path

def map_top3_indices_to_images(npz_file: str, figure_json_file: str, output_json_file: str):
    print("[INFO] Loading top3 results from:", npz_file)
    data = np.load(npz_file)
    top_indices = data['top_indices']  # shape (6600, 3)
    top_scores = data['top_scores']    # shape (6600, 3)

    print("[INFO] Loading image filenames from:", figure_json_file)
    with open(figure_json_file, 'r') as f:
        image_list = json.load(f)

    if len(image_list) <= np.max(top_indices):
        raise ValueError("[ERROR] Some indices in top3 exceed the length of figure.json")

    print("[INFO] Mapping top indices to image filenames...")
    result = []
    for i, (indices, scores) in enumerate(zip(top_indices, top_scores)):
        images = [image_list[idx] for idx in indices]
        image_with_scores = [
            {"image": img, "similarity_score": float(score)} 
            for img, score in zip(images, scores)
        ]
        result.append(image_with_scores)

    output_dir = os.path.dirname(output_json_file)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    with open(output_json_file, 'w') as f:
        json.dump(result, f, indent=2)

    print(f"[INFO] Saved mapped image results to {output_json_file}")
    print("[INFO] Example (first 2 entries):")
    for i in range(2):
        print(f"  Sample {i}:")
        for item in result[i]:
            print(f"    Image: {item['image']}, Score: {item['similarity_score']:.4f}")

def process_directory(input_dir: str, figure_json_file: str = 'Meme_Warehouse/figures.json'):

    if not os.path.exists(input_dir):
        raise ValueError(f"Directory not found: {input_dir}")
    
    npz_files = [f for f in os.listdir(input_dir) if f.endswith('_top3_results.npz')]
    
    if not npz_files:
        print(f"[WARNING] No *_top3_results.npz files found in {input_dir}")
        return
    
    print(f"[INFO] Found {len(npz_files)} .npz files in {input_dir}")
    
    for npz_file in npz_files:
        input_path = os.path.join(input_dir, npz_file)
        output_file = os.path.join(
            os.path.dirname(input_path),
            f"{Path(npz_file).stem.replace('_top3_results', '')}_top3_image_results.json"
        )
        
        print(f"\n[INFO] Processing file: {npz_file}")
        print(f"[INFO] Output will be saved to: {output_file}")
        
        try:
            map_top3_indices_to_images(input_path, figure_json_file, output_file)
        except Exception as e:
            print(f"[ERROR] Failed to process {npz_file}: {str(e)}")
            continue

if __name__ == '__main__':
    input_dir = 'roles-dialog-embedding'  
    process_directory(input_dir)
