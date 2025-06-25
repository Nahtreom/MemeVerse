import numpy as np
from numpy.linalg import norm
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import os

def plot_similarity_distribution(sims, save_dir='./plots', filename='similarity_distribution'):
    Path(save_dir).mkdir(parents=True, exist_ok=True)
    
    flat_sims = sims.flatten()
    
    plt.figure(figsize=(12, 6))
    
    sns.histplot(data=flat_sims, bins=50, kde=True)
    plt.title(f'Distribution of Similarity Scores - {filename}')
    plt.xlabel('Similarity Score')
    plt.ylabel('Count')
    
    stats_text = f'Mean: {flat_sims.mean():.4f}\n'
    stats_text += f'Std: {flat_sims.std():.4f}\n'
    stats_text += f'Min: {flat_sims.min():.4f}\n'
    stats_text += f'Max: {flat_sims.max():.4f}'
    plt.text(0.95, 0.95, stats_text,
             transform=plt.gca().transAxes,
             verticalalignment='top',
             horizontalalignment='right',
             bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    plt.savefig(f'{save_dir}/{filename}.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"[INFO] Saved similarity distribution plot to {save_dir}/{filename}.png")

def compute_weighted_top3(role_file: str, emb_file: str, out_file: str):

    w1, w2, w3, w4 = 0.3, -0.2, 0.2, 0.7

    print("[INFO] Loading role-based embeddings from:", role_file)
    role = np.load(role_file)  # shape (6600, 3, 2304)
    print("[INFO] Loaded role-based embeddings with shape:", role.shape)

    print("[INFO] Loading base embeddings from:", emb_file)
    emb = np.load(emb_file)  # shape (6024, 4, 2304)
    print("[INFO] Loaded base embeddings with shape:", emb.shape)

    role_0 = role[:, 0, :]  # (6600, 2304)
    role_1 = role[:, 1, :]
    role_2 = role[:, 2, :]

    emb_0 = emb[:, 0, :]    # (6024, 2304)
    emb_1 = emb[:, 1, :]
    emb_2 = emb[:, 2, :]
    emb_3 = emb[:, 3, :]

    def normalize(x):
        norms = norm(x, axis=1, keepdims=True)
        norms[norms == 0] = 1e-8
        return x / norms

    role_0 = normalize(role_0)
    role_1 = normalize(role_1)
    role_2 = normalize(role_2)

    emb_0 = normalize(emb_0)
    emb_1 = normalize(emb_1)
    emb_2 = normalize(emb_2)
    emb_3 = normalize(emb_3)

    n_role = role.shape[0]
    n_emb = emb.shape[0]

    print(f"[INFO] Computing weighted cosine similarities: {n_role} roles vs {n_emb} embeddings")

    top_indices = np.zeros((n_role, 3), dtype=np.int32)
    top_scores = np.zeros((n_role, 3), dtype=np.float32)

    batch_size = 100
    for start in range(0, n_role, batch_size):
        end = min(start + batch_size, n_role)

        sims0 = role_0[start:end] @ emb_0.T  # (B, 6024)
        sims1 = role_0[start:end] @ emb_1.T
        sims2 = role_1[start:end] @ emb_2.T
        sims3 = role_2[start:end] @ emb_3.T

        sims = w1 * sims0 + w2 * sims1 + w3 * sims2 + w4 * sims3

        if start == 0: 
            plot_similarity_distribution(sims, filename=Path(role_file).stem)

        idx = np.argpartition(-sims, 3, axis=1)[:, :3]
        for i in range(end - start):
            row_idx = idx[i]
            row_sims = sims[i, row_idx]
            order = np.argsort(-row_sims)
            top_indices[start + i] = row_idx[order]
            top_scores[start + i] = row_sims[order]

        print(f"[INFO] Processed samples {start} to {end-1} ({end}/{n_role})")

    np.savez(out_file, top_indices=top_indices, top_scores=top_scores)
    print("[INFO] Saved top-3 results to:", out_file)
    print("[INFO] top_indices shape:", top_indices.shape)
    print("[INFO] top_scores  shape:", top_scores.shape)
    print("[INFO] Sample results:")
    for i in range(3):
        print(f"  Sample {i}: indices={top_indices[i]}, scores={top_scores[i]}")

def process_directory(role_dir: str, emb_file: str):

    if not os.path.exists(role_dir):
        raise ValueError(f"Directory not found: {role_dir}")
    
    npy_files = [f for f in os.listdir(role_dir) if f.endswith('.npy')]
    
    if not npy_files:
        print(f"[WARNING] No .npy files found in {role_dir}")
        return
    
    print(f"[INFO] Found {len(npy_files)} .npy files in {role_dir}")
    
    for npy_file in npy_files:
        role_path = os.path.join(role_dir, npy_file)
        out_path = os.path.join(os.path.dirname(role_path),
                               f"{Path(npy_file).stem}_top3_results.npz")
        
        print(f"\n[INFO] Processing file: {npy_file}")
        print(f"[INFO] Output will be saved to: {out_path}")
        
        try:
            compute_weighted_top3(role_path, emb_file, out_path)
        except Exception as e:
            print(f"[ERROR] Failed to process {npy_file}: {str(e)}")
            continue

if __name__ == '__main__':
    role_dir = 'roles-dialog-embedding' 
    emb_path = 'Meme_Warehouse/embeddings.npy'
    process_directory(role_dir, emb_path)
