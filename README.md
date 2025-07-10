# MemeCMD

A Python-based tool for intelligent meme retrieval and matching based on role-based embeddings and similarity scoring.

## Overview

MemeCMD is a tool that helps match and retrieve memes based on role-based embeddings and similarity scoring. It uses a weighted cosine similarity approach to find the most relevant memes for given contexts.

## Features

- Role-based embedding processing
- Weighted cosine similarity computation
- Top-3 similar meme retrieval
- Similarity distribution visualization
- Batch processing support

## Project Structure

```
.
├── find_figures.py      # Maps indices to image filenames
├── retrieve.py         # Core retrieval and similarity computation
├── imgs/              # Image directory
├── Meme Warehouse/    # Contains meme embeddings and figures
├── Summary/           # Output summaries
├── Dialogs/          # Dialog data
├── Examples/           # Contains screenshots of some dialogs
├── Dialogs_with_meme/ # Stores dialogs that include memes
├── view-dialogs/      # Web-based visualization tool for browsing dialogs
├── metric             # Contains evaluation scripts
│   ├─clip_dialog_similarity_zh.py # for scoring with CLIP
│   └─gpt4o_score.py    # for scoring with GPT-4o

```

## Requirements

- Python 3.x
- NumPy
- Matplotlib
- Seaborn

## Usage

1. Prepare your role-based embeddings in the `roles-dialog-embedding` directory
2. Ensure your base embeddings are in `Meme_Warehouse/embeddings.npy`
3. Run the retrieval process:

```bash
python retrieve.py
```

4. Map the results to actual image files:

```bash
python find_figures.py
```

## How It Works

The system uses a weighted combination of multiple embedding similarities to find the most relevant memes:

- Computes cosine similarities between role-based and base embeddings
- Applies weights to different similarity components
- Retrieves top-3 matches based on combined similarity scores
- Generates similarity distribution visualizations
- Maps numerical indices to actual image files

## Output

The system generates:
- NPZ files containing top-3 indices and similarity scores
- JSON files mapping indices to actual image filenames
- Similarity distribution plots
- Detailed processing logs

## Notes

- The system uses a weighted scoring system with weights: 0.3, -0.2, 0.2, 0.7
- Batch processing is implemented to handle large datasets efficiently
- All embeddings are normalized before similarity computation 