import os
import json
import csv

SCORE_FOLDER = 'clip-score'

# Get all *_result.json files in the current directory
result_files = [f for f in os.listdir(SCORE_FOLDER) if f.endswith('_result.json')]

csv_rows = []
for file in result_files:
    with open(SCORE_FOLDER+'/'+file, 'r', encoding='utf-8') as f:
        try:
            data = json.load(f)
        except Exception as e:
            print(f"[Error] Failed to read {file}: {e}")
            continue
    ave_sims = [d.get('average_similarity') for d in data if d.get('average_similarity') is not None]
    count = len(ave_sims)
    mean_ave_sim = sum(ave_sims) / count if count > 0 else None
    norm_score = (mean_ave_sim + 1) / 2 * 100 if mean_ave_sim is not None else None
    csv_rows.append([file, count, mean_ave_sim, norm_score])

# Output to csv
with open('all_average_similarity_summary.csv', 'w', encoding='utf-8', newline='') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(['file', 'dialog_count', 'average_of_average_similarity', 'normalized_score'])
    for row in csv_rows:
        writer.writerow(row)

print('✅ Statistics completed, results saved to all_average_similarity_summary.csv')
