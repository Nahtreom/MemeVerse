import json
import numpy as np
import os
import csv

FOLDER = 'score_gpt4o'
OUTPUT_CSV = 'score_gpt4o/score_summary.csv'

results = []

for filename in os.listdir(FOLDER):
    if filename.endswith('.json'):
        filepath = os.path.join(FOLDER, filename)
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
        scores = [
            item['综合得分']
            for item in data
            if '综合得分' in item
            and item['综合得分'] is not None
            and (
                item['综合得分'] != 0 or (
                    '原因' not in item or item['原因'] is not None
                )
            )
            and ('原因' not in item or item['原因'] is not None)
        ]
        print(filename)
        print("Total:", len(data))
        print("Valid:", len(scores))
        avg_score = np.mean(scores) if scores else None
        results.append({
            'Filename': filename,
            'Total': len(data),
            'Valid': len(scores),
            'Average': f"{avg_score:.2f}" if avg_score is not None else 'No valid score'
        })

# Write to CSV
with open(OUTPUT_CSV, 'w', encoding='utf-8', newline='') as csvfile:
    fieldnames = ['Filename', 'Total', 'Valid', 'Average']
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
    writer.writeheader()
    for row in results:
        writer.writerow(row)

print(f"Results written to {OUTPUT_CSV}")