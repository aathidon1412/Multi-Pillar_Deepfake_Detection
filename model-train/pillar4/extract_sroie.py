import os
import csv

input_dir = r"SROIE2019\test\box"
output_csv = "sroie_real.csv"

with open(output_csv, 'w', newline='', encoding='utf-8') as out_f:
    writer = csv.writer(out_f)
    writer.writerow(["Extracted_Text"])
    
    for filename in os.listdir(input_dir):
        if filename.endswith(".txt"):
            filepath = os.path.join(input_dir, filename)
            with open(filepath, 'r', encoding='utf-8', errors='ignore') as in_f:
                for line in in_f:
                    parts = line.strip().split(',')
                    if len(parts) >= 9:
                        text = ','.join(parts[8:])
                        writer.writerow([text])

print(f"Extraction complete. Saved to {output_csv}")
