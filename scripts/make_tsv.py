import os
import csv

ROOT_DIR = "/workspace/5500videos"
OUTPUT_TSV = "metadata_video.tsv"

rows = []

IMAGE_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
VIDEO_EXT = {".mp4", ".avi", ".mov", ".mkv"}

for root, dirs, files in os.walk(ROOT_DIR):
    for f in files:
        path = os.path.join(root, f)
        ext = os.path.splitext(f)[1].lower()

        # modality 판별
        if ext in IMAGE_EXT:
            modality = "image"
        elif ext in VIDEO_EXT:
            modality = "video"
        else:
            continue

        # ----- Label 판별 -----
        lower_path = path.lower()

        if "/real" in lower_path:
            label = "real"
        elif "/fake" in lower_path:
            label = "fake"
        else:
            continue   # 어디에도 속하지 않으면 스킵

        rows.append([path, label, modality])

# TSV 저장
with open(OUTPUT_TSV, "w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f, delimiter="\t")
    writer.writerow(["path", "label", "modality"])
    writer.writerows(rows)

print(f"TSV saved to: {OUTPUT_TSV}")
print(f"Total samples: {len(rows)}")
