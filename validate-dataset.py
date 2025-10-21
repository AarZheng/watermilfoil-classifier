from PIL import Image
import os

dataset_dir = "dataset"  # change if needed

bad_files = []
for root, _, files in os.walk(dataset_dir):
    for f in files:
        if f.lower().endswith((".jpg", ".jpeg", ".png")):
            path = os.path.join(root, f)
            try:
                img = Image.open(path)
                img.verify()  # check integrity
                print(f"Good file: {path}")
            except Exception as e:
                print(f"Bad file: {path} ({e})")
                bad_files.append(path)

print(f"Found {len(bad_files)} bad images")
