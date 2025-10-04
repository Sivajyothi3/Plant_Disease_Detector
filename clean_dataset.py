import os
import shutil

# Path to your extracted dataset
dataset_path = r"C:\PlantCNN\Plant-Disease-Identification-using-CNN-master\input\PlantVillage"

# Path for the cleaned dataset
output_path = r"C:\PlantCNN\Plant-Disease-Identification-using-CNN-master\input\PlantVillage_clean"

# Create output directory if it doesn't exist
os.makedirs(output_path, exist_ok=True)

bad_files = []  # To store invalid files
kept_files = 0  # Counter for valid images

# Walk through the dataset
for subdir, _, files in os.walk(dataset_path):
    for fname in files:
        ext = fname.lower().split(".")[-1]
        base_name = os.path.splitext(fname)[0]

        # Keep only jpg/jpeg images without spaces
        # Remove the space check
        if ext in ("jpg", "jpeg"):
            src = os.path.join(subdir, fname)
            new_fname = os.path.splitext(fname)[0] + ".jpg"
            rel_path = os.path.relpath(subdir, dataset_path)
            dest_dir = os.path.join(output_path, rel_path)
            os.makedirs(dest_dir, exist_ok=True)
            dest = os.path.join(dest_dir, new_fname)
            shutil.copy2(src, dest)
            kept_files += 1
        else:
            bad_files.append(os.path.join(subdir, fname))

print(f"✅ Copied {kept_files} valid .jpg images (including renamed .jpeg).")
print(f"⚠️ Found {len(bad_files)} invalid images (non-jpg/jpeg or with spaces).")