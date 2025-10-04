import os

# Update this path to your PlantVillage folder
directory_root = r"C:\PlantCNN\Plant-Disease-Identification-using-CNN-master\input\PlantVillage"

if not os.path.exists(directory_root):
    print(f"[ERROR] Path does not exist: {directory_root}")
else:
    print(f"[INFO] Found root folder: {directory_root}")
    plant_folders = os.listdir(directory_root)
    print(f"[INFO] Plant folders found ({len(plant_folders)}): {plant_folders}\n")

    total_images = 0
    for plant in plant_folders:
        plant_path = os.path.join(directory_root, plant)
        if not os.path.isdir(plant_path):
            continue
        disease_folders = os.listdir(plant_path)
        print(f"  Plant: {plant} -> Disease folders: {disease_folders}")
        for disease in disease_folders:
            disease_path = os.path.join(plant_path, disease)
            if not os.path.isdir(disease_path):
                continue
            images = [f for f in os.listdir(disease_path) if f.lower().endswith((".jpg", ".jpeg", ".png"))]
            print(f"    Disease: {disease} -> {len(images)} images")
            total_images += len(images)

    print(f"\n[INFO] Total images found: {total_images}")