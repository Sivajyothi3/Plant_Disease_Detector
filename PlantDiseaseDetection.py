import os
import cv2
import numpy as np
from tqdm import tqdm
from sklearn.preprocessing import LabelBinarizer
from tensorflow.keras.preprocessing.image import ImageDataGenerator  # ✅ fixed
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from sklearn.model_selection import train_test_split
# -----------------------------
# Paths
# -----------------------------
directory_root = r"C:\PlantCNN\Plant-Disease-Identification-using-CNN-master\input\PlantVillage_clean"

# -----------------------------
# Parameters
# -----------------------------
IMAGE_DIMS = (128, 128)  # resize all images to 128x128
BATCH_SIZE = 32
EPOCHS = 20

# -----------------------------
# Prepare lists
# -----------------------------
data = []
labels = []

# -----------------------------
# Load images
# -----------------------------
print("[INFO] Loading images...")
if not os.path.exists(directory_root):
    raise ValueError(f"[ERROR] The directory '{directory_root}' does not exist!")

folders = [f for f in os.listdir(directory_root) if os.path.isdir(os.path.join(directory_root, f))]
if len(folders) == 0:
    raise ValueError(f"[ERROR] No class folders found in '{directory_root}'!")

for folder in tqdm(folders, desc="Classes"):
    folder_path = os.path.join(directory_root, folder)
    for image in os.listdir(folder_path):
        imagePath = os.path.join(folder_path, image)
        try:
            img = cv2.imread(imagePath)
            if img is None:
                continue
            img = cv2.resize(img, IMAGE_DIMS)
            data.append(img)
            labels.append(folder)
        except Exception as e:
            print(f"[WARNING] Could not process image {imagePath}: {e}")

# -----------------------------
# Convert to NumPy arrays
# -----------------------------
data = np.array(data, dtype="float32") / 255.0
labels = np.array(labels)

print(f"[INFO] Total images loaded: {len(data)}")
print(f"[INFO] Total labels loaded: {len(labels)}")

# -----------------------------
# Encode labels
# -----------------------------
label_binarizer = LabelBinarizer()
image_labels = label_binarizer.fit_transform(labels)
print(f"[INFO] Classes found: {label_binarizer.classes_}")

# -----------------------------
# Train/test split
# -----------------------------
X_train, X_test, y_train, y_test = train_test_split(
    data, image_labels, test_size=0.2, random_state=42
)

# -----------------------------
# Data augmentation
# -----------------------------
aug = ImageDataGenerator(
    rotation_range=20,
    zoom_range=0.15,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.15,
    horizontal_flip=True,
    fill_mode="nearest"
)

# -----------------------------
# Build CNN model
# -----------------------------
model = Sequential([
    Conv2D(32, (3, 3), activation="relu", input_shape=(128, 128, 3)),
    MaxPooling2D(pool_size=(2, 2)),
    
    Conv2D(64, (3, 3), activation="relu"),
    MaxPooling2D(pool_size=(2, 2)),

    Conv2D(128, (3, 3), activation="relu"),
    MaxPooling2D(pool_size=(2, 2)),

    Flatten(),
    Dense(128, activation="relu"),
    Dropout(0.5),
    Dense(len(label_binarizer.classes_), activation="softmax")
])

model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
model.summary()

# -----------------------------
# Train the model
# -----------------------------
H = model.fit(
    aug.flow(X_train, y_train, batch_size=BATCH_SIZE),
    validation_data=(X_test, y_test),
    steps_per_epoch=len(X_train) // BATCH_SIZE,
    epochs=EPOCHS
)

# -----------------------------
# Save the trained model
# -----------------------------
model.save("plant_disease_cnn_model.h5")
print("[INFO] Model saved as 'plant_disease_cnn_model.h5'")