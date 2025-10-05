from flask import Flask, render_template, request
import numpy as np
import os
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import load_model

app = Flask(__name__)

# Load your trained CNN model ONCE at startup
MODEL_PATH = "plant_disease_cnn_model.h5"
print("Loading model, please wait...")
model = load_model(MODEL_PATH)
print("Model loaded successfully!")

# Define class names (must match model output order)
class_names = [
    "Potato___Early_blight",
    "Potato___Late_blight",
    "Potato___healthy",
    "Tomato___Early_blight",
    "Tomato___Late_blight",
    "Tomato___Leaf_Mold",
    "Tomato___healthy",
    "Corn___Common_rust",
    "Corn___Northern_Leaf_Blight",
    "Corn___healthy",
    "Rice___Brown_Spot",
    "Rice___Leaf_Blast",
    "Rice___healthy"
]

# Disease information dictionary
disease_info = {
    # POTATO
    "Potato___healthy": {
        "leaf_type": "Potato Leaf",
        "disease": "Healthy",
        "description": "No disease detected. The potato leaf appears green and vibrant.",
        "suggestions": "Maintain regular watering and a balanced fertilizer schedule.",
        "link": "https://plantvillage.psu.edu/topics/potato/infos"
    },
    "Potato___Early_blight": {
        "leaf_type": "Potato Leaf",
        "disease": "Early Blight",
        "description": "Caused by Alternaria solani fungus, leading to dark concentric rings on leaves.",
        "suggestions": "Use fungicides containing chlorothalonil or copper, and remove infected leaves.",
        "link": "https://plantvillage.psu.edu/topics/potato/infos"
    },
    "Potato___Late_blight": {
        "leaf_type": "Potato Leaf",
        "disease": "Late Blight",
        "description": "Caused by Phytophthora infestans fungus, leading to dark lesions and rot.",
        "suggestions": "Use copper-based fungicides and ensure proper spacing for air circulation.",
        "link": "https://plantvillage.psu.edu/topics/potato/infos"
    },

    # TOMATO
    "Tomato___healthy": {
        "leaf_type": "Tomato Leaf",
        "disease": "Healthy",
        "description": "No disease detected. The tomato plant is in healthy condition.",
        "suggestions": "Continue regular care and avoid overwatering.",
        "link": "https://plantvillage.psu.edu/topics/tomato/infos"
    },
    "Tomato___Early_blight": {
        "leaf_type": "Tomato Leaf",
        "disease": "Early Blight",
        "description": "Dark brown circular spots on leaves caused by Alternaria solani fungus.",
        "suggestions": "Use fungicides, avoid overhead irrigation, and rotate crops annually.",
        "link": "https://plantvillage.psu.edu/topics/tomato/infos"
    },
    "Tomato___Late_blight": {
        "leaf_type": "Tomato Leaf",
        "disease": "Late Blight",
        "description": "A serious fungal disease causing black lesions on leaves and fruit rot.",
        "suggestions": "Apply fungicides promptly and destroy infected plants.",
        "link": "https://plantvillage.psu.edu/topics/tomato/infos"
    },
    "Tomato___Leaf_Mold": {
        "leaf_type": "Tomato Leaf",
        "disease": "Leaf Mold",
        "description": "Caused by Passalora fulva fungus, creating yellow spots and fuzzy mold.",
        "suggestions": "Improve air circulation, reduce humidity, and apply fungicides.",
        "link": "https://plantvillage.psu.edu/topics/tomato/infos"
    },

    # CORN
    "Corn___healthy": {
        "leaf_type": "Maize Leaf",
        "disease": "Healthy",
        "description": "The corn leaf appears strong, green, and free from spots.",
        "suggestions": "Maintain consistent watering and avoid overcrowding.",
        "link": "https://plantvillage.psu.edu/topics/maize/infos"
    },
    "Corn___Common_rust": {
        "leaf_type": "Maize Leaf",
        "disease": "Common Rust",
        "description": "Caused by Puccinia sorghi fungus forming orange pustules on leaves.",
        "suggestions": "Plant resistant hybrids and use fungicides if severe.",
        "link": "https://plantvillage.psu.edu/topics/maize/infos"
    },
    "Corn___Northern_Leaf_Blight": {
        "leaf_type": "Maize Leaf",
        "disease": "Northern Leaf Blight",
        "description": "Long gray-green lesions on leaves caused by Exserohilum turcicum.",
        "suggestions": "Rotate crops and apply fungicides like mancozeb or carbendazim.",
        "link": "https://plantvillage.psu.edu/topics/maize/infos"
    },

    # RICE
    "Rice___healthy": {
        "leaf_type": "Rice Leaf",
        "disease": "Healthy",
        "description": "The rice leaf shows no infection or discoloration.",
        "suggestions": "Follow good water management and fertilization practices.",
        "link": "https://plantvillage.psu.edu/topics/rice/infos"
    },
    "Rice___Brown_Spot": {
        "leaf_type": "Rice Leaf",
        "disease": "Brown Spot",
        "description": "Caused by Bipolaris oryzae fungus forming brown lesions on leaves.",
        "suggestions": "Use resistant varieties and apply appropriate fungicides.",
        "link": "https://plantvillage.psu.edu/topics/rice/infos"
    },
    "Rice___Leaf_Blast": {
        "leaf_type": "Rice Leaf",
        "disease": "Leaf Blast",
        "description": "Irregular diamond-shaped spots caused by Magnaporthe oryzae fungus.",
        "suggestions": "Reduce nitrogen use, maintain field hygiene, and apply tricyclazole.",
        "link": "https://plantvillage.psu.edu/topics/rice/infos"
    }
}

# Home route
@app.route('/')
def home():
    return render_template('index.html')

# Predict route
@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return render_template('result.html', error="No file uploaded.")
    file = request.files['file']
    if file.filename == '':
        return render_template('result.html', error="No file selected.")

    # Save uploaded file
    file_path = os.path.join('static', file.filename)
    file.save(file_path)

    # Preprocess image efficiently
    img = image.load_img(file_path, target_size=(128, 128))
    img_array = image.img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    # Make prediction
    predictions = model.predict(img_array)
    predicted_index = np.argmax(predictions, axis=1)[0]
    confidence = round(100 * np.max(predictions), 2)

    # Threshold for unknown leaf detection
    THRESHOLD = 60.0  # confidence < 60% considered unknown
    if confidence < THRESHOLD or predicted_index >= len(class_names):
        return render_template(
            'result.html',
            prediction="Unknown Leaf",
            confidence=confidence,
            info={
                "leaf_type": "Unknown",
                "disease": "Unknown",
                "description": "This leaf may not belong to the supported crops (Potato, Tomato, Corn, Rice).",
                "suggestions": "Please use a leaf from supported crops for reliable prediction.",
                "link": "#"
            },
            image_path=file_path
        )

    # Valid prediction
    predicted_class = class_names[predicted_index]
    info = disease_info.get(predicted_class, {
        "leaf_type": "Unknown",
        "disease": predicted_class,
        "description": "No details available for this disease.",
        "suggestions": "Please consult an agricultural expert.",
        "link": "#"
    })

    return render_template(
        'result.html',
        prediction=predicted_class,
        confidence=confidence,
        info=info,
        image_path=file_path
    )

if __name__ == '__main__':
    # Run with lower debug/memory options for Render free plan
    app.run(host='0.0.0.0', port=int(os.environ.get("PORT", 5000)))