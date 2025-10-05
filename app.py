import gradio as gr
import numpy as np
from PIL import Image
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import load_model

# Load trained CNN model
MODEL_PATH = "plant_disease_cnn_model.h5"
print("Loading model, please wait...")
model = load_model(MODEL_PATH)
print("Model loaded successfully!")

# Class names
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

# Disease information
disease_info = {
    # Potato
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
    # Tomato
    "Tomato___healthy": {
        "leaf_type": "Tomato Leaf",
        "disease": "Healthy",
        "description": "No disease detected. The tomato plant is healthy.",
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
    # Corn
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
    # Rice
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

THRESHOLD = 80.0  # minimum confidence for valid prediction

def predict_leaf(img: Image.Image):
    img_array = image.img_to_array(img.resize((128, 128))) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    predictions = model.predict(img_array)
    predicted_index = np.argmax(predictions)
    confidence = 100 * np.max(predictions)

    if confidence < THRESHOLD:
        return f"""
        <h3>⚠️ Unsupported Leaf</h3>
        <p>Confidence: {confidence:.2f}%</p>
        <p>Only Potato, Tomato, Corn, and Rice leaves are supported.</p>
        """, img

    predicted_class = class_names[predicted_index]
    info = disease_info[predicted_class]

    return f"""
    <h3>Prediction: {predicted_class}</h3>
    <p><b>Confidence:</b> {confidence:.2f}%</p>
    <p><b>Leaf Type:</b> {info['leaf_type']}</p>
    <p><b>Disease:</b> {info['disease']}</p>
    <p><b>Description:</b> {info['description']}</p>
    <p><b>Suggestions:</b> {info['suggestions']}</p>
    <p><b>More Info:</b> <a href="{info['link']}" target="_blank">Click Here</a></p>
    """, img

# Gradio Interface
ui = gr.Interface(
    fn=predict_leaf,
    inputs=gr.Image(type="pil", label="Upload Leaf Image"),
    outputs=[gr.HTML(label="Prediction Details"), gr.Image(label="Uploaded Leaf")],
    title="Plant Disease Detector",
    description="Upload a leaf image from Potato, Tomato, Corn, or Rice to detect its disease.",
    allow_flagging="never"
)

ui.launch()