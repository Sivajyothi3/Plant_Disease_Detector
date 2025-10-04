import os

# Hide TensorFlow info/warning logs
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # 0=all, 1=info, 2=warning, 3=error

from flask import Flask, request, render_template
import numpy as np

# TensorFlow / Keras imports
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from tensorflow.keras.layers import Input
from tensorflow.keras.mixed_precision import DTypePolicy

# Fix for InputLayer 'batch_shape' issue
def custom_input_layer(*args, **kwargs):
    if 'batch_shape' in kwargs:
        kwargs['shape'] = kwargs.pop('batch_shape')[1:]
    return Input(*args, **kwargs)

# Register custom objects
custom_objects = {
    'InputLayer': custom_input_layer,
    'DTypePolicy': DTypePolicy
}

# Initialize Flask app
app = Flask(__name__)

# Load trained CNN model
MODEL_PATH = "plant_disease_cnn_model.h5"
print("Loading model, please wait...")
model = load_model(MODEL_PATH, compile=False, custom_objects=custom_objects)
print("Model loaded successfully!")

# Define class labels (same order as your model)
class_labels = [
    'Pepper__bell___Bacterial_spot',
    'Pepper__bell___healthy',
    'Potato___Early_blight',
    'Potato___Late_blight',
    'Potato___healthy',
    'Tomato_Bacterial_spot',
    'Tomato_Early_blight',
    'Tomato_Late_blight',
    'Tomato_Leaf_Mold',
    'Tomato_Septoria_leaf_spot',
    'Tomato_Spider_mites_Two_spotted_spider_mite',
    'Tomato_healthy'
]

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return "No file uploaded!", 400

    file = request.files['file']
    if file.filename == '':
        return "No file selected!", 400

    # Save file temporarily
    filepath = os.path.join("static", file.filename)
    file.save(filepath)

    # Preprocess image
    img = image.load_img(filepath, target_size=(128, 128))
    img_array = image.img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    # Predict
    predictions = model.predict(img_array)
    predicted_class = class_labels[np.argmax(predictions)]
    confidence = round(100 * np.max(predictions), 2)

    return render_template('result.html',
                           filename=file.filename,
                           predicted_class=predicted_class,
                           confidence=confidence)

if __name__ == "__main__":
    app.run(debug=True)