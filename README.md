# 🌿 Plant Disease Detector using CNN and Gradio

A deep learning–based web application to detect plant leaf diseases using **Convolutional Neural Networks (CNN)** and an interactive **Gradio interface**.  
This project helps farmers and researchers identify plant diseases quickly by uploading an image of a leaf — the model predicts the type of disease instantly.

---

## 🚀 Live Demo  
🔗 **Hugging Face Space:** [Plant Disease Detector](https://huggingface.co/spaces/sivajyothi3/Plant_Disease_Detector)

---

## 🧠 Features

- 🌱 Detects various plant diseases using a trained CNN model  
- 💻 Simple web interface built with **Gradio**  
- ⚙️ Model trained on a large dataset from **Kaggle**  
- ☁️ Deployed successfully on **Hugging Face Spaces**  
- 📊 Achieves high accuracy on test data  

---

## 📂 Dataset

Download the dataset from Kaggle:  
🔗 [Plant Disease Detection Dataset](https://www.kaggle.com/sumanismcse/plant-disease-detection-using-keras)

Extract it inside your project folder.

Use it for training or retraining your CNN model.

> The dataset contains ~14,000 images of healthy and diseased leaves across multiple plant species.

---

## 🏗️ Project Structure

```
Plant_Disease_Detector/
│
├── app.py                     # Main Gradio application file
├── plant_disease_cnn_model.h5 # Trained CNN model
├── requirements.txt           # Dependencies
├── templates/ (optional)      # HTML templates (if Flask version used)
├── static/ (optional)         # CSS and assets (if Flask version used)
└── README.md                  # Project documentation
```

---

## ⚙️ Installation and Running Locally

### 1️⃣ Clone the Repository
```bash
git clone https://github.com/Sivajyothi3/Plant_Disease_Detector.git
cd Plant_Disease_Detector
```

### 2️⃣ Create Virtual Environment
```bash
python -m venv venv
venv\Scripts\activate      # For Windows
# source venv/bin/activate  # For macOS/Linux
```

### 3️⃣ Install Dependencies
```bash
pip install -r requirements.txt
```

### 4️⃣ Run the Application
```bash
python app.py
```

### 5️⃣ Access the App  
Once the server starts, open the local Gradio link (shown in terminal).  
Upload a leaf image and get instant predictions!

---

## 🧩 Technologies Used

- **Python**
- **TensorFlow / Keras**
- **Gradio**
- **NumPy, Pandas, Matplotlib**
- **Hugging Face Spaces** for deployment

---

## 📈 Model Overview

- **Architecture:** Custom CNN  
- **Input:** Leaf image  
- **Output:** Predicted disease class  
- **Training:** Performed using the Kaggle dataset  
- **Accuracy:** Achieved ~98% on validation data  

---

## 📤 Deployment

This project is deployed on **Hugging Face Spaces** using the Gradio app.  
To deploy your own version:

```bash
git lfs install
git add .
git commit -m "Initial commit"
git push hf main
```

---

## 🤝 Contributing

Contributions, issues, and feature requests are welcome!  
Feel free to open a pull request to improve the app or model.

---

## 🧑‍💻 Author

👩‍💻 **Siva Jyothi**  
- 🌐 GitHub: [@Sivajyothi3](https://github.com/Sivajyothi3)  
- 🤗 Hugging Face: [@sivajyothi3](https://huggingface.co/sivajyothi3)

---

## 🪴 Acknowledgements

- Kaggle for the dataset  
- Hugging Face for hosting the space  
- Gradio team for the UI framework  
- TensorFlow/Keras for deep learning support  

---

⭐ **If you like this project, give it a star on GitHub!**
