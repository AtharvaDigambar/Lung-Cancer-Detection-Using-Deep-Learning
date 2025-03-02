from flask import Flask, request, jsonify, render_template, send_from_directory
from flask_cors import CORS
import tensorflow as tf
import os
import logging
import cv2
import numpy as np
import base64
from io import BytesIO
from PIL import Image

# Initialize Flask App
app = Flask(__name__, static_folder="static", template_folder="templates")
CORS(app)

# Define Upload Folder
UPLOAD_FOLDER = "/tmp/uploads"
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Logging Setup
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# Model Path
MODEL_PATH = "final_cancer_model.h5"

# Check if model exists before loading
if not os.path.exists(MODEL_PATH):
    logger.error(f"Model file {MODEL_PATH} not found.")
    raise FileNotFoundError(f"Model file {MODEL_PATH} not found.")

try:
    model = tf.keras.models.load_model(MODEL_PATH)
    logger.info("Model loaded successfully")
except Exception as e:
    logger.error(f"Failed to load model: {str(e)}")
    raise

# Image Preprocessing Function
def preprocess_image(image_path):
    try:
        img = cv2.imread(image_path, cv2.IMREAD_COLOR)
        if img is None:
            raise ValueError("Invalid image file")
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (224, 224))
        img = np.expand_dims(img, axis=0) / 255.0
        return img
    except Exception as e:
        logger.error(f"Error in image preprocessing: {str(e)}")
        raise

# Generate Heatmap
def generate_heatmap(model, img_array, class_idx):
    try:
        last_conv_layer = next(layer for layer in reversed(model.layers) if 'conv' in layer.name.lower())
        grad_model = tf.keras.models.Model([model.inputs], [last_conv_layer.output, model.output])

        with tf.GradientTape() as tape:
            conv_outputs, predictions = grad_model(img_array)
            loss = predictions[:, class_idx]

        grads = tape.gradient(loss, conv_outputs)[0]
        pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
        conv_outputs = conv_outputs[0]

        heatmap = tf.reduce_mean(conv_outputs * pooled_grads, axis=-1)

        # Fix potential division by zero issue
        heatmap = np.maximum(heatmap, 0)
        if np.max(heatmap) == 0:
            logger.warning("Heatmap normalization issue: Max value is 0")
            return None  # Avoid NaN issues

        heatmap /= np.max(heatmap)  # Normalize between 0 and 1
        heatmap = cv2.resize(heatmap, (224, 224))
        return heatmap
    except Exception as e:
        logger.error(f"Error in heatmap generation: {str(e)}")
        return None

# Overlay Heatmap on Original Image
def overlay_heatmap(original_img, heatmap):
    try:
        heatmap = np.uint8(255 * heatmap)
        heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
        original_img = np.uint8(original_img[0] * 255)
        superimposed_img = cv2.addWeighted(original_img, 0.6, heatmap, 0.4, 0)

        img_pil = Image.fromarray(superimposed_img)
        buffered = BytesIO()
        img_pil.save(buffered, format="PNG")
        img_base64 = base64.b64encode(buffered.getvalue()).decode('utf-8')
        return f"data:image/png;base64,{img_base64}"
    except Exception as e:
        logger.error(f"Error overlaying heatmap: {str(e)}")
        return None

# Serve the Main Webpage
@app.route("/")
def home():
    return render_template("index.html")

# Route for Demo Page
@app.route("/demo")
def demo():
    return render_template("demo.html")

# Serve Static Files
@app.route('/static/<path:filename>')
def serve_static(filename):
    return send_from_directory("static", filename)

# Prediction Endpoint
@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return jsonify({'error': 'No image uploaded'}), 400

    file = request.files['image']
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400

    file_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
    
    try:
        file.save(file_path)
        logger.info(f"Image saved at {file_path}")

        img = preprocess_image(file_path)
        prediction = model.predict(img, verbose=0)

        if prediction.shape[-1] != 3:
            raise ValueError(f"Expected 3 classes, got shape {prediction.shape}")

        probabilities = prediction[0]
        
        # Fix NaN issue: Ensure valid probabilities

        class_probs = {
            'Benign': float(probabilities[0]),
            'Malignant': float(probabilities[1]),
            'Normal': float(probabilities[2])
        }

        predicted_class = max(class_probs, key=class_probs.get)
        confidence = max(class_probs.values())  # Ensure confidence is a real number

        if confidence == 0:
            logger.warning("Confidence score is 0, possible model issue.")
            confidence = 1e-6  # Avoid zero probability

        class_idx = list(class_probs.keys()).index(predicted_class)

        heatmap = generate_heatmap(model, img, class_idx)
        heatmap_base64 = overlay_heatmap(img, heatmap) if heatmap is not None else None

        result = {
            'predicted_class': predicted_class,
            'probability': confidence *100,
            'confidence': round(confidence * 100, 2),  # Round to 2 decimal places
            'heatmap': heatmap_base64,
            'details': {k: round(v * 100, 2) for k, v in class_probs.items()}
        }
        return jsonify(result)

    except Exception as e:
        logger.error(f"Prediction error: {str(e)}")
        return jsonify({'error': str(e)}), 500

    finally:
        if os.path.exists(file_path):
            os.remove(file_path)
            logger.info(f"Deleted temporary file: {file_path}")

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
