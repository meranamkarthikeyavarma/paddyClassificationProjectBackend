from flask import Flask, request, jsonify
from flask_cors import CORS
import keras
import cv2
import numpy as np
import os
import time
from werkzeug.utils import secure_filename
import gdown



app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Configuration
UPLOAD_FOLDER = 'temp_uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif', 'bmp'}
MAX_CONTENT_LENGTH = 16 * 1024 * 1024  # 16MB max file size

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = MAX_CONTENT_LENGTH

# Create upload directory if it doesn't exist
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Model configuration
file_id = os.environ.get("GDRIVE_FILE_ID")
GDRIVE_FILE_ID = file_id
MODEL_PATH = "BestModel.keras"

# -------------------- Download Model If Not Exists --------------------

if not os.path.exists(MODEL_PATH):
    print("Downloading model from Google Drive...")
    gdown.download(f"https://drive.google.com/uc?id={GDRIVE_FILE_ID}", MODEL_PATH, quiet=False)

# -------------------- Load Model --------------------

try:
    model = keras.models.load_model(MODEL_PATH)
    print("✅ Model loaded successfully!")
    model.summary()
except Exception as e:
    print(f"❌ Error loading model: {e}")
    model = None

IMAGE_SIZE = (150, 150)
CLASSES = ['AkhuRush', 'BalaDubraj', 'DesiShiv', 'Giyos', 'JALAKA',
           'Kiraat', 'kumkumsali', 'RadhaPagal', 'samuralahari', 'SannaJajulu',
           'UjalaManipal']

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def preprocess_image(image_path):
    """
    Preprocess the uploaded image for prediction
    """
    try:
        image = cv2.imread(image_path)
        
        if image is None:
            raise ValueError(f"Failed to load image at path: {image_path}")
        
        # Resize to model's expected input size
        image = cv2.resize(image, IMAGE_SIZE)
        
        # Normalize pixel values to [0, 1]
        image = image.astype('float32') / 255.0
        
        # Add batch dimension: (1, 150, 150, 3)
        image = np.expand_dims(image, axis=0)
        
        return image
    except Exception as e:
        raise ValueError(f"Error preprocessing image: {str(e)}")

@app.route('/predict', methods=['POST'])
def predict():
    """
    Handle image upload and return prediction
    """
    start_time = time.time()
    
    try:
        # Check if model is loaded
        if model is None:
            return jsonify({'error': 'Model not loaded properly'}), 500
        
        # Check if file is in request
        if 'image' not in request.files:
            return jsonify({'error': 'No image file provided'}), 400
        
        file = request.files['image']
        
        # Check if file is selected
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400
        
        # Check if file type is allowed
        if not allowed_file(file.filename):
            return jsonify({'error': 'File type not allowed. Please upload an image file.'}), 400
        
        # Save the uploaded file
        filename = secure_filename(file.filename)
        timestamp = str(int(time.time()))
        filename = f"{timestamp}_{filename}"
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        try:
            # Preprocess the image
            processed_image = preprocess_image(filepath)
            
            # Make prediction
            predictions = model.predict(processed_image)
            
            # Get predicted class index and confidence
            predicted_class_index = np.argmax(predictions)
            confidence = float(predictions[0][predicted_class_index])

            confidence_threshold = 0.9
            if confidence >= confidence_threshold:
                 predicted_class = CLASSES[predicted_class_index]
            else:
                predicted_class = "Unknown"
            
            
            # Calculate processing time
            processing_time = f"{time.time() - start_time:.2f}s"
            
            # Prepare response
            response = {
                'predicted_class': predicted_class,
                'confidence': confidence,
                'processing_time': processing_time,
                'all_predictions': {
                    CLASSES[i]: float(predictions[0][i]) 
                    for i in range(len(CLASSES))
                }
            }
            
            return jsonify(response)
            
        except Exception as e:
            print("❌ Error processing image:", str(e), flush=True)
            return jsonify({'error': f'Error processing image: {str(e)}'}), 500
            
        finally:
            # Clean up: remove the temporary file
            try:
                if os.path.exists(filepath):
                    os.remove(filepath)
            except:
                pass
    
    except Exception as e:
        print("❌ Error in /predict:", str(e), flush=True)
        return jsonify({'error': f'Unexpected error: {str(e)}'}), 500

@app.route('/health', methods=['GET'])
def health_check():
    """
    Health check endpoint
    """
    return jsonify({
        'status': 'healthy',
        'model_loaded': model is not None,
        'classes': CLASSES
    })

@app.route('/', methods=['GET'])
def index():
    """
    Root endpoint
    """
    return jsonify({
        'message': 'Paddy Classification API',
        'version': '1.0',
        'endpoints': {
            '/predict': 'POST - Upload image for classification',
            '/health': 'GET - Check API health'
        }
    })

if __name__ == '__main__':
    print("Starting Paddy Classification API...")
    print(f"Model loaded: {model is not None}")
    print(f"Classes: {CLASSES}")
    print("Server running on http://localhost:5000")
    # app.run(debug=True, host='0.0.0.0', port=5000)
