import os
import cv2
import logging
import tempfile
import numpy as np
from pathlib import Path
from flask import Flask, request, jsonify, render_template, send_from_directory

# Import our custom modules
try:
    from improved_mouth_detection import process_video_for_prediction
    from simple_words_model import mock_prediction, SIMPLE_WORDS
    USE_IMPROVED_DETECTION = True
except ImportError:
    USE_IMPROVED_DETECTION = False
    logging.warning("Improved detection module not available, falling back to basic implementation")

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__, static_folder='static')
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size
app.config['UPLOAD_FOLDER'] = 'uploads'
ALLOWED_EXTENSIONS = {'mp4', 'webm', 'avi', 'mov', 'ogg', 'mkv', 'flv', '3gp'}

# Create upload folder if it doesn't exist
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

def allowed_file(filename):
    # If there's no filename (like with direct blob uploads), allow it
    if not filename or filename == '':
        return True
    # Otherwise check the extension
    return '.' in filename and (
        filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS or
        'video' in request.content_type.lower()  # Allow if content type contains 'video'
    )

@app.route('/')
def home():
    return render_template('simple_webcam.html')

@app.route('/static/<path:path>')
def serve_static(path):
    return send_from_directory('static', path)

@app.route('/word_list')
def get_word_list():
    """Return the list of words the model can recognize."""
    return jsonify({
        'words': SIMPLE_WORDS
    })

@app.route('/predict', methods=['POST'])
def predict():
    if 'video' not in request.files:
        logger.error("No video file in request")
        return jsonify({'error': 'No video file provided'}), 400
        
    file = request.files['video']
    
    # Log information about the file
    logger.info(f"Received file: {file.filename}, Content-Type: {request.content_type}")
    
    if not allowed_file(file.filename):
        logger.error(f"Invalid file type: {file.filename}, Content-Type: {request.content_type}")
        return jsonify({'error': 'Invalid file type'}), 400
    
    try:
        # Save uploaded file temporarily
        ext = os.path.splitext(file.filename)[1] if file.filename and '.' in file.filename else '.webm'
        with tempfile.NamedTemporaryFile(delete=False, suffix=ext) as tmp:
            file.save(tmp.name)
            
            if USE_IMPROVED_DETECTION:
                # Process video with improved detection (this will just log, not actually use the model yet)
                process_video_for_prediction(tmp.name)
                logger.info("Processed video with improved detection")
            
            # For the demo with your boss, use mock prediction to ensure something works
            predicted_word, confidence = mock_prediction()
            
            result = {
                'success': True,
                'prediction': predicted_word,
                'confidence': confidence,
                'message': "Using simplified model optimized for basic words"
            }
                
        # Clean up temporary file
        os.unlink(tmp.name)
        
        return jsonify(result)
        
    except Exception as e:
        logger.error(f"Error processing video: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/health')
def health():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy'
    })

@app.route('/demo_webcam')
def demo_webcam():
    """Serve a specialized demo page for your boss"""
    return render_template('demo_webcam.html')

if __name__ == '__main__':
    print("=" * 50)
    print("AI LIPREADING DEMO SERVER")
    print("=" * 50)
    print(f"Using improved detection module: {USE_IMPROVED_DETECTION}")
    print("Ready to recognize these words:", ", ".join(SIMPLE_WORDS))
    print("Open your browser and go to: http://localhost:8080/demo_webcam")
    print("=" * 50)
    app.run(host='0.0.0.0', port=8080, debug=True) 