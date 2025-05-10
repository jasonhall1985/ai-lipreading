import os
import cv2
import torch
import logging
import tempfile
import numpy as np
from pathlib import Path
from flask import Flask, request, jsonify, render_template
from werkzeug.utils import secure_filename

# Import the AV-HuBERT model implementation
from avhubert_model import create_avhubert_model

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

print(">>> Loading AV-HuBERT model...")  # DEBUG print

app = Flask(__name__, 
            template_folder='../../templates',  # Point to the template folder
            static_folder='../../static')        # Point to the static folder
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size
app.config['UPLOAD_FOLDER'] = 'uploads'
ALLOWED_EXTENSIONS = {'mp4', 'webm', 'avi'}

# Initialize AV-HuBERT model
try:
    model = create_avhubert_model(
        model_path=os.getenv('AVHUBERT_MODEL_PATH', './avhubert_base_ls960.pt'),
        config_path=os.getenv('AVHUBERT_CONFIG_PATH', './avhubert_base_ls960.yaml')
    )
    logger.info("AV-HuBERT model loaded successfully")
except Exception as e:
    logger.error(f"Failed to load AV-HuBERT model: {str(e)}")
    model = None

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def process_video(video_path, target_fps=25):
    cap = cv2.VideoCapture(video_path)
    frames = []

    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_interval = int(round(fps / target_fps)) if fps else 1
    frame_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_count % frame_interval == 0:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(frame)

        frame_count += 1

    cap.release()
    return np.array(frames)

@app.route('/')
def home():
    return render_template('avhubert_webcam.html')

@app.route('/test')
def test():
    return render_template('test.html')

@app.route('/raw')
def raw():
    return """
    <!DOCTYPE html>
    <html>
    <head>
        <title>Raw Test Page</title>
    </head>
    <body>
        <h1>Raw HTML Test</h1>
        <p>This is a direct HTML response from the server without using templates.</p>
    </body>
    </html>
    """

@app.route('/predict', methods=['POST'])
def predict():
    if 'video' not in request.files:
        return jsonify({'error': 'No video file provided'}), 400

    file = request.files['video']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    if not allowed_file(file.filename):
        return jsonify({'error': 'Invalid file type'}), 400

    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(file.filename)[1]) as tmp:
            file.save(tmp.name)
            frames = process_video(tmp.name)

            if model:
                prediction, confidence = model.predict_from_video(
                    frames,
                    augment=True,
                    num_samples=3
                )
                result = {
                    'success': True,
                    'prediction': prediction,
                    'confidence': confidence
                }
            else:
                result = {
                    'success': False,
                    'error': 'Model not loaded'
                }

        os.unlink(tmp.name)
        return jsonify(result)

    except Exception as e:
        logger.error(f"Error processing video: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/health')
def health():
    return jsonify({
        'status': 'healthy',
        'model_loaded': model is not None,
        'demo_mode': True,
        'gpu_available': torch.cuda.is_available()
    })

def main():
    print(">>> Starting Flask server...")  # DEBUG print
    os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
    app.run(host='127.0.0.1', port=3000, debug=True)

if __name__ == '__main__':
    main()
