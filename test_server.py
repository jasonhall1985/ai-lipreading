from flask import Flask, render_template, request, jsonify
import os
import random
import time

app = Flask(__name__, 
            template_folder='templates',  # Point to the template folder
            static_folder='static')       # Point to the static folder

@app.route('/')
def hello():
    return render_template('avhubert_webcam.html')

@app.route('/test')
def test():
    return "Hello, World! Flask is working!"

@app.route('/predict', methods=['POST'])
def predict():
    # Simulate processing a video by waiting for a bit
    time.sleep(1.5)
    
    # Generate random prediction
    possible_words = ["hello", "thank you", "goodbye", "please", "how are you", 
                     "what", "when", "where", "why", "who", "yes", "no"]
    prediction = random.choice(possible_words)
    confidence = random.uniform(0.7, 0.95)
    
    return jsonify({
        'success': True,
        'prediction': prediction,
        'confidence': confidence
    })

if __name__ == '__main__':
    print("Starting webcam demo server on port 3000...")
    app.run(host='127.0.0.1', port=3000, debug=True) 