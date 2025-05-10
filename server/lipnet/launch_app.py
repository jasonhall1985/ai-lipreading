import os
import sys
import webbrowser
import threading
import time
import subprocess

def open_browser():
    """Open browser after a short delay"""
    time.sleep(2)  # Wait for Flask to start
    webbrowser.open('http://localhost:5000')

def main():
    # Start browser in a separate thread
    threading.Thread(target=open_browser).start()
    
    # Get the path to the activated Python executable
    python_path = sys.executable
    
    # Get the path to lip_demo.py
    current_dir = os.path.dirname(os.path.abspath(__file__))
    app_path = os.path.join(current_dir, 'lip_demo.py')
    
    print("Starting LipNet Demo web application...")
    print("Browser will open automatically...")
    
    # Run the Flask app using the activated Python
    subprocess.run([python_path, app_path])

if __name__ == "__main__":
    main() 