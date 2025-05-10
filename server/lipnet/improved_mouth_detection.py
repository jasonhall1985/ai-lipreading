import cv2
import numpy as np
import os
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def initialize_face_detector():
    """Initialize the face detector from OpenCV."""
    face_cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
    if not os.path.exists(face_cascade_path):
        logger.error(f"Face cascade file not found at {face_cascade_path}")
        return None
    return cv2.CascadeClassifier(face_cascade_path)

def detect_face(frame, face_cascade):
    """Detect faces in the given frame."""
    if face_cascade is None:
        return None
    
    # Convert to grayscale for detection
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Detect faces
    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(30, 30),
        flags=cv2.CASCADE_SCALE_IMAGE
    )
    
    if len(faces) == 0:
        return None
    
    # Return the largest face
    faces = sorted(faces, key=lambda x: x[2] * x[3], reverse=True)
    return faces[0]

def extract_mouth_region(face_img, target_size=(100, 50)):
    """Extract the mouth region from a face image."""
    if face_img is None or face_img.size == 0:
        return None
    
    height, width = face_img.shape[:2]
    
    # Define the mouth region as the lower third of the face
    mouth_top = int(height * 0.65)
    mouth_bottom = int(height * 0.95)
    mouth_left = int(width * 0.25)
    mouth_right = int(width * 0.75)
    
    # Extract the mouth region
    mouth_region = face_img[mouth_top:mouth_bottom, mouth_left:mouth_right]
    
    if mouth_region.size == 0:
        return None
    
    # Resize to target size
    mouth_region = cv2.resize(mouth_region, target_size)
    
    return mouth_region

def preprocess_video_frame(frame, face_cascade, target_size=(100, 50)):
    """Process a video frame for lipreading."""
    if frame is None:
        return None
    
    # Detect face
    face_rect = detect_face(frame, face_cascade)
    if face_rect is None:
        return None
    
    # Extract face region
    x, y, w, h = face_rect
    face_img = frame[y:y+h, x:x+w]
    
    # Extract mouth region
    mouth_region = extract_mouth_region(face_img, target_size)
    if mouth_region is None:
        return None
    
    # Convert to grayscale
    gray_mouth = cv2.cvtColor(mouth_region, cv2.COLOR_BGR2GRAY)
    
    # Normalize pixel values
    normalized_mouth = gray_mouth / 255.0
    
    return normalized_mouth

def process_video_for_prediction(video_path, face_cascade=None, max_frames=75, target_size=(100, 50)):
    """Process a video file for lipreading prediction."""
    if face_cascade is None:
        face_cascade = initialize_face_detector()
        if face_cascade is None:
            return None
    
    # Open the video file
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        logger.error(f"Failed to open video file: {video_path}")
        return None
    
    frames = []
    frame_count = 0
    
    # Read frames from the video
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Process the frame
        processed_frame = preprocess_video_frame(frame, face_cascade, target_size)
        if processed_frame is not None:
            frames.append(processed_frame)
        
        frame_count += 1
        if frame_count >= max_frames:
            break
    
    cap.release()
    
    if len(frames) == 0:
        logger.error("No valid frames were processed from the video")
        return None
    
    # Pad or truncate to max_frames
    if len(frames) < max_frames:
        # Pad with the last frame
        last_frame = frames[-1]
        frames.extend([last_frame] * (max_frames - len(frames)))
    elif len(frames) > max_frames:
        frames = frames[:max_frames]
    
    # Stack frames into a numpy array
    frames_array = np.stack(frames, axis=0)
    
    # Add channel dimension for model input
    frames_array = np.expand_dims(frames_array, axis=-1)
    
    return frames_array

def test_mouth_detection(video_path, output_path=None):
    """Test mouth detection on a video file and optionally save the result."""
    face_cascade = initialize_face_detector()
    if face_cascade is None:
        return False
    
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        logger.error(f"Failed to open video file: {video_path}")
        return False
    
    # Get video properties
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    # Create video writer if output path is specified
    if output_path is not None:
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))
    
    success = True
    frame_count = 0
    detected_count = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Detect face
        face_rect = detect_face(frame, face_cascade)
        frame_count += 1
        
        if face_rect is not None:
            detected_count += 1
            x, y, w, h = face_rect
            
            # Draw rectangle around face
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
            
            # Extract and process mouth region
            face_img = frame[y:y+h, x:x+w]
            mouth_region = extract_mouth_region(face_img)
            
            if mouth_region is not None:
                # Calculate mouth position in original frame
                height, width = face_img.shape[:2]
                mouth_top = int(height * 0.65)
                mouth_bottom = int(height * 0.95)
                mouth_left = int(width * 0.25)
                mouth_right = int(width * 0.75)
                
                # Draw rectangle around mouth
                cv2.rectangle(frame, 
                             (x + mouth_left, y + mouth_top), 
                             (x + mouth_right, y + mouth_bottom), 
                             (0, 0, 255), 2)
        
        # Write the frame
        if output_path is not None:
            out.write(frame)
    
    # Release resources
    cap.release()
    if output_path is not None:
        out.release()
    
    detection_rate = detected_count / frame_count if frame_count > 0 else 0
    logger.info(f"Face detection rate: {detection_rate:.2f} ({detected_count}/{frame_count} frames)")
    
    return detection_rate > 0.5  # Success if at least 50% of frames have detected faces

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python improved_mouth_detection.py <video_path> [output_path]")
        sys.exit(1)
    
    video_path = sys.argv[1]
    output_path = sys.argv[2] if len(sys.argv) > 2 else None
    
    result = test_mouth_detection(video_path, output_path)
    print("Detection test successful" if result else "Detection test failed") 