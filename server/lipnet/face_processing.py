import cv2
import numpy as np
import mediapipe as mp
import logging
from typing import Tuple, Optional, List

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class FaceProcessor:
    def __init__(self, static_image_mode=False, max_num_faces=1):
        """
        Initialize face processor with MediaPipe Face Mesh
        Args:
            static_image_mode: Whether to treat input images as static (not video)
            max_num_faces: Maximum number of faces to detect
        """
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            static_image_mode=static_image_mode,
            max_num_faces=max_num_faces,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        
        # Mouth landmark indices
        self.MOUTH_OUTLINE = [
            61, 146, 91, 181, 84, 17, 314, 405, 321, 375,
            291, 308, 324, 318, 402, 317, 14, 87, 178, 88,
            95, 78, 62
        ]
        
        # Additional landmarks for better mouth region
        self.MOUTH_INNER = [
            78, 95, 88, 178, 87, 14, 317, 402, 318, 324,
            308, 291, 375, 321, 405, 314, 17, 84, 181, 91,
            146, 61, 62
        ]
    
    def get_mouth_region(self, 
                        frame: np.ndarray, 
                        padding: float = 0.5
                        ) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """
        Extract mouth region from frame with landmarks
        Args:
            frame: Input frame
            padding: Padding around mouth region as percentage of mouth height
        Returns:
            mouth_frame: Extracted mouth region
            landmarks: Mouth landmarks
        """
        try:
            # Convert to RGB for MediaPipe
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self.face_mesh.process(frame_rgb)
            
            if not results.multi_face_landmarks:
                return None, None
            
            face_landmarks = results.multi_face_landmarks[0]
            
            # Get mouth landmarks
            mouth_points = []
            for idx in self.MOUTH_OUTLINE:
                landmark = face_landmarks.landmark[idx]
                x, y = int(landmark.x * frame.shape[1]), int(landmark.y * frame.shape[0])
                mouth_points.append((x, y))
            
            # Calculate bounding box with padding
            x_coords = [p[0] for p in mouth_points]
            y_coords = [p[1] for p in mouth_points]
            
            left = min(x_coords)
            right = max(x_coords)
            top = min(y_coords)
            bottom = max(y_coords)
            
            # Add padding
            mouth_height = bottom - top
            mouth_width = right - left
            
            pad_h = int(mouth_height * padding)
            pad_w = int(mouth_width * padding)
            
            top = max(0, top - pad_h)
            bottom = min(frame.shape[0], bottom + pad_h)
            left = max(0, left - pad_w)
            right = min(frame.shape[1], right + pad_w)
            
            # Extract mouth region
            mouth_frame = frame[top:bottom, left:right]
            
            # Adjust landmark coordinates relative to mouth region
            mouth_landmarks = np.array(mouth_points) - np.array([left, top])
            
            return mouth_frame, mouth_landmarks
            
        except Exception as e:
            logger.error(f"Error in mouth region extraction: {str(e)}")
            return None, None
    
    def process_video_frames(self, 
                           frames: np.ndarray, 
                           target_size: Tuple[int, int] = (96, 96),
                           use_landmarks: bool = True
                           ) -> Tuple[Optional[np.ndarray], Optional[List[np.ndarray]]]:
        """
        Process video frames to extract mouth regions
        Args:
            frames: Input video frames (T, H, W, C)
            target_size: Target size for mouth regions
            use_landmarks: Whether to return landmarks
        Returns:
            processed_frames: Processed mouth regions
            landmarks_sequence: Sequence of landmarks (if use_landmarks=True)
        """
        try:
            processed_frames = []
            landmarks_sequence = [] if use_landmarks else None
            
            for frame in frames:
                mouth_region, landmarks = self.get_mouth_region(frame)
                
                if mouth_region is None:
                    logger.warning("Failed to detect mouth in frame")
                    continue
                
                # Resize mouth region
                mouth_region = cv2.resize(mouth_region, target_size)
                
                processed_frames.append(mouth_region)
                if use_landmarks:
                    landmarks_sequence.append(landmarks)
            
            if not processed_frames:
                logger.error("No valid frames processed")
                return None, None
            
            processed_frames = np.stack(processed_frames)
            
            return processed_frames, landmarks_sequence
            
        except Exception as e:
            logger.error(f"Error processing video frames: {str(e)}")
            return None, None
    
    def apply_augmentation(self, 
                         frames: np.ndarray,
                         landmarks: Optional[List[np.ndarray]] = None
                         ) -> Tuple[np.ndarray, Optional[List[np.ndarray]]]:
        """
        Apply data augmentation to frames and landmarks
        Args:
            frames: Input frames (T, H, W, C)
            landmarks: Optional landmarks
        Returns:
            augmented_frames: Augmented frames
            augmented_landmarks: Augmented landmarks (if provided)
        """
        try:
            # Random brightness adjustment
            brightness = np.random.uniform(0.8, 1.2)
            frames = np.clip(frames * brightness, 0, 255).astype(np.uint8)
            
            # Random contrast adjustment
            contrast = np.random.uniform(0.8, 1.2)
            frames = np.clip(128 + contrast * (frames - 128), 0, 255).astype(np.uint8)
            
            # Random horizontal flip
            if np.random.random() < 0.5:
                frames = frames[:, :, ::-1, :]
                if landmarks is not None:
                    for i, lm in enumerate(landmarks):
                        landmarks[i][:, 0] = frames.shape[2] - lm[:, 0]
            
            # Random rotation
            angle = np.random.uniform(-15, 15)
            center = (frames.shape[2] // 2, frames.shape[1] // 2)
            M = cv2.getRotationMatrix2D(center, angle, 1.0)
            
            for i in range(len(frames)):
                frames[i] = cv2.warpAffine(frames[i], M, (frames.shape[2], frames.shape[1]))
                if landmarks is not None:
                    # Rotate landmarks
                    ones = np.ones(shape=(len(landmarks[i]), 1))
                    points_ones = np.hstack([landmarks[i], ones])
                    landmarks[i] = np.dot(M, points_ones.T).T
            
            return frames, landmarks
            
        except Exception as e:
            logger.error(f"Error applying augmentation: {str(e)}")
            return frames, landmarks 