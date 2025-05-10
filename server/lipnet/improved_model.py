import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Input, Conv3D, MaxPooling3D, BatchNormalization, Dropout, 
    Bidirectional, LSTM, Dense, Reshape, TimeDistributed, SpatialDropout3D,
    GRU, Activation, LeakyReLU, Add, Attention, LayerNormalization, MultiHeadAttention
)
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau, TensorBoard
from tensorflow.keras.regularizers import l2
from tensorflow.keras.applications import ResNet50

# Character set
CHAR_LIST = " abcdefghijklmnopqrstuvwxyz'?!"
CHAR_TO_NUM = {char: i for i, char in enumerate(CHAR_LIST)}
NUM_TO_CHAR = {i: char for i, char in enumerate(CHAR_LIST)}

def ctc_loss_fn():
    """CTC loss function for lipreading"""
    def loss(y_true, y_pred):
        batch_len = tf.cast(tf.shape(y_true)[0], dtype="int64")
        input_length = tf.cast(tf.shape(y_pred)[1], dtype="int64")
        label_length = tf.cast(tf.shape(y_true)[1], dtype="int64")

        input_length = input_length * tf.ones(shape=(batch_len, 1), dtype="int64")
        label_length = label_length * tf.ones(shape=(batch_len, 1), dtype="int64")

        return tf.keras.backend.ctc_batch_cost(y_true, y_pred, input_length, label_length)
    return loss

def residual_block(x, filters, kernel_size=(3, 3, 3)):
    """Residual block for 3D CNNs"""
    shortcut = x
    
    # First convolution layer
    x = Conv3D(filters, kernel_size, padding='same', kernel_regularizer=l2(1e-4))(x)
    x = BatchNormalization()(x)
    x = LeakyReLU(alpha=0.2)(x)
    
    # Second convolution layer
    x = Conv3D(filters, kernel_size, padding='same', kernel_regularizer=l2(1e-4))(x)
    x = BatchNormalization()(x)
    
    # If input and output shapes don't match, use 1x1 conv to match dimensions
    if shortcut.shape[-1] != filters:
        shortcut = Conv3D(filters, (1, 1, 1), padding='same')(shortcut)
        shortcut = BatchNormalization()(shortcut)
    
    # Add shortcut connection
    x = Add()([x, shortcut])
    x = LeakyReLU(alpha=0.2)(x)
    
    return x

def build_improved_lipnet(img_size=(50, 100), max_len=75, dropout_rate=0.3):
    """
    Build an improved LipNet model with residual connections, spatial dropout,
    and other enhancements for better accuracy.
    
    Args:
        img_size: Tuple of (height, width) for input images
        max_len: Maximum sequence length
        dropout_rate: Dropout rate for regularization
    
    Returns:
        Keras model
    """
    print("Building improved LipNet model...")
    input_shape = (max_len,) + img_size + (1,)
    input_layer = Input(shape=input_shape)
    
    # Initial 3D convolution
    x = Conv3D(32, (3, 3, 3), padding='same', kernel_regularizer=l2(1e-4))(input_layer)
    x = BatchNormalization()(x)
    x = LeakyReLU(alpha=0.2)(x)
    x = MaxPooling3D(pool_size=(1, 2, 2))(x)
    
    # First residual block
    x = residual_block(x, 64)
    x = SpatialDropout3D(dropout_rate)(x)
    x = MaxPooling3D(pool_size=(1, 2, 2))(x)
    
    # Second residual block
    x = residual_block(x, 128)
    x = SpatialDropout3D(dropout_rate)(x)
    x = MaxPooling3D(pool_size=(1, 2, 2))(x)
    
    # Reshape for RNN layers
    _, seq_len, h, w, c = x.shape
    x = Reshape((seq_len, h * w * c))(x)
    
    # RNN layers with highway connections
    x = Bidirectional(GRU(256, return_sequences=True, dropout=dropout_rate, recurrent_dropout=dropout_rate))(x)
    x = Bidirectional(GRU(256, return_sequences=True, dropout=dropout_rate, recurrent_dropout=dropout_rate))(x)
    
    # Output layer
    output = Dense(len(CHAR_LIST) + 1, activation='softmax')(x)
    
    model = Model(inputs=input_layer, outputs=output)
    print("Improved model built successfully")
    model.summary()
    
    return model

def compile_model(model, learning_rate=0.0003):
    """Compile the model with appropriate optimizer and loss function"""
    optimizer = Adam(learning_rate=learning_rate)
    model.compile(optimizer=optimizer, loss=ctc_loss_fn())
    return model

def get_callbacks(checkpoint_path, patience=10):
    """Get callbacks for training the model"""
    callbacks = [
        ModelCheckpoint(
            checkpoint_path,
            monitor='val_loss',
            save_best_only=True,
            save_weights_only=True,
            verbose=1
        ),
        EarlyStopping(
            monitor='val_loss',
            patience=patience,
            restore_best_weights=True,
            verbose=1
        ),
        ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=5,
            min_lr=1e-8,
            verbose=1
        ),
        TensorBoard(
            log_dir='./logs',
            histogram_freq=1,
            write_graph=True
        )
    ]
    return callbacks

def preprocess_for_training(video_array, augment=True):
    """
    Preprocess video array for training with optional augmentation
    
    Args:
        video_array: Numpy array of shape (frames, height, width)
        augment: Whether to apply data augmentation
    
    Returns:
        Preprocessed array of shape (frames, height, width, 1)
    """
    # Normalize
    video_array = video_array.astype(np.float32) / 255.0
    
    if augment:
        # Random brightness adjustment
        brightness_factor = np.random.uniform(0.8, 1.2)
        video_array = video_array * brightness_factor
        video_array = np.clip(video_array, 0, 1)
        
        # Random contrast adjustment
        contrast_factor = np.random.uniform(0.8, 1.2)
        mean = np.mean(video_array)
        video_array = (video_array - mean) * contrast_factor + mean
        video_array = np.clip(video_array, 0, 1)
    
    # Add channel dimension
    video_array = np.expand_dims(video_array, -1)
    
    return video_array

def decode_prediction(prediction):
    """
    Decode model prediction using CTC decoding
    
    Args:
        prediction: Model prediction array
    
    Returns:
        Decoded text
    """
    try:
        # Check prediction shape and reshape if needed
        if len(tf.shape(prediction)) == 2:
            # If prediction is [time_steps, num_classes]
            prediction = tf.expand_dims(prediction, 0)  # Add batch dimension
        
        # Use CTC greedy decoder
        indices = tf.argmax(prediction, axis=-1)
        # Convert to dense tensor
        indices = tf.cast(indices, dtype=tf.int32)
        
        # Handle single dimension case
        if len(indices.shape) == 1:
            # If we only have a 1D tensor, reshape to [1, time_steps]
            indices = tf.expand_dims(indices, 0)
            
        # CTC decoding - merge repeated characters and remove blanks
        # Safely handle indices
        if indices.shape[1] > 1:  # Only do this if we have more than one timestep
            prev_indices = tf.pad(indices[:, :-1], [[0, 0], [1, 0]], constant_values=-1)
            repeated = tf.equal(indices, prev_indices)
        else:
            # If only one timestep, nothing can be repeated
            repeated = tf.zeros_like(indices, dtype=tf.bool)
            
        blank = tf.equal(indices, 0)  # Assuming 0 is blank token
        mask = tf.logical_or(blank, repeated)
        
        # Handle case where all predictions might be masked
        if tf.reduce_all(mask):
            return ""  # Return empty string if all predictions are masked
            
        filtered_indices = tf.boolean_mask(indices, tf.logical_not(mask))
        
        # Convert to string
        chars = [NUM_TO_CHAR.get(idx.numpy(), "") for idx in filtered_indices]
        return "".join(chars).strip()
    except Exception as e:
        print(f"Error in decode_prediction: {str(e)}")
        # Fallback to a simpler decoding method
        try:
            # Just return the most likely character for each time step
            best_chars = tf.argmax(prediction, axis=-1)
            best_chars = best_chars.numpy().flatten()
            # Remove zeros (blanks) and convert to characters
            result = "".join([NUM_TO_CHAR.get(idx, "") for idx in best_chars if idx > 0])
            return result
        except:
            return "Unable to decode prediction"

class LipReadingModel:
    def __init__(self, input_shape=(75, 50, 3), num_classes=28):
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.model = self.build_model()
        
    def build_model(self):
        # Input layer
        input_layer = layers.Input(shape=self.input_shape)
        
        # Preprocessing layers
        x = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(input_layer)
        x = layers.BatchNormalization()(x)
        x = layers.MaxPooling2D((2, 2))(x)
        
        # ResNet50 backbone for feature extraction
        resnet = ResNet50(weights='imagenet', include_top=False, input_shape=(None, None, 3))
        resnet.trainable = False
        x = resnet(x)
        
        # Temporal processing
        x = layers.Reshape((-1, x.shape[-1]))(x)
        
        # Multi-head attention
        x = MultiHeadAttention(num_heads=8, key_dim=64)(x, x)
        x = LayerNormalization()(x)
        
        # Bidirectional LSTM
        x = layers.Bidirectional(layers.LSTM(128, return_sequences=True))(x)
        x = layers.Bidirectional(layers.LSTM(128, return_sequences=True))(x)
        
        # Output layer
        x = layers.Dense(256, activation='relu')(x)
        x = layers.Dropout(0.5)(x)
        output = layers.Dense(self.num_classes, activation='softmax')(x)
        
        model = models.Model(inputs=input_layer, outputs=output)
        return model
    
    def compile_model(self, learning_rate=0.001):
        self.model.compile(
            optimizer=optimizers.Adam(learning_rate=learning_rate),
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
    
    def preprocess_video(self, video_frames):
        # Enhanced preprocessing
        processed_frames = []
        for frame in video_frames:
            # Face detection and alignment
            face = self.detect_and_align_face(frame)
            if face is not None:
                # Mouth region extraction
                mouth = self.extract_mouth_region(face)
                # Normalization
                mouth = self.normalize_mouth(mouth)
                processed_frames.append(mouth)
        
        # Temporal alignment
        processed_frames = self.temporal_align(processed_frames)
        
        return np.array(processed_frames)
    
    def detect_and_align_face(self, frame):
        # Implement face detection and alignment
        # This would use a face detection model like MTCNN or Dlib
        pass
    
    def extract_mouth_region(self, face):
        # Extract and normalize mouth region
        pass
    
    def normalize_mouth(self, mouth):
        # Normalize mouth region
        pass
    
    def temporal_align(self, frames):
        # Align frames temporally
        pass 