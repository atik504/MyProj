#!/usr/bin/env python3

import os
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Conv2D, Input, BatchNormalization, LeakyReLU, Add
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping
import matplotlib.pyplot as plt
import glob
from tqdm import tqdm
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Configuration
class Config:
    # Data settings
    DATASET_PATH = "dataset"  # Path to dataset folder
    TRAIN_DIR = os.path.join(DATASET_PATH, "train")
    VAL_DIR = os.path.join(DATASET_PATH, "val")
    PATCH_SIZE = 64  # Size of image patches for training
    SCALE_FACTOR = 2  # Upscaling factor
    
    # Training settings
    BATCH_SIZE = 16
    EPOCHS = 100
    LEARNING_RATE = 1e-4
    
    # Model settings
    MODEL_SAVE_DIR = "models"
    MODEL_CHECKPOINT = os.path.join(MODEL_SAVE_DIR, "srgan_model.h5")
    
    # Create necessary directories
    os.makedirs(DATASET_PATH, exist_ok=True)
    os.makedirs(TRAIN_DIR, exist_ok=True)
    os.makedirs(VAL_DIR, exist_ok=True)
    os.makedirs(MODEL_SAVE_DIR, exist_ok=True)

# Data preparation functions
def create_training_data(video_paths, output_dir, num_frames=1000):
    """Extract frames from videos to create training data"""
    os.makedirs(output_dir, exist_ok=True)
    
    frame_count = 0
    for video_path in video_paths:
        cap = cv2.VideoCapture(video_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        frames_to_extract = min(num_frames // len(video_paths), total_frames)
        
        # Calculate frame interval to get evenly distributed frames
        interval = max(1, total_frames // frames_to_extract)
        
        for i in tqdm(range(0, total_frames, interval), desc=f"Processing {os.path.basename(video_path)}"):
            cap.set(cv2.CAP_PROP_POS_FRAMES, i)
            ret, frame = cap.read()
            if not ret:
                continue
                
            # Save high-resolution frame
            hr_path = os.path.join(output_dir, f"frame_{frame_count}_hr.png")
            cv2.imwrite(hr_path, frame)
            
            # Create and save low-resolution frame
            h, w = frame.shape[:2]
            lr_h, lr_w = h // Config.SCALE_FACTOR, w // Config.SCALE_FACTOR
            lr_frame = cv2.resize(frame, (lr_w, lr_h), interpolation=cv2.INTER_CUBIC)
            lr_path = os.path.join(output_dir, f"frame_{frame_count}_lr.png")
            cv2.imwrite(lr_path, lr_frame)
            
            frame_count += 1
            if frame_count >= num_frames:
                break
                
        cap.release()
        if frame_count >= num_frames:
            break
            
    logger.info(f"Created {frame_count} training pairs in {output_dir}")

def create_patches(image_dir, patch_size=64, scale=2, stride=32):
    """Create training patches from images"""
    hr_patches = []
    lr_patches = []
    
    hr_files = sorted(glob.glob(os.path.join(image_dir, "*_hr.png")))
    lr_files = sorted(glob.glob(os.path.join(image_dir, "*_lr.png")))
    
    for hr_file, lr_file in tqdm(zip(hr_files, lr_files), total=len(hr_files), desc="Creating patches"):
        hr_img = cv2.imread(hr_file)
        lr_img = cv2.imread(lr_file)
        
        # Convert to RGB
        hr_img = cv2.cvtColor(hr_img, cv2.COLOR_BGR2RGB)
        lr_img = cv2.cvtColor(lr_img, cv2.COLOR_BGR2RGB)
        
        h, w = hr_img.shape[:2]
        
        # Extract patches
        for i in range(0, h - patch_size + 1, stride):
            for j in range(0, w - patch_size + 1, stride):
                hr_patch = hr_img[i:i+patch_size, j:j+patch_size]
                lr_patch = lr_img[i//scale:(i+patch_size)//scale, j//scale:(j+patch_size)//scale]
                
                hr_patches.append(hr_patch)
                lr_patches.append(lr_patch)
    
    # Convert to numpy arrays and normalize
    hr_patches = np.array(hr_patches) / 255.0
    lr_patches = np.array(lr_patches) / 255.0
    
    return lr_patches, hr_patches

# Model definition
def residual_block(x, filters, kernel_size=3, strides=1, padding='same'):
    """Residual block for SRGAN"""
    skip = x
    x = Conv2D(filters, kernel_size, strides=strides, padding=padding)(x)
    x = BatchNormalization()(x)
    x = LeakyReLU(alpha=0.2)(x)
    x = Conv2D(filters, kernel_size, strides=strides, padding=padding)(x)
    x = BatchNormalization()(x)
    x = Add()([x, skip])
    return x

def build_generator(input_shape=(None, None, 3), num_res_blocks=16):
    """Build the generator model for super-resolution"""
    inputs = Input(shape=input_shape)
    
    # Initial convolution
    x = Conv2D(64, 9, padding='same')(inputs)
    x = LeakyReLU(alpha=0.2)(x)
    skip_connection = x
    
    # Residual blocks
    for _ in range(num_res_blocks):
        x = residual_block(x, 64)
    
    # Post-residual convolution
    x = Conv2D(64, 3, padding='same')(x)
    x = BatchNormalization()(x)
    x = Add()([x, skip_connection])
    
    # Upsampling blocks
    x = Conv2D(256, 3, padding='same')(x)
    x = tf.nn.depth_to_space(x, 2)  # Pixel shuffle (upscale by 2x)
    x = LeakyReLU(alpha=0.2)(x)
    
    # Output convolution
    outputs = Conv2D(3, 9, padding='same', activation='tanh')(x)
    
    return Model(inputs, outputs)

# Training functions
def train_model(train_lr, train_hr, val_lr, val_hr, epochs=100, batch_size=16):
    """Train the super-resolution model"""
    # Build and compile the model
    model = build_generator(input_shape=train_lr.shape[1:])
    model.compile(optimizer=Adam(learning_rate=Config.LEARNING_RATE), loss='mse')
    
    # Setup callbacks
    callbacks = [
        ModelCheckpoint(Config.MODEL_CHECKPOINT, monitor='val_loss', save_best_only=True, verbose=1),
        ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, verbose=1, min_lr=1e-7),
        EarlyStopping(monitor='val_loss', patience=20, verbose=1, restore_best_weights=True)
    ]
    
    # Train the model
    history = model.fit(
        train_lr, train_hr,
        validation_data=(val_lr, val_hr),
        batch_size=batch_size,
        epochs=epochs,
        callbacks=callbacks,
        verbose=1
    )
    
    # Plot training history
    plt.figure(figsize=(10, 6))
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Model Loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend()
    plt.savefig(os.path.join(Config.MODEL_SAVE_DIR, 'training_history.png'))
    
    return model

# Main execution
def main():
    logger.info("Starting VidSharp AI model training")
    
    # Check if dataset exists, otherwise create it
    if not os.path.exists(Config.TRAIN_DIR) or len(os.listdir(Config.TRAIN_DIR)) < 10:
        logger.info("Creating training dataset...")
        # You would need to provide video paths here
        video_paths = glob.glob("sample_videos/*.mp4")
        if not video_paths:
            logger.error("No sample videos found. Please add videos to sample_videos/ directory")
            return
        create_training_data(video_paths, Config.TRAIN_DIR)
    
    if not os.path.exists(Config.VAL_DIR) or len(os.listdir(Config.VAL_DIR)) < 10:
        logger.info("Creating validation dataset...")
        video_paths = glob.glob("sample_videos/val/*.mp4")
        if not video_paths:
            # Use a subset of training videos for validation
            video_paths = glob.glob("sample_videos/*.mp4")[:1]
        create_training_data(video_paths, Config.VAL_DIR, num_frames=200)
    
    # Create training patches
    logger.info("Creating training patches...")
    train_lr, train_hr = create_patches(Config.TRAIN_DIR, Config.PATCH_SIZE, Config.SCALE_FACTOR)
    val_lr, val_hr = create_patches(Config.VAL_DIR, Config.PATCH_SIZE, Config.SCALE_FACTOR)
    
    logger.info(f"Training data shape: {train_lr.shape}, {train_hr.shape}")
    logger.info(f"Validation data shape: {val_lr.shape}, {val_hr.shape}")
    
    # Train the model
    logger.info("Training model...")
    model = train_model(
        train_lr, train_hr, val_lr, val_hr,
        epochs=Config.EPOCHS,
        batch_size=Config.BATCH_SIZE
    )
    
    # Save the final model
    model.save(os.path.join(Config.MODEL_SAVE_DIR, "srgan_final_model.h5"))
    logger.info(f"Model saved to {os.path.join(Config.MODEL_SAVE_DIR, 'srgan_final_model.h5')}")

if __name__ == "__main__":
    main()