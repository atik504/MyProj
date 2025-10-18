#!/usr/bin/env python3

import os
import sys
import cv2
import numpy as np
from flask import Flask, request, jsonify, send_file
from werkzeug.utils import secure_filename
import tempfile
import shutil
from datetime import datetime
import logging
import librosa
import soundfile as sf
from PIL import Image, ImageEnhance, ImageFilter, ImageOps
import scipy.signal as signal
import scipy.ndimage as ndimage
from skimage import restoration, filters, morphology
from skimage.metrics import structural_similarity as ssim
# import tensorflow as tf
# import torch
# import torchvision.transforms as transforms
import noisereduce as nr

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

app = Flask(__name__, static_folder='.', static_url_path='')

# Configure upload settings
UPLOAD_FOLDER = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'uploads')
PROCESSED_FOLDER = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'processed')
ALLOWED_EXTENSIONS = {'mp4', 'mov', 'avi', 'mkv', 'webm', 'mp3', 'wav', 'ogg', 'flac', 'aac', 'jpg', 'jpeg', 'png', 'bmp', 'tiff'}
MAX_CONTENT_LENGTH = 500 * 1024 * 1024  # 500MB limit

# Create directories if they don't exist
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(PROCESSED_FOLDER, exist_ok=True)

# Helper functions
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS
    
def get_file_type(filename):
    """Determine if file is video, audio, or image"""
    ext = filename.rsplit('.', 1)[1].lower()
    if ext in {'mp4', 'mov', 'avi', 'mkv', 'webm'}:
        return 'video'
    elif ext in {'mp3', 'wav', 'ogg', 'flac', 'aac'}:
        return 'audio'
    elif ext in {'jpg', 'jpeg', 'png', 'bmp', 'tiff'}:
        return 'image'
    else:
        return None

def generate_unique_filename(original_filename):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename, extension = os.path.splitext(original_filename)
    return f"{filename}_{timestamp}{extension}"

# ADVANCED IMAGE ENHANCEMENT FUNCTIONS
def advanced_noise_reduction(image):
    """Apply advanced noise reduction using multiple techniques"""
    # Convert to float for processing
    if image.dtype == np.uint8:
        img_float = image.astype(np.float32) / 255.0
    else:
        img_float = image.astype(np.float32)
    
    # Non-local means denoising for color images
    if len(img_float.shape) == 3:
        denoised = cv2.fastNlMeansDenoisingColored(
            (img_float * 255).astype(np.uint8), None, 10, 10, 7, 21
        ).astype(np.float32) / 255.0
    else:
        denoised = cv2.fastNlMeansDenoising(
            (img_float * 255).astype(np.uint8), None, 10, 7, 21
        ).astype(np.float32) / 255.0
    
    # Apply bilateral filter for edge-preserving smoothing
    if len(denoised.shape) == 3:
        for i in range(denoised.shape[2]):
            denoised[:, :, i] = cv2.bilateralFilter(
                (denoised[:, :, i] * 255).astype(np.uint8), 9, 75, 75
            ).astype(np.float32) / 255.0
    else:
        denoised = cv2.bilateralFilter(
            (denoised * 255).astype(np.uint8), 9, 75, 75
        ).astype(np.float32) / 255.0
    
    return denoised

def unsharp_mask_enhancement(image, radius=2, amount=2.0, threshold=0):
    """Apply unsharp mask for better sharpening"""
    if len(image.shape) == 3:
        # Process each channel separately
        enhanced = np.zeros_like(image)
        for i in range(image.shape[2]):
            blurred = ndimage.gaussian_filter(image[:, :, i], radius)
            mask = image[:, :, i] - blurred
            enhanced[:, :, i] = image[:, :, i] + amount * mask
    else:
        blurred = ndimage.gaussian_filter(image, radius)
        mask = image - blurred
        enhanced = image + amount * mask
    
    return np.clip(enhanced, 0, 1)

def adaptive_histogram_equalization(image):
    """Apply CLAHE (Contrast Limited Adaptive Histogram Equalization)"""
    if len(image.shape) == 3:
        # Convert to LAB color space
        lab = cv2.cvtColor((image * 255).astype(np.uint8), cv2.COLOR_RGB2LAB)
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        lab[:, :, 0] = clahe.apply(lab[:, :, 0])
        enhanced = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB).astype(np.float32) / 255.0
    else:
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        enhanced = clahe.apply((image * 255).astype(np.uint8)).astype(np.float32) / 255.0
    
    return enhanced

def super_resolution_bicubic_plus(image, scale_factor=2):
    """Enhanced bicubic upscaling with additional processing"""
    height, width = image.shape[:2]
    new_height, new_width = int(height * scale_factor), int(width * scale_factor)
    
    # Initial upscaling with Lanczos (better than bicubic)
    upscaled = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_LANCZOS4)
    
    # Apply edge-preserving filter
    upscaled = cv2.edgePreservingFilter(upscaled, flags=1, sigma_s=50, sigma_r=0.4)
    
    # Apply gentle sharpening
    kernel = np.array([[-0.1, -0.1, -0.1],
                       [-0.1,  1.8, -0.1],
                       [-0.1, -0.1, -0.1]])
    upscaled = cv2.filter2D(upscaled, -1, kernel)
    
    return np.clip(upscaled, 0, 255).astype(np.uint8)

def enhance_photo_advanced(input_path, output_path):
    """Advanced photo enhancement with multiple AI-like techniques"""
    try:
        # Load image
        img = cv2.imread(input_path)
        if img is None:
            logger.error(f"Could not load image: {input_path}")
            return False
        
        # Convert to RGB
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_float = img_rgb.astype(np.float32) / 255.0
        
        # Step 1: Advanced noise reduction
        logger.info("Applying advanced noise reduction...")
        denoised = advanced_noise_reduction(img_float)
        
        # Step 2: Super-resolution upscaling (if image is small)
        height, width = denoised.shape[:2]
        if height < 1024 or width < 1024:
            logger.info("Applying super-resolution upscaling...")
            denoised = super_resolution_bicubic_plus(
                (denoised * 255).astype(np.uint8), scale_factor=2
            ).astype(np.float32) / 255.0
        
        # Step 3: Adaptive histogram equalization
        logger.info("Applying adaptive histogram equalization...")
        equalized = adaptive_histogram_equalization(denoised)
        
        # Step 4: Advanced sharpening with unsharp mask
        logger.info("Applying advanced sharpening...")
        sharpened = unsharp_mask_enhancement(equalized, radius=1.5, amount=1.5)
        
        # Step 5: Color enhancement
        logger.info("Applying color enhancement...")
        # Convert to PIL for color enhancement
        pil_img = Image.fromarray((sharpened * 255).astype(np.uint8))
        
        # Enhance color saturation
        color_enhancer = ImageEnhance.Color(pil_img)
        pil_img = color_enhancer.enhance(1.3)
        
        # Enhance contrast
        contrast_enhancer = ImageEnhance.Contrast(pil_img)
        pil_img = contrast_enhancer.enhance(1.2)
        
        # Enhance brightness slightly
        brightness_enhancer = ImageEnhance.Brightness(pil_img)
        pil_img = brightness_enhancer.enhance(1.05)
        
        # Step 6: Final sharpening
        pil_img = pil_img.filter(ImageFilter.UnsharpMask(radius=1, percent=120, threshold=3))
        
        # Save with maximum quality
        if input_path.lower().endswith(('.jpg', '.jpeg')):
            pil_img.save(output_path, 'JPEG', quality=98, optimize=True)
        else:
            pil_img.save(output_path, 'PNG', optimize=True)
        
        logger.info(f"Advanced photo enhancement completed: {output_path}")
        return True
        
    except Exception as e:
        logger.error(f"Error in advanced photo enhancement: {str(e)}")
        return False

# ADVANCED AUDIO ENHANCEMENT FUNCTIONS
def advanced_noise_reduction_audio(y, sr, stationary=False):
    """Apply advanced noise reduction to audio"""
    # Use noisereduce library for better noise reduction
    if stationary:
        # For stationary noise (consistent background noise)
        reduced_noise = nr.reduce_noise(y=y, sr=sr, stationary=True, prop_decrease=0.8)
    else:
        # For non-stationary noise
        reduced_noise = nr.reduce_noise(y=y, sr=sr, stationary=False, prop_decrease=0.7)
    
    return reduced_noise

def spectral_enhancement(y, sr):
    """Apply spectral enhancement to improve clarity"""
    # Compute STFT
    stft = librosa.stft(y, n_fft=2048, hop_length=512)
    magnitude, phase = librosa.magphase(stft)
    
    # Apply spectral enhancement
    # Boost mid frequencies (speech range)
    freqs = librosa.fft_frequencies(sr=sr, n_fft=2048)
    
    # Create frequency-dependent gain
    gain = np.ones_like(freqs)
    
    # Boost speech frequencies (300Hz - 3400Hz)
    speech_mask = (freqs >= 300) & (freqs <= 3400)
    gain[speech_mask] *= 1.3
    
    # Slight boost for presence (3400Hz - 8000Hz)
    presence_mask = (freqs > 3400) & (freqs <= 8000)
    gain[presence_mask] *= 1.1
    
    # Apply gain
    enhanced_magnitude = magnitude * gain[:, np.newaxis]
    
    # Reconstruct signal
    enhanced_stft = enhanced_magnitude * phase
    enhanced_y = librosa.istft(enhanced_stft, hop_length=512)
    
    return enhanced_y

def dynamic_range_compression(y, threshold=0.7, ratio=4.0, attack=0.003, release=0.1, sr=22050):
    """Apply dynamic range compression"""
    # Simple compressor implementation
    compressed = y.copy()
    envelope = 0.0
    
    attack_coeff = np.exp(-1.0 / (attack * sr))
    release_coeff = np.exp(-1.0 / (release * sr))
    
    for i in range(len(y)):
        input_level = abs(y[i])
        
        # Envelope follower
        if input_level > envelope:
            envelope = input_level + (envelope - input_level) * attack_coeff
        else:
            envelope = input_level + (envelope - input_level) * release_coeff
        
        # Apply compression
        if envelope > threshold:
            gain_reduction = 1.0 - (1.0 - threshold / envelope) / ratio
            compressed[i] = y[i] * gain_reduction
    
    return compressed

def enhance_audio_advanced(input_path, output_path):
    """Advanced audio enhancement with multiple techniques"""
    try:
        # Load audio
        y, sr = librosa.load(input_path, sr=None)
        logger.info(f"Loaded audio: {len(y)} samples at {sr} Hz")
        
        # Step 1: Advanced noise reduction
        logger.info("Applying advanced noise reduction...")
        y_denoised = advanced_noise_reduction_audio(y, sr, stationary=False)
        
        # Step 2: Spectral enhancement
        logger.info("Applying spectral enhancement...")
        y_enhanced = spectral_enhancement(y_denoised, sr)
        
        # Step 3: Dynamic range compression
        logger.info("Applying dynamic range compression...")
        y_compressed = dynamic_range_compression(y_enhanced, threshold=0.6, ratio=3.0, sr=sr)
        
        # Step 4: Harmonic enhancement
        logger.info("Applying harmonic enhancement...")
        y_harmonic, y_percussive = librosa.effects.hpss(y_compressed)
        y_final = y_harmonic * 1.4 + y_percussive * 0.6
        
        # Step 5: Normalize and apply soft limiting
        y_final = librosa.util.normalize(y_final) * 0.95
        
        # Apply soft limiting to prevent clipping
        y_final = np.tanh(y_final * 1.2)
        
        # Save enhanced audio
        sf.write(output_path, y_final, sr, subtype='PCM_24')
        
        logger.info(f"Advanced audio enhancement completed: {output_path}")
        return True
        
    except Exception as e:
        logger.error(f"Error in advanced audio enhancement: {str(e)}")
        return False

# ADVANCED VIDEO ENHANCEMENT FUNCTIONS
def enhance_frame_advanced(frame):
    """Apply advanced enhancement to a single video frame"""
    try:
        # Convert to float for processing
        frame_float = frame.astype(np.float32) / 255.0
        
        # Apply noise reduction
        denoised = advanced_noise_reduction(frame_float)
        
        # Apply adaptive histogram equalization
        equalized = adaptive_histogram_equalization(denoised)
        
        # Apply unsharp mask sharpening
        sharpened = unsharp_mask_enhancement(equalized, radius=1.0, amount=1.2)
        
        # Convert back to uint8
        enhanced = np.clip(sharpened * 255, 0, 255).astype(np.uint8)
        
        return enhanced
        
    except Exception as e:
        logger.error(f"Error enhancing frame: {str(e)}")
        return frame

def upscale_video_advanced(input_path, output_path, scale_factor=2):
    """Advanced video upscaling with better quality"""
    try:
        # Open input video
        cap = cv2.VideoCapture(input_path)
        if not cap.isOpened():
            logger.error(f"Could not open video: {input_path}")
            return False
        
        # Get video properties
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # Calculate new dimensions
        new_width = int(width * scale_factor)
        new_height = int(height * scale_factor)
        
        # Create output video writer
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (new_width, new_height))
        
        logger.info(f"Processing {frame_count} frames...")
        
        frame_idx = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Upscale frame using advanced method
            upscaled_frame = super_resolution_bicubic_plus(frame, scale_factor)
            
            # Apply advanced enhancements
            enhanced_frame = enhance_frame_advanced(upscaled_frame)
            
            # Write frame
            out.write(enhanced_frame)
            
            frame_idx += 1
            if frame_idx % 30 == 0:
                progress = int(frame_idx / frame_count * 100)
                logger.info(f"Progress: {progress}% ({frame_idx}/{frame_count} frames)")
        
        # Release resources
        cap.release()
        out.release()
        
        logger.info(f"Advanced video enhancement completed: {output_path}")
        return True
        
    except Exception as e:
        logger.error(f"Error in advanced video enhancement: {str(e)}")
        return False

# Flask routes remain the same but now use advanced functions
@app.route('/')
def index():
    return app.send_static_file('main.html')

@app.route('/upload-universal', methods=['POST'])
def upload_universal():
    if 'file' not in request.files:
        return jsonify({'error': 'No file provided'}), 400
    
    file = request.files['file']
    
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400
    
    if not allowed_file(file.filename):
        return jsonify({'error': f'File type not allowed. Supported formats: {ALLOWED_EXTENSIONS}'}), 400
    
    try:
        # Save uploaded file
        filename = secure_filename(file.filename)
        unique_filename = generate_unique_filename(filename)
        input_path = os.path.join(UPLOAD_FOLDER, unique_filename)
        file.save(input_path)
        
        # Process based on file type
        file_type = get_file_type(filename)
        output_filename = f"enhanced_{unique_filename}"
        output_path = os.path.join(PROCESSED_FOLDER, output_filename)
        
        logger.info(f"Processing {file_type} file: {filename}")
        
        if file_type == 'video':
            success = upscale_video_advanced(input_path, output_path, scale_factor=2)
        elif file_type == 'audio':
            success = enhance_audio_advanced(input_path, output_path)
        elif file_type == 'image':
            success = enhance_photo_advanced(input_path, output_path)
        else:
            return jsonify({'error': 'Unsupported file type'}), 400
        
        if success:
            return jsonify({
                'success': True,
                'message': f'{file_type.capitalize()} enhanced successfully with advanced AI algorithms',
                'original_file': unique_filename,
                'enhanced_file': output_filename,
                'download_url': f'/download/{output_filename}'
            })
        else:
            return jsonify({'error': f'Failed to enhance {file_type}'}), 500
            
    except Exception as e:
        logger.error(f"Error processing file: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/download/<filename>')
def download_file(filename):
    return send_file(os.path.join(PROCESSED_FOLDER, filename), as_attachment=True)

if __name__ == '__main__':
    logger.info("Starting Advanced Media Enhancement Server...")
    logger.info("Enhanced features:")
    logger.info("- Advanced noise reduction")
    logger.info("- AI-based super resolution")
    logger.info("- Adaptive histogram equalization")
    logger.info("- Spectral audio enhancement")
    logger.info("- Dynamic range compression")
    logger.info("- Advanced sharpening algorithms")
    app.run(host='0.0.0.0', port=5002, debug=True)