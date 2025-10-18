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
from PIL import Image, ImageEnhance
import scipy.signal as signal

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

app = Flask(__name__, static_folder='.', static_url_path='')

# Configure upload settings
UPLOAD_FOLDER = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'uploads')
PROCESSED_FOLDER = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'processed')
ALLOWED_EXTENSIONS = {'mp4', 'mov', 'avi', 'mkv', 'webm', 'mp3', 'wav', 'ogg', 'jpg', 'jpeg', 'png'}
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
    elif ext in {'mp3', 'wav', 'ogg'}:
        return 'audio'
    elif ext in {'jpg', 'jpeg', 'png'}:
        return 'image'
    else:
        return None

def generate_unique_filename(original_filename):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename, extension = os.path.splitext(original_filename)
    return f"{filename}_{timestamp}{extension}"

# Video enhancement functions
def upscale_video(input_path, output_path, scale_factor=2):
    """Upscale video resolution using OpenCV"""
    try:
        # Open the input video
        cap = cv2.VideoCapture(input_path)
        if not cap.isOpened():
            logger.error(f"Could not open video file: {input_path}")
            return False

        # Get video properties
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        # Calculate new dimensions
        new_width = int(width * scale_factor)
        new_height = int(height * scale_factor)

        # Create video writer
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (new_width, new_height))

        # Process each frame
        frame_idx = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # Upscale the frame
            upscaled_frame = cv2.resize(frame, (new_width, new_height), interpolation=cv2.INTER_CUBIC)

            # Apply additional enhancements
            enhanced_frame = enhance_frame(upscaled_frame)

            # Write the frame
            out.write(enhanced_frame)

            # Log progress
            frame_idx += 1
            if frame_idx % 100 == 0:
                logger.info(f"Processed {frame_idx}/{frame_count} frames ({int(frame_idx/frame_count*100)}%)")

        # Release resources
        cap.release()
        out.release()
        logger.info(f"Video upscaling completed: {output_path}")
        return True

    except Exception as e:
        logger.error(f"Error upscaling video: {str(e)}")
        return False

def enhance_frame(frame):
    """Apply various enhancements to a single frame"""
    try:
        # Convert to float32 for processing
        frame_float = frame.astype(np.float32) / 255.0

        # Apply sharpening
        kernel = np.array([[-1, -1, -1],
                          [-1, 9, -1],
                          [-1, -1, -1]])
        sharpened = cv2.filter2D(frame_float, -1, kernel)

        # Apply color enhancement
        hsv = cv2.cvtColor(sharpened, cv2.COLOR_BGR2HSV)
        hsv[:,:,1] = hsv[:,:,1] * 1.2  # Increase saturation by 20%
        enhanced = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

        # Apply contrast enhancement
        enhanced = enhanced * 1.1  # Increase contrast by 10%
        enhanced = np.clip(enhanced, 0, 1)  # Ensure values stay in valid range

        # Convert back to uint8
        enhanced = (enhanced * 255).astype(np.uint8)

        # Apply noise reduction
        enhanced = cv2.fastNlMeansDenoisingColored(enhanced, None, 10, 10, 7, 21)

        return enhanced

    except Exception as e:
        logger.error(f"Error enhancing frame: {str(e)}")
        return frame  # Return original frame if enhancement fails

# Flask routes
@app.route('/')
def index():
    return app.send_static_file('main.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'video' not in request.files:
        return jsonify({'error': 'No video file provided'}), 400
    
    file = request.files['video']
    
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400
    
    if not allowed_file(file.filename):
        return jsonify({'error': f'File type not allowed. Supported formats: {ALLOWED_EXTENSIONS}'}), 400
    
    try:
        # Save uploaded file with unique name
        filename = secure_filename(file.filename)
        unique_filename = generate_unique_filename(filename)
        input_path = os.path.join(UPLOAD_FOLDER, unique_filename)
        file.save(input_path)
        
        # Process the video
        output_filename = f"enhanced_{unique_filename}"
        output_path = os.path.join(PROCESSED_FOLDER, output_filename)
        
        # Perform enhancement
        success = upscale_video(input_path, output_path)
        
        if success:
            return jsonify({
                'success': True,
                'message': 'Video enhanced successfully',
                'original_file': unique_filename,
                'enhanced_file': output_filename,
                'download_url': f'/download/{output_filename}'
            })
        else:
            return jsonify({'error': 'Failed to enhance video'}), 500
            
    except Exception as e:
        logger.error(f"Error processing upload: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/download/<filename>')
def download_file(filename):
    return send_file(os.path.join(PROCESSED_FOLDER, filename), as_attachment=True)

# Audio enhancement functions
def enhance_audio(input_path, output_path):
    """Enhance audio quality and voice clarity"""
    try:
        # Load audio file
        y, sr = librosa.load(input_path, sr=None)
        
        # Apply preprocessing
        y_processed = y.copy()
        
        # Voice enhancement - Vocal isolation using harmonic-percussive source separation
        y_harmonic, y_percussive = librosa.effects.hpss(y_processed)
        
        # Boost harmonic part (where voice typically resides)
        y_processed = y_harmonic * 1.2 + y_percussive * 0.8
        
        # Write enhanced audio to file
        sf.write(output_path, y_processed, sr)
        
        logger.info(f"Audio enhancement completed: {output_path}")
        return True
        
    except Exception as e:
        logger.error(f"Error enhancing audio: {str(e)}")
        return False

def noise_cancellation(input_path, output_path, noise_reduction_strength=0.8):
    """Apply noise cancellation to audio file"""
    try:
        # Load audio file
        y, sr = librosa.load(input_path, sr=None)
        
        # Estimate noise profile from the first 1 second
        noise_sample = y[:min(sr, len(y))]
        
        # Compute noise power spectrum
        noise_stft = librosa.stft(noise_sample)
        noise_power = np.mean(np.abs(noise_stft)**2, axis=1)
        
        # Compute STFT of the signal
        stft = librosa.stft(y)
        stft_mag, stft_phase = librosa.magphase(stft)
        
        # Apply spectral gating
        threshold = noise_reduction_strength * np.mean(noise_power)
        mask = (stft_mag > threshold).astype(float)
        
        # Apply mask and reconstruct signal
        stft_mag_filtered = stft_mag * mask
        stft_filtered = stft_mag_filtered * stft_phase
        y_filtered = librosa.istft(stft_filtered)
        
        # Write processed audio to file
        sf.write(output_path, y_filtered, sr)
        
        logger.info(f"Noise cancellation completed: {output_path}")
        return True
        
    except Exception as e:
        logger.error(f"Error applying noise cancellation: {str(e)}")
        return False

# Image enhancement functions
def enhance_photo(input_path, output_path):
    """Enhance photo quality similar to Remini app"""
    try:
        # Open image
        img = Image.open(input_path)
        
        # Apply enhancements
        # 1. Sharpness enhancement
        enhancer = ImageEnhance.Sharpness(img)
        img = enhancer.enhance(1.5)  # Increase sharpness by 50%
        
        # 2. Contrast enhancement
        enhancer = ImageEnhance.Contrast(img)
        img = enhancer.enhance(1.2)  # Increase contrast by 20%
        
        # 3. Color enhancement
        enhancer = ImageEnhance.Color(img)
        img = enhancer.enhance(1.1)  # Increase color saturation by 10%
        
        # 4. Brightness adjustment
        enhancer = ImageEnhance.Brightness(img)
        img = enhancer.enhance(1.1)  # Increase brightness by 10%
        
        # Save enhanced image
        img.save(output_path, quality=95)  # High quality JPEG
        
        logger.info(f"Photo enhancement completed: {output_path}")
        return True
        
    except Exception as e:
        logger.error(f"Error enhancing photo: {str(e)}")
        return False

# Additional Flask routes for audio and image processing
@app.route('/upload-audio', methods=['POST'])
def upload_audio():
    if 'audio' not in request.files:
        return jsonify({'error': 'No audio file provided'}), 400
    
    file = request.files['audio']
    
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400
    
    if not allowed_file(file.filename):
        return jsonify({'error': f'File type not allowed. Supported audio formats: mp3, wav, ogg'}), 400
    
    try:
        # Save uploaded file with unique name
        filename = secure_filename(file.filename)
        unique_filename = generate_unique_filename(filename)
        input_path = os.path.join(UPLOAD_FOLDER, unique_filename)
        file.save(input_path)
        
        # Process the audio
        output_filename = f"enhanced_{unique_filename}"
        output_path = os.path.join(PROCESSED_FOLDER, output_filename)
        
        # Check if noise cancellation is requested
        noise_cancel = request.form.get('noise_cancellation', 'false').lower() == 'true'
        
        if noise_cancel:
            # Apply noise cancellation
            temp_output = os.path.join(PROCESSED_FOLDER, f"temp_{unique_filename}")
            success = enhance_audio(input_path, temp_output)
            if success:
                success = noise_cancellation(temp_output, output_path)
                os.remove(temp_output)  # Clean up temp file
        else:
            # Just enhance audio
            success = enhance_audio(input_path, output_path)
        
        if success:
            return jsonify({
                'success': True,
                'message': 'Audio enhanced successfully',
                'original_file': unique_filename,
                'enhanced_file': output_filename,
                'download_url': f'/download/{output_filename}'
            })
        else:
            return jsonify({'error': 'Failed to enhance audio'}), 500
            
    except Exception as e:
        logger.error(f"Error processing audio upload: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/upload-photo', methods=['POST'])
def upload_photo():
    if 'photo' not in request.files:
        return jsonify({'error': 'No photo file provided'}), 400
    
    file = request.files['photo']
    
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400
    
    if not allowed_file(file.filename):
        return jsonify({'error': f'File type not allowed. Supported image formats: jpg, jpeg, png'}), 400
    
    try:
        # Save uploaded file with unique name
        filename = secure_filename(file.filename)
        unique_filename = generate_unique_filename(filename)
        input_path = os.path.join(UPLOAD_FOLDER, unique_filename)
        file.save(input_path)
        
        # Process the photo
        output_filename = f"enhanced_{unique_filename}"
        output_path = os.path.join(PROCESSED_FOLDER, output_filename)
        
        # Enhance photo
        success = enhance_photo(input_path, output_path)
        
        if success:
            return jsonify({
                'success': True,
                'message': 'Photo enhanced successfully',
                'original_file': unique_filename,
                'enhanced_file': output_filename,
                'download_url': f'/download/{output_filename}'
            })
        else:
            return jsonify({'error': 'Failed to enhance photo'}), 500
            
    except Exception as e:
        logger.error(f"Error processing photo upload: {str(e)}")
        return jsonify({'error': str(e)}), 500

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
        # Save uploaded file with unique name
        filename = secure_filename(file.filename)
        unique_filename = generate_unique_filename(filename)
        input_path = os.path.join(UPLOAD_FOLDER, unique_filename)
        file.save(input_path)
        
        # Process based on file type
        file_type = get_file_type(filename)
        output_filename = f"enhanced_{unique_filename}"
        output_path = os.path.join(PROCESSED_FOLDER, output_filename)
        
        if file_type == 'video':
            success = upscale_video(input_path, output_path)
        elif file_type == 'audio':
            # Check if noise cancellation is requested
            noise_cancel = request.form.get('noise_cancellation', 'false').lower() == 'true'
            
            if noise_cancel:
                # Apply noise cancellation
                temp_output = os.path.join(PROCESSED_FOLDER, f"temp_{unique_filename}")
                success = enhance_audio(input_path, temp_output)
                if success:
                    success = noise_cancellation(temp_output, output_path)
                    os.remove(temp_output)  # Clean up temp file
            else:
                # Just enhance audio
                success = enhance_audio(input_path, output_path)
        elif file_type == 'image':
            success = enhance_photo(input_path, output_path)
        else:
            return jsonify({'error': 'Unsupported file type'}), 400
        
        if success:
            return jsonify({
                'success': True,
                'message': f'{file_type.capitalize()} enhanced successfully',
                'original_file': unique_filename,
                'enhanced_file': output_filename,
                'download_url': f'/download/{output_filename}'
            })
        else:
            return jsonify({'error': f'Failed to enhance {file_type}'}), 500
            
    except Exception as e:
        logger.error(f"Error processing universal upload: {str(e)}")
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5001, debug=True)