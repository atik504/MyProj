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
from PIL import Image, ImageEnhance, ImageFilter
import scipy.signal as signal
import scipy.ndimage as ndimage
from skimage import restoration, filters
import subprocess

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
    return f"fixed_enhanced_{filename}_{timestamp}{extension}"

# IMPROVED IMAGE ENHANCEMENT FUNCTIONS
def smart_noise_reduction(image):
    """Apply intelligent noise reduction that preserves details"""
    if image.dtype == np.uint8:
        img_float = image.astype(np.float32) / 255.0
    else:
        img_float = image.astype(np.float32)
    
    # Gentle noise reduction that preserves edges
    if len(img_float.shape) == 3:
        # Use gentler parameters to avoid blurriness
        denoised = cv2.fastNlMeansDenoisingColored(
            (img_float * 255).astype(np.uint8), None, 3, 3, 7, 21
        ).astype(np.float32) / 255.0
    else:
        denoised = cv2.fastNlMeansDenoising(
            (img_float * 255).astype(np.uint8), None, 3, 7, 21
        ).astype(np.float32) / 255.0
    
    return denoised

def edge_preserving_sharpening(image, strength=0.3):
    """Apply sharpening that preserves edges and avoids artifacts"""
    if len(image.shape) == 3:
        # Convert to LAB for better color preservation
        lab = cv2.cvtColor((image * 255).astype(np.uint8), cv2.COLOR_RGB2LAB)
        l_channel = lab[:, :, 0].astype(np.float32) / 255.0
        
        # Apply gentle unsharp mask to L channel only
        blurred = cv2.GaussianBlur(l_channel, (3, 3), 1.0)
        mask = l_channel - blurred
        sharpened_l = l_channel + strength * mask
        sharpened_l = np.clip(sharpened_l, 0, 1)
        
        lab[:, :, 0] = (sharpened_l * 255).astype(np.uint8)
        result = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB).astype(np.float32) / 255.0
    else:
        blurred = cv2.GaussianBlur(image, (3, 3), 1.0)
        mask = image - blurred
        result = image + strength * mask
        result = np.clip(result, 0, 1)
    
    return result

def intelligent_upscaling(image, scale_factor=1.5):
    """Intelligent upscaling that maintains sharpness"""
    if scale_factor == 1.0:
        return image
    
    height, width = image.shape[:2]
    new_height, new_width = int(height * scale_factor), int(width * scale_factor)
    
    # Use INTER_CUBIC for better quality
    upscaled = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_CUBIC)
    
    # Apply gentle sharpening after upscaling
    kernel = np.array([[0, -0.25, 0],
                       [-0.25, 2.0, -0.25],
                       [0, -0.25, 0]])
    sharpened = cv2.filter2D(upscaled, -1, kernel)
    
    # Blend original upscaled with sharpened version
    result = cv2.addWeighted(upscaled, 0.7, sharpened, 0.3, 0)
    
    return np.clip(result, 0, 255).astype(np.uint8)

def enhance_photo_fixed(input_path, output_path):
    """Fixed photo enhancement that avoids blurriness"""
    try:
        # Load image
        img = cv2.imread(input_path)
        if img is None:
            logger.error(f"Could not load image: {input_path}")
            return False
        
        # Convert to RGB
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        original_shape = img_rgb.shape
        logger.info(f"Processing image of size: {original_shape}")
        
        # Convert to float for processing
        img_float = img_rgb.astype(np.float32) / 255.0
        
        # Step 1: Gentle noise reduction
        logger.info("Applying gentle noise reduction...")
        denoised = smart_noise_reduction(img_float)
        
        # Step 2: Intelligent upscaling only if image is very small
        height, width = denoised.shape[:2]
        if height < 800 and width < 800:
            logger.info("Applying intelligent upscaling...")
            denoised = intelligent_upscaling(
                (denoised * 255).astype(np.uint8), scale_factor=1.5
            ).astype(np.float32) / 255.0
        
        # Step 3: Gentle contrast enhancement
        logger.info("Applying contrast enhancement...")
        # Convert to LAB for better color handling
        lab = cv2.cvtColor((denoised * 255).astype(np.uint8), cv2.COLOR_RGB2LAB)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        lab[:, :, 0] = clahe.apply(lab[:, :, 0])
        contrast_enhanced = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB).astype(np.float32) / 255.0
        
        # Step 4: Edge-preserving sharpening
        logger.info("Applying edge-preserving sharpening...")
        sharpened = edge_preserving_sharpening(contrast_enhanced, strength=0.2)
        
        # Step 5: Color enhancement using PIL
        logger.info("Applying color enhancement...")
        pil_img = Image.fromarray((sharpened * 255).astype(np.uint8))
        
        # Very gentle enhancements
        color_enhancer = ImageEnhance.Color(pil_img)
        pil_img = color_enhancer.enhance(1.1)  # Reduced from 1.3
        
        contrast_enhancer = ImageEnhance.Contrast(pil_img)
        pil_img = contrast_enhancer.enhance(1.1)  # Reduced from 1.2
        
        # Save with high quality
        if input_path.lower().endswith(('.jpg', '.jpeg')):
            pil_img.save(output_path, 'JPEG', quality=95, optimize=True)
        else:
            pil_img.save(output_path, 'PNG', optimize=True)
        
        logger.info(f"Fixed photo enhancement completed: {output_path}")
        return True
        
    except Exception as e:
        logger.error(f"Error in photo enhancement: {str(e)}")
        return False

# IMPROVED AUDIO ENHANCEMENT FUNCTIONS
def gentle_noise_reduction_audio(y, sr):
    """Apply gentle noise reduction that preserves audio quality"""
    # Use spectral gating approach
    stft = librosa.stft(y, n_fft=2048, hop_length=512)
    magnitude, phase = librosa.magphase(stft)
    
    # Estimate noise floor from quiet portions
    power = magnitude ** 2
    noise_power = np.percentile(power, 10)  # Use 10th percentile as noise estimate
    
    # Create soft mask
    threshold = noise_power * 3  # Conservative threshold
    mask = np.minimum(1.0, np.maximum(0.1, power / threshold))
    
    # Apply soft gating
    cleaned_stft = magnitude * mask * phase
    cleaned_y = librosa.istft(cleaned_stft, hop_length=512)
    
    return cleaned_y

def preserve_audio_quality(y, sr):
    """Apply enhancements while preserving audio quality"""
    # Gentle harmonic enhancement
    y_harmonic, y_percussive = librosa.effects.hpss(y, margin=3.0)
    
    # Blend harmonic and percussive with conservative ratio
    enhanced = y_harmonic * 1.1 + y_percussive * 0.9
    
    # Gentle EQ boost for voice clarity
    stft = librosa.stft(enhanced, n_fft=2048, hop_length=512)
    magnitude, phase = librosa.magphase(stft)
    
    freqs = librosa.fft_frequencies(sr=sr, n_fft=2048)
    
    # Very gentle EQ curve
    eq_curve = np.ones_like(freqs)
    voice_mask = (freqs >= 300) & (freqs <= 3000)
    eq_curve[voice_mask] *= 1.05  # Very gentle boost
    
    eq_stft = magnitude * eq_curve[:, np.newaxis] * phase
    eq_y = librosa.istft(eq_stft, hop_length=512)
    
    return eq_y

def enhance_audio_fixed(input_path, output_path):
    """Fixed audio enhancement that preserves playback compatibility"""
    try:
        # Load audio with original sample rate
        y, sr = librosa.load(input_path, sr=None)
        logger.info(f"Loaded audio: {len(y)} samples at {sr} Hz")
        
        # Step 1: Gentle noise reduction
        logger.info("Applying gentle noise reduction...")
        y_denoised = gentle_noise_reduction_audio(y, sr)
        
        # Step 2: Quality-preserving enhancement
        logger.info("Applying quality-preserving enhancement...")
        y_enhanced = preserve_audio_quality(y_denoised, sr)
        
        # Step 3: Normalize gently
        peak = np.abs(y_enhanced).max()
        if peak > 0:
            y_enhanced = y_enhanced / peak * 0.9  # Leave some headroom
        
        # Step 4: Save with appropriate format
        logger.info(f"Saving enhanced audio to: {output_path}")
        
        # Determine output format and save accordingly
        output_ext = os.path.splitext(output_path)[1].lower()
        
        if output_ext == '.mp3':
            # For MP3, we need to use a different approach
            temp_wav = output_path.replace('.mp3', '_temp.wav')
            sf.write(temp_wav, y_enhanced, sr, subtype='PCM_16')
            
            # Convert to MP3 using FFmpeg if available
            try:
                subprocess.run([
                    'ffmpeg', '-i', temp_wav, '-codec:a', 'libmp3lame', 
                    '-b:a', '192k', '-y', output_path
                ], check=True, capture_output=True)
                os.remove(temp_wav)
            except (subprocess.CalledProcessError, FileNotFoundError):
                # If FFmpeg not available, save as WAV
                logger.warning("FFmpeg not found, saving as WAV instead")
                output_path = output_path.replace('.mp3', '.wav')
                sf.write(output_path, y_enhanced, sr, subtype='PCM_16')
        else:
            # For WAV and other formats supported by soundfile
            sf.write(output_path, y_enhanced, sr, subtype='PCM_16')
        
        logger.info(f"Fixed audio enhancement completed: {output_path}")
        return True
        
    except Exception as e:
        logger.error(f"Error in audio enhancement: {str(e)}")
        return False

# IMPROVED VIDEO ENHANCEMENT FUNCTIONS
def enhance_frame_fixed(frame):
    """Apply gentle frame enhancement that avoids blurriness"""
    try:
        # Convert to float
        frame_float = frame.astype(np.float32) / 255.0
        
        # Very gentle noise reduction
        denoised = smart_noise_reduction(frame_float)
        
        # Gentle sharpening
        sharpened = edge_preserving_sharpening(denoised, strength=0.15)
        
        # Convert back to uint8
        result = np.clip(sharpened * 255, 0, 255).astype(np.uint8)
        
        return result
        
    except Exception as e:
        logger.error(f"Error enhancing frame: {str(e)}")
        return frame

def enhance_video_fixed(input_path, output_path):
    """Fixed video enhancement that maintains quality and compatibility"""
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
        
        logger.info(f"Video properties: {width}x{height} @ {fps} FPS, {frame_count} frames")
        
        # Use better codec for output
        fourcc = cv2.VideoWriter_fourcc(*'H264')  # Better codec
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        # If H264 doesn't work, fallback to XVID
        if not out.isOpened():
            fourcc = cv2.VideoWriter_fourcc(*'XVID')
            out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        if not out.isOpened():
            logger.error("Could not create output video writer")
            cap.release()
            return False
        
        logger.info(f"Processing {frame_count} frames...")
        
        frame_idx = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Apply gentle enhancement
            enhanced_frame = enhance_frame_fixed(frame)
            
            # Write frame
            out.write(enhanced_frame)
            
            frame_idx += 1
            if frame_idx % 30 == 0:
                progress = int(frame_idx / frame_count * 100)
                logger.info(f"Progress: {progress}% ({frame_idx}/{frame_count} frames)")
        
        # Release resources
        cap.release()
        out.release()
        
        # If the video file is very small, something went wrong
        if os.path.getsize(output_path) < 1000:
            logger.error("Output video file is too small, enhancement may have failed")
            return False
        
        logger.info(f"Fixed video enhancement completed: {output_path}")
        return True
        
    except Exception as e:
        logger.error(f"Error in video enhancement: {str(e)}")
        return False

# Flask routes
@app.route('/')
def index():
    return app.send_static_file('fixed_main.html')

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
            success = enhance_video_fixed(input_path, output_path)
        elif file_type == 'audio':
            success = enhance_audio_fixed(input_path, output_path)
        elif file_type == 'image':
            success = enhance_photo_fixed(input_path, output_path)
        else:
            return jsonify({'error': 'Unsupported file type'}), 400
        
        if success:
            # Get file size for verification
            output_size = os.path.getsize(output_path) if os.path.exists(output_path) else 0
            
            return jsonify({
                'success': True,
                'message': f'{file_type.capitalize()} enhanced successfully with fixed algorithms (no blur, proper playback)',
                'original_file': unique_filename,
                'enhanced_file': output_filename,
                'output_size': output_size,
                'download_url': f'/download/{output_filename}'
            })
        else:
            return jsonify({'error': f'Failed to enhance {file_type}'}), 500
            
    except Exception as e:
        logger.error(f"Error processing file: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/download/<filename>')
def download_file(filename):
    file_path = os.path.join(PROCESSED_FOLDER, filename)
    if os.path.exists(file_path):
        return send_file(file_path, as_attachment=True)
    else:
        return jsonify({'error': 'File not found'}), 404

if __name__ == '__main__':
    logger.info("Starting FIXED Advanced Media Enhancement Server...")
    logger.info("Fixed features:")
    logger.info("- No-blur video enhancement")
    logger.info("- Proper audio codec handling")
    logger.info("- Gentle noise reduction")
    logger.info("- Edge-preserving sharpening")
    logger.info("- Quality-preserving processing")
    logger.info("- Better codec compatibility")
    app.run(host='0.0.0.0', port=5003, debug=True)