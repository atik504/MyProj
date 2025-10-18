#!/usr/bin/env python3

import os
import sys
import glob
from fixed_enhanced_backend import enhance_photo_fixed, enhance_audio_fixed, enhance_video_fixed, get_file_type

def test_enhancements():
    """Test the fixed enhancement algorithms on existing files"""
    
    # Create test output directory
    test_output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'test_output')
    os.makedirs(test_output_dir, exist_ok=True)
    
    # Get some existing files from uploads
    upload_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'uploads')
    
    if not os.path.exists(upload_dir):
        print("No uploads directory found. Please upload some files first.")
        return
    
    # Find test files
    test_files = []
    for ext in ['*.jpg', '*.jpeg', '*.png', '*.mp3', '*.wav', '*.mp4', '*.avi']:
        test_files.extend(glob.glob(os.path.join(upload_dir, ext)))
    
    if not test_files:
        print("No test files found in uploads directory.")
        return
    
    print(f"Found {len(test_files)} test files:")
    for i, file_path in enumerate(test_files[:5], 1):  # Test first 5 files
        filename = os.path.basename(file_path)
        file_type = get_file_type(filename)
        print(f"{i}. {filename} ({file_type})")
        
        # Generate output path
        output_filename = f"test_enhanced_{filename}"
        output_path = os.path.join(test_output_dir, output_filename)
        
        print(f"   Processing {file_type}...")
        
        try:
            # Apply appropriate enhancement
            if file_type == 'image':
                success = enhance_photo_fixed(file_path, output_path)
            elif file_type == 'audio':
                success = enhance_audio_fixed(file_path, output_path)
            elif file_type == 'video':
                success = enhance_video_fixed(file_path, output_path)
            else:
                print(f"   Unsupported file type: {file_type}")
                continue
            
            if success:
                if os.path.exists(output_path):
                    output_size = os.path.getsize(output_path)
                    original_size = os.path.getsize(file_path)
                    print(f"   ✓ Success! Output: {output_filename} ({output_size} bytes, original: {original_size} bytes)")
                else:
                    print(f"   ✗ Failed: Output file not created")
            else:
                print(f"   ✗ Failed: Enhancement returned False")
                
        except Exception as e:
            print(f"   ✗ Error: {str(e)}")
        
        print()

if __name__ == "__main__":
    print("Testing Fixed Enhancement Algorithms")
    print("=" * 40)
    test_enhancements()
    print("Test completed!")
    print("\nFixed features:")
    print("- Reduced blurriness in videos")
    print("- Better audio codec compatibility")
    print("- Gentler processing parameters")
    print("- Edge-preserving enhancement")