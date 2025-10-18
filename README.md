# VidSharp AI

VidSharp AI is an advanced video enhancement application that uses machine learning to improve video quality. It can upscale resolution, reduce noise, enhance colors, and improve overall clarity while preserving the original content.

## Features

- **4K Upscaling**: Transform low-resolution videos into crystal-clear 4K content
- **Noise Reduction**: Eliminate grain and digital noise while preserving important details
- **Color Enhancement**: Revitalize dull colors and improve contrast for more vibrant visuals
- **Frame Rate Optimization**: Smooth out choppy videos with intelligent frame interpolation
- **Detail Recovery**: Restore lost details in compressed or low-quality footage
- **Fast Processing**: Efficient algorithms for quick enhancement of your videos

## Installation

1. Clone this repository:
   ```
   git clone https://github.com/yourusername/vidsharp.git
   cd vidsharp
   ```

2. Install the required dependencies:
   ```
   pip install -r requirements.txt
   ```

## Usage

### Running the Web Interface

1. Start the Flask server:
   ```
   python back.py
   ```

2. Open your browser and navigate to:
   ```
   http://localhost:5000/firstpage.html
   ```

### Using the Application

1. From the landing page, click "Try It Now" to access the main application
2. Upload your video by dragging and dropping or using the file selector
3. Adjust enhancement settings as desired
4. Click "Enhance Video" to process your video
5. Once processing is complete, preview the enhanced video
6. Download the enhanced video using the "Download" button

## Training Custom Models

To train custom enhancement models:

1. Prepare your dataset of low-quality and high-quality video pairs
2. Adjust training parameters in `training.py`
3. Run the training script:
   ```
   python training.py
   ```

## Project Structure

- `firstpage.html`: Landing page with information about VidSharp AI
- `main.html`: Main application interface for video enhancement
- `learn more.html`: Detailed information about the technology
- `back.py`: Flask backend for video processing
- `training.py`: Machine learning model training script
- `styles.css`: Global styles for the application

## Requirements

See `requirements.txt` for a complete list of dependencies.

## License

MIT

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.