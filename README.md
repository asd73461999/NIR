# Laser-drivenSpinel-type Ceramics Enabling NIR-II Light Sources for Penetration OpticalImaging Assisted by a Guided Filter Network Algorithm
doi: 10.1016/j.mattod.2025.10.019

ref: C. Li et al. Materials Today 91 (2025) 196–203
# NIR Image Processing

A comprehensive toolkit for Near-Infrared (NIR) image processing, enhancement, and analysis. This project includes various image processing techniques specifically designed for NIR imagery, including edge detection, contrast enhancement, and quality assessment.

## Features

- **Infrared Edge Detection**: Advanced edge detection algorithm tailored for infrared images
- **CLAHE Enhancement**: Contrast Limited Adaptive Histogram Equalization for improved image quality
- **Guided Filtering**: Enhances images using guided filtering techniques
- **Batch Processing**: Process entire directories of images automatically
- **Quality Metrics**: Calculate PSNR and SSIM metrics for image quality assessment
- **Visualization Tools**: Compare original and processed images side-by-side

## Installation

1. Clone this repository:
   ```bash
   git clone https://github.com/yourusername/NIR-Image-Processing.git
   cd NIR-Image-Processing
   ```

2. Install the required packages:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

### Command Line Interface

The toolkit supports three modes of operation:

#### Single Image Processing
Process a single image with CLAHE and sharpening:
```bash
python main.py --mode single --input path/to/image.jpg --output path/to/output/
```

#### Batch Processing
Process all images in a directory:
```bash
python main.py --mode batch --input path/to/input/dir --output path/to/output/dir
```

#### Quality Metrics Calculation
Calculate PSNR and SSIM metrics against a reference image:
```bash
python main.py --mode metrics --ref path/to/reference.jpg --input path/to/test/images --output path/to/results/
```

### Programmatic Usage

```python
from image_processor import InfraredEdgeDetector, enhance_image_clahe

# Initialize the detector
detector = InfraredEdgeDetector(sigma=1.5)

# Process an image
image_path = "path/to/your/image.jpg"
edges = detector.detect_edges(image_path, visualize=True)

# Or enhance with CLAHE and sharpening
original, enhanced = enhance_image_clahe(image_path)
```

## Modules

- `image_processor.py`: Core image processing classes and functions
- `batch_processor.py`: Functions for processing entire directories of images
- `metrics_calculator.py`: Tools for calculating image quality metrics
- `main.py`: Main entry point with CLI support

## Key Algorithms

### Infrared Edge Detection
1. Preprocessing with bilateral filtering and CLAHE
2. Detection of salient regions using Hessian matrix
3. Guided filtering enhancement
4. Canny edge detection

### CLAHE Enhancement
1. Conversion to YUV color space
2. Application of CLAHE to the Y channel
3. Conversion back to BGR color space
4. Sharpening with a custom kernel

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add some amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- Based on research in near-infrared imaging and processing techniques
- Inspired by various computer vision applications in industrial inspection
