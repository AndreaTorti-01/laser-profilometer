# Laser Profilometer

Laser Profilometer is a Python-based software designed for extracting calibration data from images of chessboards and laser lines. The software can also profile objects scanned using the same laser, making it a powerful tool for laser-based 3D scanning applications.

## Features

- **Calibration Extraction**: Automatically detects chessboard patterns and laser lines to generate calibration files.
- **Object Profiling**: Processes laser line images to create profiles of scanned objects.
- **OpenCV-Powered**: Utilizes the OpenCV library for robust image processing.

## Requirements

- Python 3.8+
- OpenCV
- NumPy
- Matplotlib (optional, for visualization)

## Installation

Clone the repository and install the required dependencies:
```bash
git clone https://github.com/yourusername/laser-profilometer.git
cd laser-profilometer
pip install -r requirements.txt
