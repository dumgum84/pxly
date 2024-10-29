PXLY is a tool for converting videos, images, and live streams into pixel art. It supports video conversion, live video capture, virtual camera streaming, and screen capturing.

# Features

- Video to Pixel Art**: Convert video files into pixel art video files.
- Image to Pixel Art**: Convert image files into pixel art images.
- Live Video to Pixel Art**: Utilizes a virtual camera to capture live video from a camera or screen and display it as pixel art.

# Installation

Install PXLY directly from PyPI:

pip install pxly

PXLY can be used from the command line or within a Python script.

Command Line Usage

- Convert an image to pixel art:
  pxly image path/to/image.jpg

- Convert a video to pixel art:
  pxly video path/to/video.mp4

- Start live video pixel art conversion:
  pxly live

# Python Script Usage

You can use PXLY’s CLI from within a Python script using the subprocess module:

import subprocess

Convert an image to pixel art using the CLI
subprocess.run(['pxly', 'image', 'path/to/image.jpg'])

Convert a video to pixel art using the CLI
subprocess.run(['pxly', 'video', 'path/to/video.mp4'])

Start live video pixel art conversion using the CLI
subprocess.run(['pxly', 'live'])

# Customizable Parameters

PXLY allows you to adjust various parameters to customize the pixel art output:

- Brightness, Contrast, and Vibrancy Adjustments**: Modify the appearance of the pixel art.
- Gamma Correction**: Fine-tune brightness and color balance for a more accurate visual representation.
- Background Removal**: Enable background removal using segmentation for a cleaner output.
- Palette Size**: Set the number of colors used in the pixel art output.
- Resolution and Frame Rate**: Set resolution and frame rate for virtual camera streaming or video processing.

# License

This project is licensed under the MIT License.

Enjoy creating pixel art with PXLY!
