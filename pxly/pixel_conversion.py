# pxly/pixel_conversion.py

import os
import logging
from PIL import Image
import cv2
import numpy as np
from multiprocessing import Pool, cpu_count
import shutil

from .pxly_imports import *
from .utils import adaptive_enhance_image, create_pixel_palette, apply_background_removal, apply_gamma_correction

# Initialize a global variable for selfie_segmentation
selfie_segmentation = None

def set_selfie_segmentation(segmentation):
    """
    Sets the global selfie_segmentation variable.

    Parameters:
    - segmentation: MediaPipe SelfieSegmentation object or None
    """
    global selfie_segmentation
    selfie_segmentation = segmentation

def vectorized_color_adjustment(pixels, lightness_boost, vibrancy_boost):
    """
    Apply lightness and vibrancy boosts to an array of RGB pixels using OpenCV's optimized functions.

    Parameters:
    - pixels: NumPy array of shape (height, width, 3) with RGB values in [0, 255].
    - lightness_boost: Float multiplier for lightness.
    - vibrancy_boost: Float multiplier for vibrancy (saturation).

    Returns:
    - Adjusted pixels as a NumPy array with RGB values in [0, 255].
    """
    # Convert RGB to HLS
    hls = cv2.cvtColor(pixels, cv2.COLOR_RGB2HLS).astype(np.float32)

    # Apply boosts
    hls[:, :, 1] = np.clip(hls[:, :, 1] * lightness_boost, 0, 255)  # Lightness channel
    hls[:, :, 2] = np.clip(hls[:, :, 2] * vibrancy_boost, 0, 255)   # Saturation channel

    # Convert back to RGB
    rgb_adjusted = cv2.cvtColor(hls.astype(np.uint8), cv2.COLOR_HLS2RGB)

    return rgb_adjusted

def image_to_pixel_art(
    image, 
    brightness_boost, contrast_boost,
    lightness_boost, vibrancy_boost, gamma,
    size, palette_size, 
    background_removal=False,
    black_and_white=False
):
    """Convert an image to pixel art with gamma correction."""

    global selfie_segmentation

    # Ensure gamma is above a minimum threshold
    min_gamma = 0.1
    gamma = max(gamma, min_gamma)

    # Remove background if enabled
    if background_removal and selfie_segmentation:
        image = apply_background_removal(image, selfie_segmentation)

    # Convert PIL image to NumPy array for OpenCV processing
    image_np = np.array(image)

    # Enhance image brightness and contrast
    if brightness_boost != 1.0 or contrast_boost != 1.0:
        image_np = adaptive_enhance_image(image_np, brightness_boost, contrast_boost)

    # Apply gamma correction
    image_np = apply_gamma_correction(image_np, gamma)

    # Adjust lightness and vibrancy
    if lightness_boost != 1.0 or vibrancy_boost != 1.0:
        image_np = vectorized_color_adjustment(image_np, lightness_boost, vibrancy_boost)

    # Resize image to pixelate
    resized_image = cv2.resize(
        image_np, 
        (max(1, image_np.shape[1] // size), max(1, image_np.shape[0] // size)), 
        interpolation=cv2.INTER_NEAREST
    )

    # Reduce color palette
    pixel_art = create_pixel_palette(resized_image, palette_size)

    # Scale back to original size
    pixel_art = cv2.resize(
        pixel_art, 
        (image_np.shape[1], image_np.shape[0]), 
        interpolation=cv2.INTER_NEAREST
    )

    pixel_art_pil = Image.fromarray(pixel_art)

    # Apply black and white filter if specified
    if black_and_white:
        pixel_art_pil = pixel_art_pil.convert("L").convert("RGB")  # Convert to grayscale and back to RGB

    return pixel_art_pil

def process_frame(
    file_name, frame_dir,
    brightness_boost, contrast_boost,
    lightness_boost, vibrancy_boost, gamma,
    size, palette_size,
    background_removal=False,
    black_and_white=False
):
    """Process a single frame into pixel art."""
    frame_path = os.path.join(frame_dir, file_name)
    try:
        with Image.open(frame_path) as image:
            pixel_art_img = image_to_pixel_art(
                image=image,
                brightness_boost=brightness_boost,
                contrast_boost=contrast_boost,
                lightness_boost=lightness_boost,
                vibrancy_boost=vibrancy_boost,
                gamma=gamma,
                size=size,
                palette_size=palette_size,
                background_removal=background_removal,
                black_and_white=black_and_white
            )
            pixel_art_img.save(frame_path)
    except Exception as e:
        logging.error(f"Error processing frame {file_name}: {e}")

def process_batch_frames(
    batch_files, frame_dir,
    brightness_boost, contrast_boost,
    lightness_boost, vibrancy_boost, gamma,
    size, palette_size,
    background_removal=False,
    black_and_white=False
):
    """Process a batch of frames into pixel art."""
    for file_name in batch_files:
        process_frame(
            file_name=file_name,
            frame_dir=frame_dir,
            brightness_boost=brightness_boost,
            contrast_boost=contrast_boost,
            lightness_boost=lightness_boost,
            vibrancy_boost=vibrancy_boost,
            gamma=gamma,
            size=size,
            palette_size=palette_size,
            background_removal=background_removal,
            black_and_white=black_and_white
        )

def worker_init(background_removal_enabled):
    """Initializer function for multiprocessing Pool workers."""
    global selfie_segmentation
    if background_removal_enabled:
        try:
            import mediapipe as mp
            mp_selfie_segmentation = mp.solutions.selfie_segmentation
            selfie_segmentation = mp_selfie_segmentation.SelfieSegmentation(model_selection=0)
            logging.info("Worker: MediaPipe Selfie Segmentation initialized.")
        except ImportError:
            logging.error("Worker: MediaPipe is not installed. Please install it using 'pip install mediapipe'.")
            selfie_segmentation = None
    else:
        selfie_segmentation = None

def convert_frames_to_pixel_art(
    frame_dir, 
    brightness_boost, contrast_boost,
    lightness_boost, vibrancy_boost, gamma,
    size, palette_size,
    background_removal=False,
    black_and_white=False
):
    """Convert frames to pixel art using multiprocessing with batch processing."""
    frame_files = sorted([f for f in os.listdir(frame_dir) if f.endswith('.png')])

    if not frame_files:
        logging.error(f"No PNG frames found in directory: {frame_dir}")
        return

    # Set the number of processes and batch size internally
    max_processes = max(1, int(cpu_count() / 4))
    batch_size = 10

    # Split frame_files into batches
    batches = [frame_files[i:i + batch_size] for i in range(0, len(frame_files), batch_size)]

    # Create a pool of worker processes with an initializer
    with Pool(processes=max_processes, initializer=worker_init, initargs=(background_removal,)) as pool:
        pool.starmap(
            process_batch_frames,
            [
                (
                    batch,
                    frame_dir,
                    brightness_boost,
                    contrast_boost,
                    lightness_boost,
                    vibrancy_boost,
                    gamma,
                    size,
                    palette_size,
                    background_removal,
                    black_and_white
                )
                for batch in batches
            ]
        )

def combine_pixel_frames_to_video(frame_dir, output_video_path, fps):
    """Combine pixel frames into a video using h264_nvenc or fallback to libx264."""
    logging.info("Combining pixel frames into video...")

    # Create a temporary directory to hold sequentially named frames
    sequential_dir = os.path.join(frame_dir, "sequential")
    os.makedirs(sequential_dir, exist_ok=True)

    # Rename frames to 'frame_%04d.png' sequentially
    sequential_files = sorted([f for f in os.listdir(frame_dir) if f.startswith('frame_') and f.endswith('.png')])
    for index, file_name in enumerate(sequential_files, start=1):
        src = os.path.join(frame_dir, file_name)
        dst = os.path.join(sequential_dir, f"frame_{index:04d}.png")
        shutil.move(src, dst)

    frame_path_pattern = os.path.join(sequential_dir, "frame_%04d.png")

    # Primary command using h264_nvenc
    primary_command = [
        "ffmpeg", "-y", "-framerate", str(fps), "-i", frame_path_pattern,
        "-c:v", "h264_nvenc",    # Hardware-accelerated encoder
        "-preset", "p7",         # Speed preset optimized for fastest encoding
        "-profile:v", "high",    # High profile for better quality
        "-pix_fmt", "yuv444p",   # Preserve color information
        "-b:v", "10M",           # Set bitrate to 10 Mbps (example)
        "-threads", "0",         # Utilize all available CPU cores
        output_video_path
    ]

    # Fallback command using libx264 with -tune animation
    fallback_command = [
        "ffmpeg", "-y", "-framerate", str(fps), "-i", frame_path_pattern,
        "-c:v", "libx264",         # Software encoder
        "-preset", "veryslow",     # Slower encoding with better compression
        "-crf", "18",              # Quality control (lower CRF for higher quality)
        "-profile:v", "high",      # High profile for better quality
        "-pix_fmt", "yuv444p",     # Preserve color information
        "-tune", "animation",      # Tune for animations
        "-threads", "0",           # Utilize all available CPU cores
        output_video_path
    ]

    success = execute_ffmpeg_with_fallback(primary_command, fallback_command)
    if not success:
        logging.error("Error combining frames into video with both encoders.")

    # Clean up the sequential frames directory
    shutil.rmtree(sequential_dir, ignore_errors=True)

def merge_audio_with_pixel_video(pixel_video, audio_path, final_output):
    """Combine the pixel video with the original audio."""
    logging.info("Merging audio with pixel video...")
    if audio_path:
        success = execute_ffmpeg_command([
            "ffmpeg", "-y", "-i", pixel_video, "-i", audio_path,
            "-c:v", "copy", "-c:a", "aac", "-b:a", "320k", "-strict", "experimental", "-threads", "0",
            final_output
        ])  # Set highest bitrate and utilize all CPU cores
        if success:
            logging.info(f"Final video with audio saved to {final_output}.")
        else:
            logging.error("Failed to merge audio with pixel video.")
    else:
        # If there's no audio, just copy the video
        shutil.copy(pixel_video, final_output)
        logging.info("No audio to merge. Final video saved without audio.")
