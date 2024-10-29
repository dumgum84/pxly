# pxly/core.py

import logging
from .video_processing import process_video
from .image_processing import process_image
from .live_video_processing import process_live_video
from .utils import (
    is_image_file,
    is_video_file
)

def process_source(
    fps, brightness_boost, contrast_boost,
    lightness_boost, vibrancy_boost, gamma, pixel_size, palette_size, 
    target_width=None, target_height=None,
    black_and_white=False, background_removal=False, source='scr',
    selfie_segmentation=None
):
    """Wrapper to process live video."""
    process_live_video(
        fps=fps,
        brightness_boost=brightness_boost,
        contrast_boost=contrast_boost,
        lightness_boost=lightness_boost,
        vibrancy_boost=vibrancy_boost,
        gamma=gamma,
        pixel_size=pixel_size,
        palette_size=palette_size,
        target_width=target_width,
        target_height=target_height,
        black_and_white=black_and_white,
        background_removal=background_removal,
        source=source
    )

def process_input(
    input_path, fps, brightness_boost, contrast_boost,
    lightness_boost, vibrancy_boost, gamma, pixel_size, palette_size, 
    target_width=None, target_height=None,
    black_and_white=False, background_removal=False,
):
    """Determine the type of input and process accordingly."""
    if is_image_file(input_path):
        process_image(
            input_path=input_path,
            brightness_boost=brightness_boost,
            contrast_boost=contrast_boost,
            lightness_boost=lightness_boost,
            vibrancy_boost=vibrancy_boost,
            gamma=gamma,
            size=pixel_size,
            palette_size=palette_size,
            background_removal=background_removal,
            target_width=target_width,
            target_height=target_height,
            black_and_white=black_and_white
        )
    elif is_video_file(input_path):
        process_video(
            input_path=input_path,
            brightness_boost=brightness_boost,
            contrast_boost=contrast_boost,
            lightness_boost=lightness_boost,
            vibrancy_boost=vibrancy_boost,
            gamma=gamma,
            size=pixel_size,
            palette_size=palette_size,
            background_removal=background_removal,
            target_width=target_width,
            target_height=target_height,
            black_and_white=black_and_white,
            fps=fps
        )
    else:
        logging.error("Error: Unsupported file format.")
