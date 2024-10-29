# pxly/utils.py

from .pxly_imports import *

logging.basicConfig(level=logging.INFO, format='%(message)s')

def execute_ffmpeg_command(command):
    """Run an FFmpeg command with error handling, suppressing FFmpeg output."""
    try:
        subprocess.run(command, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        return True
    except subprocess.CalledProcessError as e:
        logging.error(f"FFmpeg command failed: {e}")
        return False

def execute_ffmpeg_with_fallback(primary_command, fallback_command):
    """
    Attempt to run the primary FFmpeg command. If it fails, run the fallback command.

    Parameters:
    - primary_command: List of strings representing the primary FFmpeg command.
    - fallback_command: List of strings representing the fallback FFmpeg command.

    Returns:
    - True if either command succeeds.
    - False if both commands fail.
    """
    try:
        subprocess.run(primary_command, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        return True
    except subprocess.CalledProcessError:
        try:
            subprocess.run(fallback_command, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            return True
        except subprocess.CalledProcessError as e:
            logging.error(f"FFmpeg command failed: {e}")
            return False

def get_video_frame_rate(video_path):
    """Get the frame rate of the video using OpenCV."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        logging.error(f"Cannot open video file: {video_path}")
        return None
    frame_rate = cap.get(cv2.CAP_PROP_FPS)
    cap.release()
    return frame_rate

def is_image_file(file_path):
    """Check if the file is an image based on its extension."""
    image_extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.gif', '.tiff', '.webp')
    return file_path.lower().endswith(image_extensions)

def is_video_file(file_path):
    """Check if the file is a video based on its extension."""
    video_extensions = ('.mp4', '.avi', '.mov', '.mkv', '.flv', '.wmv', '.webm', '.3gp')
    return file_path.lower().endswith(video_extensions)

def clean_up(files_to_delete, dirs_to_delete, input_path):
    """Delete intermediate files and directories."""
    logging.info("Cleaning up intermediate files...")
    for file_path in files_to_delete:
        if file_path and os.path.exists(file_path) and file_path != input_path:
            os.remove(file_path)
    for dir_path in dirs_to_delete:
        if os.path.exists(dir_path):
            shutil.rmtree(dir_path, ignore_errors=True)

def apply_background_removal(image, selfie_segmentation):
    """Apply background removal using MediaPipe's Selfie Segmentation."""

    image_rgb = np.array(image.convert('RGB'))
    original_height, original_width = image_rgb.shape[:2]

    # Determine model type based on aspect ratio
    aspect_ratio = original_width / original_height
    model_type = 0 if aspect_ratio <= 1 else 1

    # Resize for model compatibility
    resized_frame = resize_for_model(image_rgb, model_type)

    # Run segmentation model
    results = selfie_segmentation.process(resized_frame)
    segmentation_mask = results.segmentation_mask

    # Resize mask back to original size
    resized_mask = resize_mask(segmentation_mask, original_width, original_height)

    # Apply binary thresholding
    _, binary_mask = cv2.threshold(resized_mask, 0.5, 1, cv2.THRESH_BINARY)

    # Apply morphological operations
    refined_mask = refine_mask(binary_mask)

    # Apply Gaussian smoothing
    blurred_mask = cv2.GaussianBlur(refined_mask.astype(np.float32), (5, 5), 0)

    # Convert mask to 3 channels
    alpha_mask = np.repeat(blurred_mask[:, :, np.newaxis], 3, axis=2)

    # Apply the mask to the image
    segmented_image = image_rgb * alpha_mask + (1 - alpha_mask) * 0  # Assuming black background
    segmented_image = segmented_image.astype(np.uint8)

    return Image.fromarray(segmented_image)

def resize_for_model(frame, model_type):
    """Resize the frame for MediaPipe model compatibility."""
    if model_type == 0:
        resized_frame = cv2.resize(frame, (256, 256), interpolation=cv2.INTER_AREA)
    else:
        resized_frame = cv2.resize(frame, (256, 144), interpolation=cv2.INTER_AREA)
    return resized_frame

def resize_mask(mask, original_width, original_height):
    """Resize the mask back to the original frame size."""
    return cv2.resize(mask, (original_width, original_height), interpolation=cv2.INTER_LINEAR)

def refine_mask(mask):
    """Apply morphological operations to refine the mask."""
    kernel = np.ones((3, 3), np.uint8)
    eroded_mask = cv2.erode(mask, kernel, iterations=1)
    refined_mask = cv2.dilate(eroded_mask, kernel, iterations=1)
    return refined_mask

def convert_to_png(input_path, output_dir):
    """Convert input image to PNG format or copy it if already PNG."""
    original_filename = os.path.splitext(os.path.basename(input_path))[0]
    output_path = os.path.join(output_dir, f"{original_filename}_converted.png")
    if not input_path.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif', '.tiff', '.webp')):
        logging.error("Unsupported image format.")
        return None
    if not input_path.lower().endswith(".png"):
        logging.info("Converting image to PNG format...")
        try:
            with Image.open(input_path) as img:
                img.save(output_path, format="PNG")
            logging.info("Image successfully converted to PNG.")
        except Exception as e:
            logging.error(f"Error converting image to PNG: {e}")
            return None
    else:
        # Input is already PNG, make a copy to avoid overwriting
        if os.path.exists(output_path):
            os.remove(output_path)
        shutil.copy(input_path, output_path)
        logging.info("Copied original PNG image for processing.")
    return output_path

def create_pixel_palette(image, palette_size):
    """
    Reduce the image to a specific number of colors using k-means clustering.

    Parameters:
    - image: NumPy array of the image in RGB format.
    - palette_size: Number of colors in the palette.

    Returns:
    - Pixel art image as a NumPy array.
    """
    # Reshape the image to a 2D array of pixels
    pixels = image.reshape(-1, 3)

    # Convert to float32 for k-means
    pixels = np.float32(pixels)

    # Define criteria and apply k-means
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 0.2)
    _, labels, centers = cv2.kmeans(pixels, palette_size, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)

    # Convert back to uint8 and replace pixel colors with centers
    centers = np.uint8(centers)
    pixel_art = centers[labels.flatten()]
    pixel_art = pixel_art.reshape(image.shape)

    return pixel_art

def pad_frame_to_target_size(frame, target_width, target_height):
    """Pad or resize the frame to match the target width and height."""
    height, width, _ = frame.shape

    # If the frame is larger than the target, resize it to fit within the target dimensions
    if width > target_width or height > target_height:
        frame = cv2.resize(frame, (target_width, target_height), interpolation=cv2.INTER_AREA)
        return frame

    # Calculate padding for height and width
    pad_top = max((target_height - height) // 2, 0)
    pad_bottom = max(target_height - height - pad_top, 0)
    pad_left = max((target_width - width) // 2, 0)
    pad_right = max(target_width - width - pad_left, 0)

    # Pad the frame with black borders if necessary
    padded_frame = cv2.copyMakeBorder(
        frame, pad_top, pad_bottom, pad_left, pad_right,
        cv2.BORDER_CONSTANT, value=[0, 0, 0]
    )

    return padded_frame

def convert_to_mp4(input_path, output_dir, target_width=None, target_height=None):
    """
    Convert input video to MP4 format using FFmpeg.
    If the input is already MP4, copy it to avoid overwriting.

    Parameters:
    - input_path: Path to the input video file.
    - output_dir: Directory to save the converted video.
    - target_width: Optional target width.
    - target_height: Optional target height.

    Returns:
    - Path to the converted MP4 video.
    """
    original_filename = os.path.splitext(os.path.basename(input_path))[0]
    output_path = os.path.join(output_dir, f"{original_filename}_temp.mp4")
    
    if not input_path.lower().endswith(".mp4"):
        logging.info("Converting video to MP4 format...")
        command = [
            "ffmpeg", "-y", "-i", input_path
        ]
        if target_width and target_height:
            command.extend(["-vf", f"scale={target_width}:{target_height}"])
        command.append(output_path)
        
        success = execute_ffmpeg_command(command)
        if success:
            logging.info("Video successfully converted to MP4.")
            return output_path
        else:
            logging.error("Failed to convert video to MP4.")
            return None
    else:
        # Input is already MP4, make a copy to avoid overwriting
        if os.path.exists(output_path):
            os.remove(output_path)
        shutil.copy(input_path, output_path)
        logging.info("Copied original MP4 video for processing.")
        return output_path

def extract_audio(video_path, output_dir):
    """
    Extract audio from the video using FFmpeg.

    Parameters:
    - video_path: Path to the input video file.
    - output_dir: Directory to save the extracted audio.

    Returns:
    - Path to the extracted audio file (in AAC format).
    """
    original_filename = os.path.splitext(os.path.basename(video_path))[0]
    audio_path = os.path.join(output_dir, f"{original_filename}_audio.aac")
    logging.info("Extracting audio from video...")
    command = [
        "ffmpeg", "-y", "-i", video_path, "-vn", "-acodec", "aac", audio_path
    ]
    success = execute_ffmpeg_command(command)
    if success:
        logging.info("Audio successfully extracted.")
        return audio_path
    else:
        logging.error("Failed to extract audio from video.")
        return None

def remove_audio_from_video(video_path, output_dir):
    """
    Remove audio from the video using FFmpeg.

    Parameters:
    - video_path: Path to the input video file.
    - output_dir: Directory to save the silent video.

    Returns:
    - Path to the silent video file (in MP4 format).
    """
    original_filename = os.path.splitext(os.path.basename(video_path))[0]
    silent_video_path = os.path.join(output_dir, f"{original_filename}_video.mp4")
    logging.info("Removing audio from video...")
    command = [
        "ffmpeg", "-y", "-i", video_path, "-an", "-c:v", "copy", silent_video_path
    ]
    success = execute_ffmpeg_command(command)
    if success:
        logging.info("Audio successfully removed from video.")
        return silent_video_path
    else:
        logging.error("Failed to remove audio from video.")
        return None
    
def adaptive_enhance_image(image_np, brightness_boost, contrast_boost):
    """
    Optimized version of adaptive_enhance_image using OpenCV.
    """
    # Compute mean brightness directly on RGB channels for speed
    avg_brightness = np.mean(image_np, axis=(0, 1))

    # Compute scaling factor with precomputed constants
    scaling_factor = 1 + ((128 - avg_brightness.mean()) / 128) * 0.1
    scaling_factor = np.clip(scaling_factor, 0.9, 1.1)

    # Adjust brightness and contrast factors
    brightness_factor = brightness_boost * scaling_factor
    contrast_factor = contrast_boost * scaling_factor

    # Adjust brightness and contrast using in-place operation for efficiency
    cv2.convertScaleAbs(image_np, alpha=contrast_factor, beta=brightness_factor, dst=image_np)

    return image_np

def apply_gamma_correction(image_np, gamma):
    """
    Apply gamma correction to the image.

    Parameters:
    - image_np: NumPy array of shape (H, W, 3) with RGB values in [0, 255].
    - gamma: Gamma value for correction.

    Returns:
    - Gamma-corrected image as a NumPy array.
    """
    inv_gamma = 1.0 / gamma
    # Build a lookup table mapping pixel values [0, 255] to adjusted gamma values
    table = np.array([((i / 255.0) ** inv_gamma) * 255 for i in np.arange(256)]).astype("uint8")
    # Apply gamma correction using the lookup table
    return cv2.LUT(image_np, table)