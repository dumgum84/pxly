# pxly/video_processing.py

from .pxly_imports import *
from .utils import (
    convert_to_mp4, 
    extract_audio, 
    remove_audio_from_video, 
    execute_ffmpeg_command,
    execute_ffmpeg_with_fallback,
    clean_up,
    get_video_frame_rate
)
from .pixel_conversion import (
    process_batch_frames,
    convert_frames_to_pixel_art,
    worker_init
)

def extract_segment(video_path, frame_dir, fps, segment_index, segment_duration):
    """
    Extract a segment of the video using FFmpeg.

    Parameters:
    - video_path: Path to the input video file.
    - frame_dir: Directory to save the extracted frames.
    - fps: Frames per second for extraction.
    - segment_index: Index of the current segment.
    - segment_duration: Duration of the segment in seconds.
    """
    start_time = segment_index * segment_duration
    output_pattern = os.path.join(frame_dir, f"frame_{segment_index}_%04d.png")
    command = [
        "ffmpeg", "-y", "-i", video_path,
        "-ss", str(start_time),
        "-t", str(segment_duration),
        "-vf", f"fps={fps}",
        output_pattern
    ]
    success = subprocess.run(command, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL).returncode == 0
    if success:
        logging.info(f"Segment {segment_index} extracted successfully.")
    else:
        logging.error(f"Failed to extract segment {segment_index}.")

def extract_frames_parallel(video_path, frame_dir, fps):
    """
    Extract frames in parallel by splitting the video into segments and extracting each segment separately.
    This method leverages multiple FFmpeg processes to speed up frame extraction.
    """
    os.makedirs(frame_dir, exist_ok=True)
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        logging.error(f"Cannot open video file: {video_path}")
        return
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()

    frame_rate = get_video_frame_rate(video_path)
    duration = total_frames / frame_rate
    num_segments = max(1, int(cpu_count() / 4))  # Ensure at least one segment
    segment_duration = duration / num_segments

    logging.info("Starting frame extraction...")
    # Prepare arguments for each segment
    args_list = [
        (video_path, frame_dir, fps, segment_index, segment_duration)
        for segment_index in range(num_segments)
    ]

    # Use Pool.starmap to pass multiple arguments
    with Pool(processes=num_segments) as pool:
        pool.starmap(extract_segment, args_list)
    logging.info("Frame extraction completed.\nConverting frames to pixel art...")

def combine_pixel_frames_to_video(frame_dir, output_video_path, fps):
    """Combine pixel art frames into a video using h264_nvenc or fallback to libx264."""
    logging.info("Combining pixel art frames into video...")
    
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
    """Combine the pixel art video with the original audio."""
    logging.info("Merging audio with pixel art video...")
    if audio_path:
        success = execute_ffmpeg_command([
            "ffmpeg", "-y", "-i", pixel_video, "-i", audio_path,
            "-c:v", "copy", "-c:a", "aac", "-b:a", "320k", "-strict", "experimental", "-threads", "0",
            final_output
        ])  # Set highest bitrate and utilize all CPU cores
        if success:
            logging.info(f"Final video with audio saved to {final_output}.")
        else:
            logging.error("Failed to merge audio with pixel art video.")
    else:
        # If there's no audio, just copy the video
        shutil.copy(pixel_video, final_output)
        logging.info("No audio to merge. Final video saved without audio.")

def process_video(
    input_path,
    fps,
    brightness,
    contrast,
    lightness,
    vibrancy,
    gamma,
    size,
    palette_size,
    background_removal,
    target_width,
    target_height,
    black_white
):
    """Process a video file and convert it to pixel art."""
    output_dir = os.path.dirname(input_path)
    original_filename = os.path.splitext(os.path.basename(input_path))[0]

    mp4_video = convert_to_mp4(input_path, output_dir, target_width, target_height)
    if not mp4_video:
        return

    original_fps = get_video_frame_rate(mp4_video)

    audio_file = extract_audio(mp4_video, output_dir)
    if not audio_file:
        logging.warning("Proceeding without audio.")
        audio_file = None

    silent_video = remove_audio_from_video(mp4_video, output_dir)
    if not silent_video:
        return

    frame_dir = os.path.join(output_dir, "pxly_frames")
    
    # Use parallel frame extraction
    extract_frames_parallel(silent_video, frame_dir, fps)
    
    # Convert frames to pixel art with batch processing
    convert_frames_to_pixel_art(
        frame_dir, 
        brightness, contrast,
        lightness, vibrancy, gamma,
        size, palette_size, 
        background_removal,
        black_white
    )

    pixel_video_path = os.path.join(output_dir, f"{original_filename}_pxly_converted.mp4")
    combine_pixel_frames_to_video(frame_dir, pixel_video_path, fps)

    final_output = os.path.join(output_dir, f"{original_filename}_pxly.mp4")
    merge_audio_with_pixel_video(pixel_video_path, audio_file, final_output)

    files_to_delete = [mp4_video, audio_file, pixel_video_path, silent_video] if audio_file else [mp4_video, pixel_video_path]
    dirs_to_delete = [frame_dir]
    clean_up(files_to_delete, dirs_to_delete, input_path)

    logging.info("Video processing completed successfully.")
