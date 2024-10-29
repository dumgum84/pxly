# pxly/cli.py

from .pxly_imports import *
from .core import process_image, process_video, process_live_video
from .pixel_conversion import (
    convert_frames_to_pixel_art,
    combine_pixel_frames_to_video,
    merge_audio_with_pixel_video,
    set_selfie_segmentation
)
from PIL import Image

def parse_arguments():
    parser = argparse.ArgumentParser(
        description="pxly: Convert images and videos to pixel art, or process live video streams.",
    )

    subparsers = parser.add_subparsers(dest='command', help='Available subcommands')

    # Image Processing Subcommand
    image_parser = subparsers.add_parser('image', help='Convert an image to pixel art.')
    image_parser.add_argument("input_path", help="Path to the input image file.", metavar='')
    image_parser.add_argument("-b", "--brightness", type=float, default=1.0, 
                              help="Brightness boost (default: 1.0)", metavar='')
    image_parser.add_argument("-c", "--contrast", type=float, default=1.0, 
                              help="Contrast boost (default: 1.0)", metavar='')
    image_parser.add_argument("-l", "--lightness", type=float, default=1.0, 
                              help="Lightness boost (default: 1.0)", metavar='')
    image_parser.add_argument("-v", "--vibrancy", type=float, default=1.0, 
                              help="Vibrancy boost (default: 1.0)", metavar='')
    image_parser.add_argument("-g", "--gamma", type=float, default=1.0, 
                              help="Gamma correction (default: 1.0)", metavar='')
    image_parser.add_argument("-s", "--size", type=int, default=10,
                              help="Size of each pixel block (default: 10)", metavar='')
    image_parser.add_argument("-p", "--palette_size", type=int, default=16,
                              help="Number of colors in the palette (default: 16)", metavar='')
    image_parser.add_argument("-r", "--resolution", type=str, default=None,
                              help="Target resolution for the output pixel art in WIDTHxHEIGHT format, e.g., 1920x1080.", metavar='')
    image_parser.add_argument("-bw", "--black_white", action="store_true", 
                              help="Render pixel art in black and white.")
    image_parser.add_argument("-br", "--background_removal", action="store_true",
                              help="Enable background removal using MediaPipe Selfie Segmentation.")

    # Video Processing Subcommand
    video_parser = subparsers.add_parser('video', help='Convert a video to pixel art.')
    video_parser.add_argument("input_path", help="Path to the input video file.", metavar='')
    video_parser.add_argument("-f", "--fps", type=int, default=30, 
                              help="Frames per second for frame extraction (default: 30)", metavar='')
    video_parser.add_argument("-b", "--brightness", type=float, default=1.0, 
                              help="Brightness boost (default: 1.0)", metavar='')
    video_parser.add_argument("-c", "--contrast", type=float, default=1.0, 
                              help="Contrast boost (default: 1.0)", metavar='')
    video_parser.add_argument("-l", "--lightness", type=float, default=1.0, 
                              help="Lightness boost (default: 1.0)", metavar='')
    video_parser.add_argument("-v", "--vibrancy", type=float, default=1.0, 
                              help="Vibrancy boost (default: 1.0)", metavar='')
    video_parser.add_argument("-g", "--gamma", type=float, default=1.0, 
                              help="Gamma correction (default: 1.0)", metavar='')
    video_parser.add_argument("-s", "--size", type=int, default=10,
                              help="Size of each pixel block (default: 10)", metavar='')
    video_parser.add_argument("-p", "--palette_size", type=int, default=16,
                              help="Number of colors in the palette (default: 16)", metavar='')
    video_parser.add_argument("-r", "--resolution", type=str, default=None,
                              help="Target resolution for the output pixel art/video in WIDTHxHEIGHT format.", metavar='')
    video_parser.add_argument("-bw", "--black_white", action="store_true", 
                              help="Render pixel art in black and white.")
    video_parser.add_argument("-br", "--background_removal", action="store_true",
                              help="Enable background removal using MediaPipe Selfie Segmentation.")

    # Live Video Processing Subcommand
    live_parser = subparsers.add_parser('live', help='Process live video and stream pixel art to a virtual camera.')
    live_parser.add_argument("-f", "--fps", type=int, default=30, 
                             help="Frames per second for processing (default: 30)", metavar='')
    live_parser.add_argument("-b", "--brightness", type=float, default=1.0, 
                             help="Brightness boost (default: 1.0)", metavar='')
    live_parser.add_argument("-c", "--contrast", type=float, default=1.0, 
                             help="Contrast boost (default: 1.0)", metavar='')
    live_parser.add_argument("-l", "--lightness", type=float, default=1.0, 
                             help="Lightness boost (default: 1.0)", metavar='')
    live_parser.add_argument("-v", "--vibrancy", type=float, default=1.0, 
                             help="Vibrancy boost (default: 1.0)", metavar='')
    live_parser.add_argument("-g", "--gamma", type=float, default=1.0, 
                             help="Gamma correction (default: 1.0)", metavar='')
    live_parser.add_argument("-s", "--size", type=int, default=10,
                             help="Size of each pixel block (default: 10)", metavar='')
    live_parser.add_argument("-pal", "--palette_size", type=int, default=16,
                             help="Number of colors in the palette (default: 16)", metavar='')
    live_parser.add_argument("-r", "--resolution", type=str, default=None,
                             help="Target resolution for the output pixel art/video in WIDTHxHEIGHT format.", metavar='')
    live_parser.add_argument("-bw", "--black_white", action="store_true", 
                             help="Render pixel art in black and white.")
    live_parser.add_argument("-br", "--background_removal", action="store_true",
                             help="Enable background removal using MediaPipe Selfie Segmentation.")
    live_parser.add_argument("-src", "--source", type=str, choices=['ca', 'sc'], default='sc',
                             help="Source for live video: 'ca' (camera) or 'sc' (screen).", metavar='')

    return parser.parse_args()

def main():
    args = parse_arguments()

    # Set up logging
    logging.basicConfig(level=logging.INFO, format='%(message)s')

    # Ensure a subcommand is provided
    if not args.command:
        print("Error: No subcommand provided. Use -h for help.", file=sys.stderr)
        sys.exit(1)

    # Parse resolution if specified
    target_width, target_height = None, None
    if args.resolution:
        try:
            target_width, target_height = map(int, args.resolution.lower().split('x'))
            logging.info(f"Requested target resolution: {target_width}x{target_height}.")
        except ValueError:
            logging.error("Error: Resolution must be in the format WIDTHxHEIGHT.")
            sys.exit(1)

    # Initialize MediaPipe Selfie Segmentation if background removal is enabled
    if args.background_removal:
        try:
            import mediapipe as mp
            selfie_segmentation_instance = mp.solutions.selfie_segmentation.SelfieSegmentation(model_selection=1)
            logging.info("MediaPipe Selfie Segmentation initialized.")
            # Set the global selfie_segmentation in pixel_conversion.py
            set_selfie_segmentation(selfie_segmentation_instance)
        except ImportError:
            logging.error("Error: MediaPipe is not installed. Please install it using 'pip install mediapipe'.")
            sys.exit(1)
    else:
        # Ensure the global selfie_segmentation is None if not removing background
        set_selfie_segmentation(None)

    # Dispatch to the appropriate processing function based on subcommand
    if args.command == 'image':
        process_image(
            input_path=args.input_path,
            brightness=args.brightness,
            contrast=args.contrast,
            lightness=args.lightness,
            vibrancy=args.vibrancy,
            gamma=args.gamma,
            size=args.size,
            palette_size=args.palette_size,
            background_removal=args.background_removal,
            target_width=target_width,
            target_height=target_height,
            black_white=args.black_white
        )
    elif args.command == 'video':
        process_video(
            input_path=args.input_path,
            fps=args.fps,
            brightness=args.brightness,
            contrast=args.contrast,
            lightness=args.lightness,
            vibrancy=args.vibrancy,
            gamma=args.gamma,
            size=args.size,
            palette_size=args.palette_size,
            background_removal=args.background_removal,
            target_width=target_width,
            target_height=target_height,
            black_white=args.black_white,
        )
    elif args.command == 'live':
        process_live_video(
            fps=args.fps,
            brightness=args.brightness,
            contrast=args.contrast,
            lightness=args.lightness,
            vibrancy=args.vibrancy,
            gamma=args.gamma,
            size=args.size,
            palette_size=args.palette_size,
            background_removal=args.background_removal,
            target_width=target_width,
            target_height=target_height,
            black_white=args.black_white,
            source=args.source
        )
    else:
        logging.error("Error: Unknown command.")
        sys.exit(1)

    # Close the segmentation model if initialized
    if args.background_removal:
        if 'selfie_segmentation_instance' in locals() and selfie_segmentation_instance:
            selfie_segmentation_instance.close()
            logging.info("MediaPipe Selfie Segmentation closed.")

if __name__ == "__main__":
    main()