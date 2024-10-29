# pxly/live_video_processing.py

from .pxly_imports import *
from .pixel_conversion import image_to_pixel_art
from .utils import pad_frame_to_target_size

def process_live_video(
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
    black_white,
    source
):
    """Capture live video from the camera or screen and stream pixel art to a virtual camera."""
    if source == 'ca':
        cap = cv2.VideoCapture(0)  # Open the default camera
        if not cap.isOpened():
            logging.error("Error: Could not open camera.")
            return

        # Use camera's resolution if target resolution is not specified
        if not target_width or not target_height:
            target_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            target_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        # Set the desired resolution if specified
        if target_width and target_height:
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, target_width)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, target_height)

        try:
            with pyvirtualcam.Camera(width=target_width,
                                     height=target_height,
                                     fps=fps,
                                     fmt=PixelFormat.BGR) as vcam:
                logging.info(f'Virtual camera device: {vcam.device}')
                running = True
                frame_queue = deque(maxlen=1)  # Create a frame queue with max length of 1

                def read_frames():
                    """Thread for reading frames from the camera."""
                    nonlocal running
                    while running:
                        ret, frame = cap.read()
                        if not ret:
                            logging.error("Error: Failed to capture video frame.")
                            break
                        # Add frame to the queue, replacing the old frame if full
                        frame_queue.append(frame)

                # Start the frame reading thread
                read_thread = threading.Thread(target=read_frames)
                read_thread.start()

                try:
                    while running:
                        if not frame_queue:
                            time.sleep(0.01)
                            continue  # Wait for frames to be available

                        # Get the latest frame from the queue
                        frame = frame_queue.pop()

                        # Convert the frame to RGB
                        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                        image = Image.fromarray(frame_rgb)

                        # Convert to pixel art
                        pixel_art = image_to_pixel_art(
                            image, 
                            brightness, contrast,
                            lightness, vibrancy, gamma,
                            size, palette_size, 
                            background_removal,
                            black_white
                        )

                        # Convert the pixel art image back to an OpenCV image
                        pixel_frame = cv2.cvtColor(np.array(pixel_art), cv2.COLOR_RGB2BGR)

                        # Pad the pixel frame to match the target resolution
                        pixel_frame = pad_frame_to_target_size(pixel_frame, target_width, target_height)

                        # Send the frame to the virtual camera
                        vcam.send(pixel_frame)
                        vcam.sleep_until_next_frame()

                except KeyboardInterrupt:
                    logging.info("Virtual camera streaming interrupted by user.")
                    running = False

                finally:
                    running = False
                    read_thread.join()  # Wait for the frame reading thread to finish
                    cap.release()  # Release the camera
                    logging.info("Virtual camera streaming stopped. Exiting script.")

        except pyvirtualcam.CameraDeviceError as e:
            logging.error(f"Virtual camera error: {e}")

    elif source == 'sc':
        # Determine screen dimensions
        if target_width and target_height:
            screen_width = target_width
            screen_height = target_height
        else:
            with mss.mss() as sct_temp:
                monitor_settings = sct_temp.monitors[1]
                screen_width = monitor_settings["width"]
                screen_height = monitor_settings["height"]

        try:
            with pyvirtualcam.Camera(width=screen_width, height=screen_height, fps=fps, fmt=PixelFormat.BGR) as vcam:
                logging.info(f'Virtual camera device: {vcam.device}')
                running = True
                frame_queue = deque(maxlen=1)  # Create a frame queue with max length of 1

                def read_frames():
                    """Thread for reading frames from the screen."""
                    nonlocal running
                    try:
                        with mss.mss() as sct:
                            monitor_settings = sct.monitors[1]  # Capture the primary monitor
                            while running:
                                img = sct.grab(monitor_settings)
                                frame = np.array(img)
                                frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)
                                frame_queue.append(frame)
                    except Exception as e:
                        logging.error(f"Error in read_frames thread: {e}")
                        running = False

                # Start the frame reading thread
                read_thread = threading.Thread(target=read_frames, daemon=True)
                read_thread.start()

                try:
                    while running:
                        if not frame_queue:
                            time.sleep(0.01)
                            continue  # Wait for frames to be available

                        # Get the latest frame from the queue
                        frame = frame_queue.pop()

                        # Convert the frame to RGB
                        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                        image = Image.fromarray(frame_rgb)

                        # Convert to pixel art
                        pixel_art = image_to_pixel_art(
                            image, 
                            brightness, contrast,
                            lightness, vibrancy, gamma,
                            size, palette_size, 
                            background_removal,
                            black_white
                        )

                        # Convert the pixel art image back to an OpenCV image
                        pixel_frame = cv2.cvtColor(np.array(pixel_art), cv2.COLOR_RGB2BGR)

                        # Pad the pixel frame to match the target resolution
                        pixel_frame = pad_frame_to_target_size(pixel_frame, screen_width, screen_height)

                        # Send the frame to the virtual camera
                        vcam.send(pixel_frame)
                        vcam.sleep_until_next_frame()

                except KeyboardInterrupt:
                    logging.info("Virtual camera streaming interrupted by user.")
                    running = False

                except Exception as e:
                    logging.error(f"An error occurred: {e}")
                    running = False

                finally:
                    running = False
                    read_thread.join()  # Wait for the frame reading thread to finish
                    logging.info("Live pixel art streaming stopped. Exiting script.")

        except pyvirtualcam.CameraDeviceError as e:
            logging.error(f"Virtual camera error: {e}")
