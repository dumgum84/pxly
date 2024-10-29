# glfy/image_processing.py

from .pxly_imports import *
from .pixel_conversion import image_to_pixel_art
from .utils import convert_to_png

def process_image(
    input_path,
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
    """Process a single image and convert it to pixel art."""
    output_dir = os.path.dirname(input_path)
    original_filename = os.path.splitext(os.path.basename(input_path))[0]

    png_image = convert_to_png(input_path, output_dir)
    if not png_image:
        return

    try:
        with Image.open(png_image) as image:
            original_w, original_h = image.size
            target_ratio = target_width / target_height if target_width and target_height else original_w / original_h
            original_ratio = original_w / original_h

            if target_width and target_height:
                if original_ratio > target_ratio:
                    new_width = target_width
                    new_height = int(target_width / original_ratio)
                else:
                    new_height = target_height
                    new_width = int(target_height * original_ratio)
            else:
                new_width, new_height = original_w, original_h

            # Convert to NumPy array and resize using OpenCV
            image_np = np.array(image)
            image_resized = cv2.resize(image_np, (new_width, new_height), interpolation=cv2.INTER_LINEAR)

            # Convert back to PIL Image
            image_resized = Image.fromarray(image_resized)

            pixel_art = image_to_pixel_art(
                image_resized, 
                brightness, contrast,
                lightness, vibrancy, gamma,
                size, palette_size, 
                background_removal,
                black_white
            )
            
            final_output = os.path.join(output_dir, f"{original_filename}_pxly.png")
            pixel_art.save(final_output)
            logging.info(f"Pixel art image saved to {final_output}.")

    except Exception as e:
        logging.error(f"Error processing image: {e}")

    finally:
        if os.path.exists(png_image):
            os.remove(png_image)
