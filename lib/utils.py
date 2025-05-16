# Standard imports
import time
import logging
import os
import datetime

# External imports
from astropy.io import fits
from skimage.transform import downscale_local_mean
import cv2
import matplotlib.pyplot as plt
import numpy as np

# Custom imports
from lib.common import get_dt, current_timestamp

# Initialize logging
log = logging.getLogger('pueo')


def timed_function(func, *args, **kwargs):
    """Measures the execution time of a function and prints the duration.

    Args:
        func (callable): The function to be executed and timed.
        *args: Positional arguments to be passed to the function.
        **kwargs: Keyword arguments to be passed to the function.

    Returns:
        The result of the function `func` after execution.
    """
    t0 = time.monotonic()
    print(f"Running: {func.__name__}: ")
    result = func(*args, **kwargs)
    exec_time = time.monotonic() - t0
    print(f"  Executed {func.__name__} in {exec_time:.3f} seconds")
    return result, exec_time

def box_plot_compare(data1, data2, title="box plot"):
    """Generates a comparative box plot of two sets of data.

    Parameters:
    data1 (list or array-like): Data for the first box plot.
    data2 (list or array-like): Data for the second box plot.
    title (str): Title for the plot (default: "box plot").

    Returns:
    None (Generates and saves the plot as a PNG file).
    """
    fig, ax = plt.subplots(1, 2)
    ax[0].boxplot(data1)
    ax[0].set_xlabel('Original')
    ax[0].set_xticks([])
    ax[0].set_ylabel('Lengths')

    ax[1].boxplot(data2)
    ax[1].set_xlabel('Filtred')
    ax[1].set_xticks([])
    ax[1].set_ylabel('Lengths')

    fig.suptitle(title)
    plt.savefig("./partial_results/"+title+'.png')


def read_fits(image_path: str) -> np.ndarray:
    """Read image data from FITS file, and normalize it.

    Args:
        image_path (str): path to input image

    Returns:
        np.ndarray: image data in 1D array
    """

    with fits.open(image_path) as hdul:
        # find pixel min and max values
        min_val = int(hdul[0].header['CBLACK'])
        max_val = int(hdul[0].header['CWHITE'])

        # read data array
        img = hdul[0].data

    return img


def read_image_grayscale(image_path: str) -> np.ndarray:
    """Load JPGs, PNGs and FITs in grayscale

    Args:
        image_path (str): path to input image

    Returns:
        np.ndarray: image array
    """
    if os.path.splitext(image_path)[1] == ".FIT":
        img = read_fits(image_path)
    elif os.path.splitext(image_path)[1] == ".tiff":
        img = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
    else:
        img = cv2.imread(image_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    return img


def read_image_BGR(image_path: str) -> np.ndarray:
    """Load JPGs, PNGs and FITs in BGR

    Args:
        image_path (str): path to input image

    Returns:
        np.ndarray: image array
    """
    if os.path.splitext(image_path)[1] == ".FIT":
        img = read_fits(image_path)
        # Normalize data
        img = ((img / 65535.0) * 255).astype(np.uint8)
        # Create a BGR image
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    elif os.path.splitext(image_path)[1] in [".PNG", ".png", ".jpg", ".JPEG"]:
        img = cv2.imread(image_path, cv2.IMREAD_COLOR)
    elif os.path.splitext(image_path)[1] == ".tiff":
        img = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
        # Scale the 16-bit values to 8-bit (0-255) range
        scaled_data = ((img / 65535.0) * 255).astype(np.uint8)
        # Create a BGR image
        img = cv2.merge([scaled_data, scaled_data, scaled_data])
    else:
        img = cv2.imread(image_path, cv2.IMREAD_COLOR)
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

    return img


def display_image(img: np.ndarray, window_name: str) -> None:
    """Display image using openCV with a fixed width and variable height.

    Args:
        img (np.ndarray): input image array.
        window_name (str): window name.
    """
    # Fixed width for the window
    fixed_width = 800

    # Calculate the corresponding height based on the image's aspect ratio
    aspect_ratio = img.shape[0] / img.shape[1]  # height / width
    variable_height = int(fixed_width * aspect_ratio)

    # Create window with keep ratio flag
    cv2.namedWindow(window_name, cv2.WINDOW_KEEPRATIO)
    cv2.resizeWindow(window_name, fixed_width, variable_height)

    while cv2.getWindowProperty(window_name, cv2.WND_PROP_VISIBLE) > 0:
        cv2.imshow(window_name, img)
        
        # wait for 'esc' key to exit the loop
        keyCode = cv2.waitKey(50)
        if keyCode == 27:  # ESC key to close
            break

    cv2.destroyAllWindows()

def reparse_timestamp(timestamp_str):
    """
    Reparses a timestamp string in the format "YYMMDD_HHMMSS.ffffff"
    into a datetime object and formats it as "YYYY-MM-DD HH:MM:SS.fff".

    Args:
    timestamp_str: The input timestamp string.

    Returns:
    The reformatted timestamp string.
    """
    try:
        dt = datetime.datetime.strptime(timestamp_str, "%y%m%d_%H%M%S.%f")
        return dt.strftime("%Y-%m-%d %H:%M:%S.%f")[:23]
    except ValueError:
        # print(f"Invalid timestamp format: {timestamp_str}")
        return timestamp_str

def overlay_raw(img, downscale_factors, message):
    """
    Add an overlay to the image with a timestamp and a message.

    Args:
        img (numpy.ndarray): Input image.
        downscale_factors (tuple): Downscale factors (ds_x, ds_y) for adjustments.
        message (str): Message to overlay on the image.

    Returns:
        numpy.ndarray: Image with overlay text.
    """
    # NO, want to change original image -> Create a copy to preserve the original image

    overlay_image = img

    ds_x, ds_y = downscale_factors

    # Generate the timestamp string
    timestamp_string = current_timestamp()
    timestamp = f"Timestamp: {timestamp_string}"

    # Font properties
    font = cv2.FONT_HERSHEY_SIMPLEX
    green = (0, 255, 0)
    white = (255, 255, 255)
    font_color = green      # Green text
    shadow_color = (0, 0, 0)      # Black shadow for readability
    font_thickness = 1  # was 2
    line_type = cv2.LINE_AA

    # Initial font size and dynamic adjustment
    fontsize = 1 / (ds_y / 3)
    img_fraction = 0.25  # Target text width as a fraction of image width
    text_size, _ = cv2.getTextSize(timestamp, font, fontsize, font_thickness)
    while text_size[0] < img_fraction * img.shape[1]:
        fontsize += 1 / (ds_y)  # / 2
        text_size, _ = cv2.getTextSize(timestamp, font, fontsize, font_thickness)

    # Dynamic placement
    text_x = int(0.02 * img.shape[1])  # 2% from left edge
    text_y = int(0.1 * img.shape[0])  # 10% from top
    line_spacing = int(35 / ds_y)

    # Overlay timestamp with shadow
    cv2.putText(overlay_image, timestamp, (text_x + 2, text_y + 2), font, fontsize, shadow_color, font_thickness, line_type)
    cv2.putText(overlay_image, timestamp, (text_x, text_y), font, fontsize, font_color, font_thickness, line_type)

    # Update Y-coordinate for the next line of text
    text_y += text_size[1] + line_spacing

    # Overlay message with shadow
    cv2.putText(overlay_image, message, (text_x + 2, text_y + 2), font, fontsize, shadow_color, font_thickness, line_type)
    cv2.putText(overlay_image, message, (text_x, text_y), font, fontsize, font_color, font_thickness, line_type)

    return overlay_image


def display_overlay_info(img, timestamp_string, astrometry, omega, display=True, image_filename=None, final_path='./output', partial_results_path="./partial_results", scale_factors=(8, 8), resize_mode='downscale', png_compression=0, is_save=True):
    """Displays text overlay information about astrometry on output image.
    """

    # Normalize back to uint8 if the image is float
    if np.issubdtype(img.dtype, np.uint16):
        img = np.uint8(img / 256)

    print_img_info(img, 'Final')
    # display_image(img, 'Final')
    overlay_image = convert_to_3d(img) # .copy()
    timestamp_fmt = reparse_timestamp(timestamp_string)  # "250106_050111.929213" --> "2025-01-06 05:01:11.929"
    timestamp = f"Timestamp: {timestamp_fmt}"
    # font properties and color
    font = cv2.FONT_HERSHEY_SIMPLEX

    font_color = (0, 255, 0)  # Bright green color in BGR format (OpenCV)
    font_thickness = 2
    line_type = cv2.LINE_AA
    # Calculate text size to position it properly
    fontsize = 1  # starting font size
    img_fraction = 0.25 # portion of image width you want text width to be
    text_size, _ = cv2.getTextSize(timestamp, font, fontsize, font_thickness)
    while text_size[0] < img_fraction*img.shape[0]:
        fontsize += 1
        text_size, _ = cv2.getTextSize(timestamp, font, fontsize, font_thickness)
    text_x = 40
    text_y = 80
    line_spacing = 35
    # overlay timestamp
    cv2.putText(overlay_image, timestamp, (text_x, text_y), font, fontsize, font_color, font_thickness, line_type)
    text_y += text_size[1] + line_spacing  # Adjust Y-coordinate for the next text

    solver = astrometry['solver'] if astrometry else ''
    if astrometry and astrometry.get('RA', 0):
        # astrometry info
        ra = astrometry['RA']
        dec = astrometry['Dec']
        roll = astrometry['Roll']

        # body rates info
        omegax = omega[0]
        omegay = omega[1]
        omegaz = omega[2]

        astrometric_position = f"Astrometric Position ({solver}): ({ra:.4f}, {dec:.4f}, {roll:.4f}) deg"
        _rmse = astrometry['Cross-Boresight RMSE'] if 'Cross-Boresight RMSE' in astrometry else astrometry['RMSE']
        # rmse = f"RMSE: {_rmse/3600.0:.4E} deg"
        rmse = f"RMSE: {_rmse:.4E} arcsec"
        velocity = f"Angular velocity: (omegax, omegay, omegaz) = ({omegax:.4E}, {omegay:.4E}, {omegaz:.4E}) deg/s"
        probability_of_false_positive = f"Probability of False Positive: {astrometry['Prob']:.4E}"
        exec_time = (
            f"Execution time: {astrometry['total_exec_time']:.3f} s, Tetra3: {astrometry['tetra3_exec_time']:.3f} s"
        )

        # overlay text
        cv2.putText(overlay_image, astrometric_position, (text_x, text_y), font, fontsize, font_color, font_thickness, line_type)
        text_y += text_size[1] + line_spacing
        cv2.putText(overlay_image, rmse, (text_x, text_y), font, fontsize, font_color, font_thickness, line_type)
        text_y += text_size[1] + line_spacing
        cv2.putText(overlay_image, velocity, (text_x, text_y), font, fontsize, font_color, font_thickness, line_type)
        text_y += text_size[1] + line_spacing
        cv2.putText(overlay_image, probability_of_false_positive, (text_x, text_y), font, fontsize, font_color, font_thickness, line_type)
        text_y += text_size[1] + line_spacing
        cv2.putText(overlay_image, exec_time,(text_x, text_y), font, fontsize, font_color, font_thickness, line_type)
        text_y += text_size[1] + line_spacing

        # Add legend for the circles
        cv2.putText(overlay_image, 'Legend:',(text_x, text_y), font, fontsize, font_color, font_thickness, line_type)
        text_y += int(text_size[1] + line_spacing * 1.3)
        # OpenCV uses BGR
        color_blue = (255, 0, 0)
        color_green = (0, 255, 0)
        color_red = (0, 0, 255)
        colors = [(color_red, 0.6, 'Detected source'), (color_blue, 0.8, 'Valid source'), (color_green, 1.0, 'Matched star')]
        # print(f'Text suze: {text_size}')
        sources_radius = overlay_image.shape[0] / 80
        for color, radius_scaling, description in colors:
            text_cy = text_y - int(text_size[1] / 2) # Move center to text_size/2 up, since text_y is bottom of the text
            cv2.circle(overlay_image, (text_x + int(2*sources_radius), text_cy), int(sources_radius*radius_scaling), color, 4)
            cv2.putText(overlay_image, description, (text_x+int(4*sources_radius), text_y), font, fontsize, font_color, font_thickness, line_type)
            text_y += int(text_size[1] + line_spacing * 1.3)

    else:
        cv2.putText(overlay_image, f"Not Solved ({solver})", (text_x, text_y), font, fontsize, font_color, font_thickness, line_type)
        text_y += text_size[1] + line_spacing  # Adjust Y-coordinate for the next text

    # Save image
    # foi_name = f"./partial_results/Final_overlay_image_{timestamp_string}.png"
    foi_filename = os.path.join(partial_results_path, f"Final_overlay_image_{timestamp_string}.png")
    foi_scaled_filename = None
    foi_scaled_shape = None
    if image_filename is not None:
        path, filename, extension = split_path(image_filename)
        foi_filename = f"{final_path}/Final_overlay_image_{filename}_{timestamp_string}.png"
        if is_save:
            t0 = time.perf_counter()
            # Save using png compression
            cv2.imwrite(foi_filename, overlay_image, [int(cv2.IMWRITE_PNG_COMPRESSION), png_compression])
            # Get file size in MB (converted from bytes)
            file_size = os.path.getsize(foi_filename) / (1024 * 1024)  # bytes to MB conversion
            log.debug(f'Saved: {foi_filename} compression: {png_compression} file_size: {file_size:.2f} Mb in {get_dt(t0)}.')

        # TODO: An overlay image, do we use this image for showing on the GUI?
        foi_scaled_filename = f"{final_path}/Final_overlay_image_{filename}_{timestamp_string}_downscaled.png"
        foi_scaled_shape = image_resize(overlay_image, scale_factors, foi_scaled_filename, resize_mode=resize_mode)

    # Display image
    if display:
        display_image(overlay_image, "Output overlay image")

    return foi_filename, overlay_image.shape, foi_scaled_filename, foi_scaled_shape


def print_img_info(img, msg='img'):
    return
    print(f'  Image size: {msg} {img.shape[0]}x{img.shape[1]} {img.dtype} {img.size} ndim: {img.ndim}')

def convert_to_3d(img):
    """
    Converts a 2D grayscale image into a 3D image with three identical channels (e.g., RGB/BGR).
    Args:
        img: 2D numpy array (grayscale image).
    Returns:
        3D numpy array with three identical channels.
    """
    if img.ndim == 2:  # Check if the image is grayscale
        # Create a 3-channel image
        img_3d = cv2.merge([img, img, img])
        return img_3d
    else:
        # The image is already 3D
        return img

def image_downscale_orig(img, downscale_factors, image_filename, overlay=None, downscale_mode = 'resize'):
    t0 = time.monotonic()
    log.debug(f'downscale_factors: {downscale_factors} mode: {downscale_mode}')

    print_img_info(img, 'orig')
    img = convert_to_3d(img)
    # display_image(img, "Output orig image")
    if downscale_mode == 'downscale_local_mean':
        # Convert each element of the tuple to an integer
        downscale_factors_int = tuple(int(x) for x in downscale_factors)
        try:
            # Process multi-channel images
            if img.ndim == 3:  # Color image (RGB or BGR)
                downscaled_img = np.stack([
                    downscale_local_mean(img[:, :, channel], downscale_factors_int)
                    for channel in range(img.shape[2])
                ], axis=-1)
            else:  # Single-channel grayscale image
                downscaled_img = downscale_local_mean(img, downscale_factors_int)
        except ValueError as e:
            log.warning(f'Downscale failed, scaling as {min(downscale_factors_int)}')
            downscaled_img = downscale_local_mean(img, min(downscale_factors_int))
    elif downscale_mode == 'resize':
        # TODO: Add code to resize the image by factors
        pass
    # Normalize back to uint8 if the image is float
    if np.issubdtype(downscaled_img.dtype, np.floating):
        downscaled_img = (downscaled_img * 255 / downscaled_img.max()).astype(np.uint8)

    if overlay is not None:
        log.debug(f'Adding an overlay over the downscaled image: {overlay}')
        downscaled_img = overlay_raw(downscaled_img, downscale_factors, overlay)

        # display_image(downscaled_img, "Output overlay image")

    print_img_info(downscaled_img, 'orig')
    # display_image(downscaled_img, "Output downscaled image")
    cv2.imwrite(image_filename,downscaled_img)
    log.debug(f'Saved downscaled image to sd path: {image_filename} in {get_dt(t0)}.')


def image_resize(img, scale_factors, image_filename, overlay=None, resize_mode='downscale', png_compression=0):
    """
    Resize an image using either downscaling (with interpolation or local mean) or downsampling.

    Args:
        img (numpy.ndarray): The input image as a NumPy array.
        scale_factors (tuple): A tuple of (height_factor, width_factor) for resizing.
        image_filename (str): The filename to save the resized image.
        overlay (numpy.ndarray, optional): An optional overlay image to apply after resizing.
        resize_mode (str, optional): The resizing mode. Options are:
            - 'downscale': Uses interpolation (cv2.INTER_AREA) for high-quality downscaling.
            - 'downsample': Selects every nth pixel for fast downsampling.
            - 'local_mean': Uses local mean averaging for downscaling.
            Default is 'downscale'.

    Returns:
        tuple: A tuple representing the image dimensions.
               For grayscale images: (height, width)
               For color images (BGR): (height, width, channels)
               Returns None if the image cannot be loaded.

    Notes:
        - 'downscale' mode uses cv2.INTER_AREA interpolation, which is ideal for reducing image size
          while preserving sharpness and avoiding aliasing.
        - 'downsample' mode simply selects every nth pixel, which is faster but may introduce aliasing.
        - 'local_mean' mode uses skimage's downscale_local_mean, which averages pixel values in local
          neighborhoods to produce a smooth downscaled image.
    """
    t0 = time.monotonic()
    log.debug(f'scale_factors: {scale_factors} mode: {resize_mode}')

    print_img_info(img, 'orig')
    img = convert_to_3d(img)

    if resize_mode == 'local_mean':
        # Convert each element of the tuple to an integer
        scale_factors_int = tuple(int(x) for x in scale_factors)
        try:
            # Process multi-channel images
            if img.ndim == 3:  # Color image (RGB or BGR)
                resized_img = np.stack([
                    downscale_local_mean(img[:, :, channel], scale_factors_int)
                    for channel in range(img.shape[2])
                ], axis=-1)
            else:  # Single-channel grayscale image
                resized_img = downscale_local_mean(img, scale_factors_int)
        except ValueError as e:
            log.warning(f'Downscale failed, scaling as {min(scale_factors_int)}')
            resized_img = downscale_local_mean(img, min(scale_factors_int))
    elif resize_mode == 'downscale':
        # Calculate the new dimensions based on the scale factors
        new_height = int(img.shape[0] / scale_factors[0])
        new_width = int(img.shape[1] / scale_factors[1])
        new_dimensions = (new_width, new_height)

        # Resize the image using OpenCV's resize function with INTER_AREA interpolation
        resized_img = cv2.resize(img, new_dimensions, interpolation=cv2.INTER_AREA)
        log.debug(f'resized: {img.shape} -> {resized_img.shape}')
    elif resize_mode == 'downsample':
        # Downsample by selecting every nth pixel
        resized_img = img[::int(scale_factors[0]), ::int(scale_factors[1])]
    else:
        raise ValueError(f"Invalid resize_mode: {resize_mode}. Choose 'downscale', 'downsample', or 'local_mean'.")

    # Normalize back to uint8 if the image is float
    if np.issubdtype(resized_img.dtype, np.floating):
        resized_img = (resized_img * 255 / resized_img.max()).astype(np.uint8)

    if overlay is not None:
        log.debug(f'Adding an overlay over the resized image: {overlay}')
        resized_img = overlay_raw(resized_img, scale_factors, overlay)

    print_img_info(resized_img, 'resized')


    cv2.imwrite(image_filename, resized_img, [int(cv2.IMWRITE_PNG_COMPRESSION), png_compression])
    # Get file size in MB (converted from bytes)
    file_size = os.path.getsize(image_filename) / (1024 * 1024)  # bytes to MB conversion
    log.debug(f'Saved resized image to sd path: {image_filename} compression: {png_compression} file_size: {file_size:.2f} Mb in {get_dt(t0)}.')

    return resized_img.shape

def save_raws(img, ssd_path="", sd_card_path="", image_name="", scale_factors=(8, 8), resize_mode='downscale', png_compression=0, is_save=True):
    # Save original image to ssd
    img1 = img  # .copy()
    image_filename = f"{ssd_path}/{image_name}-raw.png"
    t0 = time.monotonic()
    # The IMWRITE_PNG_COMPRESSION - 0: The compression level (0 = no compression, 9 = maximum compression), see config.ini
    # Only save files in flight mode
    if is_save:
        cv2.imwrite(image_filename, img1, [int(cv2.IMWRITE_PNG_COMPRESSION), png_compression])
    log.debug(f'Saved original image to ssd path: {image_filename} in {get_dt(t0)}.')

    # Save downscaled image to sd card
    # downscale_factors = (8, 8)
    image_resized_filename = f"{sd_card_path}/{image_name}-raw-ds.png"
    image_resized_shape = image_resize(img1, scale_factors, image_resized_filename, 'Raw Image', resize_mode=resize_mode)

    return image_filename, img.shape, image_resized_filename, image_resized_shape


def get_files(path: str, extension='.png') -> list:
    """
    Gets a list of all .png files in a given directory.

    Args:
    path: The path to the directory.

    Returns:
    A list of file paths.
    """

    files = []
    for file in os.listdir(path):
        if file.endswith(extension):
            files.append(os.path.join(path, file))
    return files


def split_path(filename: str) -> tuple:
    """
    Splits a filename into its path, filename, and extension components.

    Args:
    filename: The full path to the file.

    Returns:
    A tuple containing the path, filename, and extension.
    """
    path, filename = os.path.split(filename)
    filename, extension = os.path.splitext(filename)
    return path, filename, extension

if __name__ == '__main__':
    ts_orig = '250106_094041.794214'
    ts = reparse_timestamp(ts_orig)
    print(f'{ts_orig} --> {ts}')

# Last line