# Standard imports
import time
import logging
import os
import datetime
from datetime import timezone
from pathlib import Path
from typing import Union, List, Tuple
from contextlib import suppress
from typing import Optional
import platform
import errno
import shutil

# External imports
from astropy.io import fits
from skimage.transform import downscale_local_mean
import cv2
import matplotlib.pyplot as plt
import numpy as np

# Custom imports
from lib.common import get_dt, current_timestamp, logit


class Utils:
    """Utils, helper functions"""

    camera_settings = {}
    meta = {}

    def __init__(self, cfg, logger: Optional[logging.Logger] = None, server=None):
        self.cfg = cfg
        self.log = logger or logging.getLogger("pueo")
#        self.server = server
        self._system = platform.system().lower()
        self._is_windows = self._system == "windows"
        self._is_linux = self._system == "linux"

    @staticmethod
    def get_current_utc_timestamp() -> Tuple[datetime.datetime, datetime.datetime, float]:
        """
        Retrieves the current Coordinated Universal Time (UTC) as a Unix timestamp (seconds since epoch).

        The function ensures the resulting datetime object is explicitly set to UTC
        before converting it to a timestamp, guaranteeing accuracy regardless of
        the system's local timezone settings.

        Returns:
            float: The current UTC timestamp (seconds since 1970-01-01 00:00:00 UTC).
        """
        # Get the current time, automatically set to the system's timezone.
        dt = datetime.datetime.now(timezone.utc)

        # 🌟 Refinement: Ensure the timezone is explicitly set to UTC before getting the timestamp.
        # While datetime.datetime.now(timezone.utc) already returns a timezone-aware object,
        # this pattern is robust and often necessary for older or complex datetime creation paths.
        utc_time = dt.replace(tzinfo=timezone.utc)

        curr_utc_timestamp = utc_time.timestamp()
        return dt, utc_time, curr_utc_timestamp

    @staticmethod
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

    @staticmethod
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
        plt.close()

    @staticmethod
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

    def read_image_grayscale(self, image_path: str) -> np.ndarray:
        """Load JPGs, PNGs and FITs in grayscale

        Args:
            image_path (str): path to input image

        Returns:
            np.ndarray: image array
        """
        if os.path.splitext(image_path)[1] == ".FIT":
            img = self.read_fits(image_path)
        elif os.path.splitext(image_path)[1] == ".tiff":
            img = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
        else:
            img = cv2.imread(image_path)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        return img

    def read_image_BGR(self, image_path: str, pixel_well_depth=14) -> np.ndarray:
        """Load JPGs, PNGs and FITs in BGR

        Args:
            image_path (str): path to input image

        Returns:
            np.ndarray: image array
        """
        scale_factor = float(2 ** pixel_well_depth) - 1
        if os.path.splitext(image_path)[1] == ".FIT":
            img = self.read_fits(image_path)
            # Normalize data
            img = ((img /scale_factor) * 255).astype(np.uint8)
            # Create a BGR image
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        elif os.path.splitext(image_path)[1] in [".PNG", ".png", ".jpg", ".JPEG"]:
            img = cv2.imread(image_path, cv2.IMREAD_COLOR)
        elif os.path.splitext(image_path)[1] == ".tiff":
            img = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
            # Scale the 16-bit values to 8-bit (0-255) range
            scaled_data = ((img / scale_factor) * 255).astype(np.uint8)
            # Create a BGR image
            img = cv2.merge([scaled_data, scaled_data, scaled_data])
        else:
            img = cv2.imread(image_path, cv2.IMREAD_COLOR)
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

        return img

    @staticmethod
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

    @staticmethod
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


    def overlay_raw(self, img, downscale_factors, message):
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
        img_fraction = 0.50  # Target text width as a fraction of image width
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

        # --- Optional: draw ROI overlay(s) published by SourceFinder ---
        self.overlay_rois(overlay_image)

        return overlay_image

    def overlay_rois(self, image, is_inspection=False):
        """
        Draw ROI outlines on the input image.

        This function renders two independent Region-of-Interest (ROI) outlines:
          1) A centered rectangular ROI defined by fractional width/height
             configuration parameters.
          2) A vertically clipped circular ROI. The clipped circle is rendered
             as a fully closed outline consisting of:
               - Left and right circular arcs
               - Horizontal closure segments at the top and bottom clip limits

        The geometry is computed analytically to avoid rasterization and clipping
        artifacts. Only outline geometry is drawn:
          - No filled regions
          - No shading or transparency
          - No construction or guidelines

        Parameters
        ----------
        image : np.ndarray
            Input image in BGR format. The image is modified in place.
        is_inspection : bool
            Used to differentiate colors for inspection images
        """

        if (not self.cfg.overlay_rois or
                is_inspection and not self.cfg.overlay_rois_inspection):
            return

        try:
            h, w = image.shape[:2]

            if is_inspection:
                rect_color = (255, 0, 0)  # blue
                oval_color = (127, 255, 0)  # mix
            else:
                rect_color = (255, 0, 0)   # blue
                oval_color = (0, 255, 255) # yellow
            thickness = 2

            # ------------------------------------------------------------------
            # ROI 1: centered rectangle
            # ------------------------------------------------------------------
            fx = self.cfg.roi_keep_frac_x
            fy = self.cfg.roi_keep_frac_y

            rw = max(1, int(round(w * fx)))
            rh = max(1, int(round(h * fy)))
            x0 = (w - rw) // 2
            y0 = (h - rh) // 2
            x1 = x0 + rw
            y1 = y0 + rh

            cv2.rectangle(image, (x0, y0), (x1, y1),
                          rect_color, thickness, cv2.LINE_AA)

            # ------------------------------------------------------------------
            # ROI 2: clipped circle outline (fully closed)
            # ------------------------------------------------------------------
            diam_frac_w = self.cfg.roi_circle_diam_frac_w
            strip_frac_y = self.cfg.roi_strip_frac_y

            cx, cy = w // 2, h // 2
            r = int(round(0.5 * diam_frac_w * w))

            y_top = int(round(strip_frac_y * h))
            y_bot = int(round((1.0 - strip_frac_y) * h))

            # --- Draw left/right arcs as polyline ---
            pts = []
            for y in range(y_top, y_bot + 1):
                dy = y - cy
                v = r * r - dy * dy
                if v <= 0:
                    continue
                dx = int(round(np.sqrt(v)))
                pts.append((cx - dx, y))
                pts.append((cx + dx, y))

            # Sort to form a continuous outline
            left = sorted(pts[::2], key=lambda p: p[1])
            right = sorted(pts[1::2], key=lambda p: p[1], reverse=True)
            arc_pts = np.array(left + right, dtype=np.int32)

            cv2.polylines(
                image,
                [arc_pts],
                isClosed=False,
                color=oval_color,
                thickness=thickness,
                lineType=cv2.LINE_AA,
            )

            # --- Draw horizontal closure segments ---
            for y in (y_top, y_bot):
                dy = y - cy
                v = r * r - dy * dy
                if v > 0:
                    dx = int(round(np.sqrt(v)))
                    cv2.line(
                        image,
                        (cx - dx, y),
                        (cx + dx, y),
                        oval_color,
                        thickness,
                        cv2.LINE_AA,
                    )

        except Exception as e:
            self.log.error(f"Error adding rois overlay: {e}")

    def display_overlay_info(self, img, timestamp_string, astrometry, omega, display=True, image_filename=None, final_path='./output', partial_results_path="./partial_results", scale_factors=(8, 8), resize_mode='downscale', png_compression=0, is_save=True, is_downsize=True):
        """Displays text overlay information about astrometry on output image - used on the final overlay image (foi)."""

        # Normalize back to uint8 if the image is float
        if np.issubdtype(img.dtype, np.uint16):
            img = np.uint8(img / 256)

        self.print_img_info(img, 'Final')
        # display_image(img, 'Final')
        overlay_image = self.convert_to_3d(img) # .copy()
        timestamp_fmt = self.reparse_timestamp(timestamp_string)  # "250106_050111.929213" --> "2025-01-06 05:01:11.929"
        timestamp = f"Start Timestamp: {timestamp_fmt}"
        # font properties and color
        font = cv2.FONT_HERSHEY_SIMPLEX

        font_color = (0, 255, 0)  # Bright green color in BGR format (OpenCV)
        font_thickness = 2
        line_type = cv2.LINE_AA
        # Calculate text size to position it properly
        font_size = 1  # starting font size
        img_fraction = 0.50 # portion of image width you want text width to be
        text_size, _ = cv2.getTextSize(timestamp, font, font_size, font_thickness)
        while text_size[0] < img_fraction*img.shape[0]:
            font_size += 1
            text_size, _ = cv2.getTextSize(timestamp, font, font_size, font_thickness)

        # Multiplier to make font size larger for inspection images. It is to make the overlay readable to the user.
        img_fraction *= self.cfg.foi_font_multiplier # 1.5

        text_x = 40
        text_y = 80
        line_spacing = 35
        # overlay timestamp
        cv2.putText(overlay_image, timestamp, (text_x, text_y), font, font_size, font_color, font_thickness, line_type)
        text_y += text_size[1] + line_spacing  # Adjust Y-coordinate for the next text

        solver = astrometry['solver_name'] if astrometry else ''
        if astrometry and astrometry.get('RA', 0):
            # astrometry info
            ra = astrometry.get('RA', 0.0)
            dec = astrometry.get('Dec', 0.0)
            roll = astrometry.get('Roll', 0.0)

            ra_rms = astrometry.get('RA_RMS', 0.0)
            dec_rms = astrometry.get('Dec_RMS', 0.0)
            roll_rms = astrometry.get('Roll_RMS', 0.0)

            # body rates info
            #omegax = omega[0]
            #omegay = omega[1]
            #omegaz = omega[2]

            # solved_txt = ' (Not solved)' if ra == dec == roll == 0.0 else ' (Solved)'
            astrometric_position = f"Astrometric Position ({solver}): ({ra:.4f}, {dec:.4f}, {roll:.4f}) deg"
            _rmse = astrometry.get('Cross-Boresight RMSE', 0.0) if 'Cross-Boresight RMSE' in astrometry else astrometry.get('RMSE', 0.0)
            rmse = f"RMSE: {_rmse:.4E} arcsec"
            #rmse = f"RMSE: {_rmse:.3f} arcsec, RMS: ({ra_rms:.3f}, {dec_rms:.3f}, {roll_rms:.3f}) arcesc"
            # velocity = f"Angular velocity: (omegax, omegay, omegaz) = ({omegax:.4E}, {omegay:.4E}, {omegaz:.4E}) deg/s"
            # Or even more explicitly:
            # velocity = f"Angular velocity: w_x = {omega[0]:.4E}, w_y = {omega[1]:.4E}, w_z = {omega[2]:.4E} deg/s"
            velocity = f"Angular velocity: w = ({omega[0]:.6f}, {omega[1]:.6f}, {omega[2]:.6f}) deg/s"

            probability_of_false_positive = f"Probability of False Positive: {astrometry['Prob']:.4E}"
            exec_time = f"Execution time: {astrometry['total_exec_time']:.3f} s, Solver: {astrometry['solver_exec_time']:.3f} s"

            plate_scale = astrometry.get('PlateScale', '- arcsec/px')
            exposure_time = astrometry.get('ExposureTime', '-')
            gain_cb = astrometry.get('Gain', '-')
            misc = f"Plate Scale: {plate_scale}, Exposure Time: {exposure_time}, Gain: {gain_cb}"

            # overlay text
            cv2.putText(overlay_image, astrometric_position, (text_x, text_y), font, font_size, font_color, font_thickness, line_type)
            text_y += text_size[1] + line_spacing
            cv2.putText(overlay_image, rmse, (text_x, text_y), font, font_size, font_color, font_thickness, line_type)
            text_y += text_size[1] + line_spacing
            cv2.putText(overlay_image, velocity, (text_x, text_y), font, font_size, font_color, font_thickness, line_type)
            text_y += text_size[1] + line_spacing
            cv2.putText(overlay_image, probability_of_false_positive, (text_x, text_y), font, font_size, font_color, font_thickness, line_type)
            text_y += text_size[1] + line_spacing
            cv2.putText(overlay_image, exec_time,(text_x, text_y), font, font_size, font_color, font_thickness, line_type)
            text_y += text_size[1] + line_spacing

            cv2.putText(overlay_image, misc,(text_x, text_y), font, font_size, font_color, font_thickness, line_type)
            text_y += text_size[1] + line_spacing

            # Add legend for the circles
            cv2.putText(overlay_image, 'Legend:',(text_x, text_y), font, font_size, font_color, font_thickness, line_type)
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
                cv2.putText(overlay_image, description, (text_x+int(4*sources_radius), text_y), font, font_size, font_color, font_thickness, line_type)
                text_y += int(text_size[1] + line_spacing * 1.3)

        else:
            cv2.putText(overlay_image, f"Not Solved ({solver})", (text_x, text_y), font, font_size, font_color, font_thickness, line_type)
            text_y += text_size[1] + line_spacing  # Adjust Y-coordinate for the next text

        # Add rois overlay if enabled in config via overlay_rois = True
        self.overlay_rois(overlay_image)

        # Save image
        # foi_name = f"./partial_results/Final_overlay_image_{timestamp_string}.png"
        foi_filename = os.path.join(partial_results_path, f"Final_overlay_image_{timestamp_string}.png")
        foi_scaled_filename = None
        foi_scaled_shape = None
        if image_filename is not None:
            path, filename, extension = self.split_path(image_filename)
            foi_filename = f"{final_path}/Final_overlay_image_{filename}.png" # _{timestamp_string}
            if is_save:
                t0 = time.perf_counter()
                # Save using png compression
                # Note The final overlay image can and shall only be saved as downscaled version
                if False:
                    cv2.imwrite(foi_filename, overlay_image, [int(cv2.IMWRITE_PNG_COMPRESSION), png_compression])
                    # Get file size in MB (converted from bytes)
                    # file_size = os.path.getsize(foi_filename) / (1024 * 1024)  # bytes to MB conversion
                    # log.debug(f'Saved: {foi_filename} compression: {png_compression} file_size: {file_size:.2f} Mb in {get_dt(t0)}.')
            # A downscaled overlay image used image for showing on the GUI
            foi_scaled_shape = foi_scaled_filename = None
            if is_downsize:
                # Saving final overlay image
                foi_scaled_filename = f"{final_path}/Final_overlay_image_{filename}_downscaled.png" # {timestamp_string}_
                foi_scaled_shape = self.image_resize(overlay_image, scale_factors, foi_scaled_filename, resize_mode=resize_mode)

        # Display image
        if display:
            self.display_image(overlay_image, "Output overlay image")

        return foi_filename, overlay_image.shape, foi_scaled_filename, foi_scaled_shape

    @staticmethod
    def print_img_info(img, msg='img'):
        return
        print(f'  Image size: {msg} {img.shape[0]}x{img.shape[1]} {img.dtype} {img.size} ndim: {img.ndim}')

    @staticmethod
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

    def convert_dummy_to_mono(img):
        """
        Convert dummy camera RGB image (2822x4144x3, uint8)
        to monochrome (2822x4144, uint16) like real ASI camera.

        Args:
            img (np.ndarray): Input image (either real ASI mono or dummy RGB).

        Returns:
            np.ndarray: Monochrome image in ASI format (2822x4144, uint16).
        """
        # Check if image is from dummy (3-channel RGB)
        if len(img.shape) == 3 and img.shape[2] == 3 and img.dtype == np.uint8:
            print("Detected dummy camera RGB image. Converting to monochrome...")
            # Convert RGB to grayscale (luminance formula)
            mono = np.dot(img[..., :3], [0.299, 0.587, 0.114]).astype(np.uint16)
            # Scale 8-bit (0-255) to 16-bit (0-65535) to mimic ASI RAW16
            mono = (mono * 257).astype(np.uint16)  # 255 * 257 = 65535
            return mono
        else:
            print("Image is already monochrome or not a dummy RGB.")
            return img  # Return unchanged if not dummy RGB

    def save_as_jpeg_with_stretch(self, img_16bit, save_mode, jpeg_path, quality=80, lower_percentile=1, upper_percentile=99):
        """
        Convert 16-bit monochrome ASI image to 8-bit JPEG with contrast stretching.

        Args:
            img_16bit (np.ndarray): 16-bit input image (shape: height × width).
            save_mode (str): normal | stretch
            jpeg_path (str): Output JPEG path.
            quality (int): JPEG quality (0-100).
            lower_percentile (float): Lower percentile for stretching (e.g., 1%).
            upper_percentile (float): Upper percentile for stretching (e.g., 99%).
        """
        save_mode = save_mode or "normal"
        # Check validity of save_mode
        if save_mode not in ["normal", "stretch"]:
            raise ValueError("Invalid mode")

        img_8bit = None
        if save_mode == "stretch":
            # Calculate percentiles to ignore outliers (e.g., hot pixels)
            low_val = float(np.nanpercentile(img_16bit, lower_percentile))
            high_val = float(np.nanpercentile(img_16bit, upper_percentile))

            # Guard against collapsed or NaN range
            img = img_16bit.copy().astype(np.float32, copy=False)
            den = high_val - low_val
            if not np.isfinite(den) or den <= 0.0:
                # Fallback: robust median±3*MAD window
                med = float(np.nanmedian(img))
                mad = float(np.nanmedian(np.abs(img - med)))
                lo, hi = med - 3.0 * mad, med + 3.0 * mad
                den2 = hi - lo
                if not np.isfinite(den2) or den2 <= 0.0:
                    img_8bit = np.zeros(img.shape, np.uint8)
                else:
                    img_stretched = np.clip((img - lo) * (255.0 / den2), 0, 255)
                    img_8bit = img_stretched.astype(np.uint8)
            else:
                img_stretched = np.clip((img - low_val) * (255.0 / den), 0, 255)
                img_8bit = img_stretched.astype(np.uint8)

        # Save as JPEG into .temp file then renaming to target filename

        # Extract directory path and filename
        directory = os.path.dirname(jpeg_path)
        filename = os.path.basename(jpeg_path)
        # Create temp filename by adding "temp_" prefix
        temp_filename = "temp_" + filename
        temp_path = os.path.join(directory, temp_filename)

        # Save Image
        success = True
        if save_mode == "stretch":
            self.overlay_rois(img_8bit, is_inspection=True)
            success = cv2.imwrite(temp_path, img_8bit, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
        elif save_mode == "normal":
            self.overlay_rois(img_16bit, is_inspection=True)
            success = cv2.imwrite(temp_path, img_16bit, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
        if not success:
            logit(f"Error saving image to {temp_path}", color="red")

        try:
            os.replace(temp_path, jpeg_path)
        except OSError as exc:
            # cleanup temp file and raise
            with suppress(OSError):
                os.remove(temp_path)

    def save_as_jp2(self, image, jp2path, compression_x1000=1000):
        # Define compression parameters for JPEG 2000
        compression_params = [
            cv2.IMWRITE_JPEG2000_COMPRESSION_X1000, compression_x1000,  # Compression ratio (1000 = lossless, lower values for lossy)
            # Note: Some OpenCV builds may not fully support lossless compression:cite[2]:cite[7]
        ]

        # Save the image with JPEG 2000 compression
        success = cv2.imwrite(jp2path, image, compression_params)
        if not success:
            logit(f"Failed to write the image {jp2path}.", color="red")

    def create_symlink(self, path, filename, symlink_name='last_inspection_image.jpg', use_relative_path=True):
        """
        Create or update a symlink pointing to the specified file.

        Args:
            path (str): Path to the directory where the symlink should be created
            filename (str): Full path to the target file
            symlink_name (str): Name of the symlink to create
            use_relative_path (bool): If True, use relative path for symlink (default: True)

        Returns:
            bool: True if successful, False otherwise
        """
        try:
            symlink_path = os.path.join(path, symlink_name)

            # Remove existing symlink or file
            if os.path.exists(symlink_path) or os.path.islink(symlink_path):
                os.remove(symlink_path)

            # Create parent directory if needed
            self.make_dirs_safe(path, exist_ok=True)

            if use_relative_path:
                # Calculate relative path from symlink directory to target file
                symlink_dir = os.path.abspath(path)
                target_abs_path = os.path.abspath(filename)

                # Get relative path from symlink directory to target
                relative_path = os.path.relpath(target_abs_path, symlink_dir)
                os.symlink(relative_path, symlink_path)
                self.log.debug(f"Created relative symlink '{symlink_path}' -> '{relative_path}'")
            else:
                # Use absolute path
                target_abs_path = os.path.abspath(filename)
                os.symlink(target_abs_path, symlink_path)
                self.log.debug(f"Created absolute symlink '{symlink_path}' -> '{target_abs_path}'")

            return True

        except OSError as e:
            self.log.error(f"Error creating symlink: {e}")
            return False
        except Exception as e:
            self.log.error(f"Unexpected error: {e}")
            return False

    def delete_trash(self,
        trash: Union[str, List[Tuple[float, str]]],
        ext: str = '.txt',
        keep: int = 5
    ) -> None:
        """
        Deletes the oldest files, keeping the most recent 'keep' files.
        Handles both folder paths and pre-processed file lists.

        Args:
            trash: Either:
                - A folder path (str) to scan for files with given extension, or
                - A list of tuples [(timestamp, filepath)] of files to process
            ext: File extension to filter by (if trash is a folder path)
            keep: Number of most recent files to preserve

        Behavior:
            - For folder input: Finds all files with given extension, gets their creation times,
              and deletes oldest, keeping newest 'keep' files.
            - For list input: Processes the given files directly using existing tuple logic.
        """
        if not trash:
            self.log.debug("Trash is empty. Nothing to delete.")
            return

        if keep <= 0:
            self.log.debug("Invalid keep value. Must be greater than 0.")
            return

        # Handle folder path input
        if isinstance(trash, str):
            folder_path = Path(trash)
            if not folder_path.exists():
                self.log.debug(f"Folder not found: {folder_path}")
                return

            # Create list of (creation_time, filepath) tuples
            file_list = []
            for file in folder_path.glob(f'*{ext}'):
                try:
                    ctime = file.stat().st_ctime
                    file_list.append((ctime, str(file)))
                except OSError as e:
                    self.log.debug(f"Error processing {file}: {e}")
                    continue

            trash = file_list

        # Original logic for processing file list
        trash.sort(key=lambda item: item[0])  # Sort by timestamp (oldest first)
        files_to_delete = trash[:-keep] if keep < len(trash) else []

        for timestamp, file in files_to_delete:
            try:
                os.remove(file)
                self.log.debug(f"Deleted: {file} (timestamp: {timestamp})")
            except FileNotFoundError:
                self.log.debug(f"File not found: {file}")
            except OSError as e:
                self.log.debug(f"Error deleting {file}: {e}")

    import os

    def target_filename(self, basename, target_path=None, ext=None, postfix=None):
        """
        Build a target filename from a base name with optional path, extension, and postfix.

        Examples:
            # base can be full path, or just a file name with/without extension
            # target_filename('web/test.png', 'inspection_images', '.jpg', '-histogram')
            # => 'inspection_images/test-histogram.jpg'
            # target_filename('web/test.png', None, '.jpg', '-histogram')
            # => 'web/test-histogram.jpg'

        Args:
            basename (str): Source file name or path.
            target_path (str, optional): Directory to prepend to the output filename. If None, preserve original path.
            ext (str, optional): Extension to use (e.g., '.jpg'). If None, keep original.
            postfix (str, optional): String to append before extension (e.g., '-histogram').

        Returns:
            str: Constructed filename with target path, postfix, and extension.
        """
        # Extract directory and file
        dir_name = os.path.dirname(basename)
        base_name_only = os.path.basename(basename)
        name, orig_ext = os.path.splitext(base_name_only)

        # Apply postfix if given
        if postfix:
            name += postfix

        # Determine final extension
        final_ext = ext if ext is not None else orig_ext

        # Build final filename
        filename = name + (final_ext or '')

        # Determine final path
        if target_path is not None:
            filename = os.path.join(target_path, filename)
        elif dir_name:
            filename = os.path.join(dir_name, filename)
        # else keep just the filename

        return filename

    def create_image_histogram(self, arr, basename, target_path=None, postfix="_histogram", title="Image Histogram"):
        """Create Adhoc Histogram and save it to basename

        Params:
            arr: image
            basename: filename without extension, e.g. test/test

        """
        t0 = time.monotonic()
        bins = np.linspace( 0, self.cfg.pixel_saturated_value_raw16, self.cfg.autogain_num_bins)
        counts, bins = np.histogram( arr, bins=bins)

        # TODO: Add the following three values to the plot as Gain:, Exposure:, p999:
        current_gain = self.camera_settings.get('Gain', -1)
        current_exposure_us = self.camera_settings.get('Exposure', -1)
        p999_value = self.meta.get('p999_value', -1)
        dtc = self.meta.get('curr_img_dtc', datetime.datetime.now(timezone.utc))
        dtc_fmt = dtc.strftime('%Y-%m-%dT%H:%M:%S.%f')[:-3] + 'Z'

        fig, ax = plt.subplots()

        ax.hist(
            bins[:-1],
            bins,
            weights=counts,
            histtype="stepfilled",
            linewidth=1.5,
            alpha=0.8,
            log=True
        )

        ax.set_xlabel("Pixel value [ADU]")
        ax.set_ylabel("Frequency [pixels]")

        # Bold main title using fontweight
        ax.set_title(f'{title}\n', fontweight='bold')

        # Parameter names bold, values normal
        param_line = f"$\\bf{{Gain:}}$ {current_gain}   " \
                     f"$\\bf{{Exposure:}}$ {current_exposure_us} µs   " \
                     f"$\\bf{{99.9ᵗʰ\xa0percentile:}}$ {p999_value}"
        ax.text(0.5, 1.0, param_line, transform=ax.transAxes, ha="center", va="bottom", fontsize=10, color="black")

        # Timestamp as annotation slightly right, two lines down
        ax.text(0.98, 0.96, dtc_fmt, transform=ax.transAxes, ha="right", va="top", fontsize=9, color="gray")

        ax.tick_params(direction="in", top=True, right=True)

        fig.tight_layout()
        # Save histogram PNG next to the image (same basename, different extension)
        # logit(f"Histogram plot created in {get_dt(t0)}.")
        filename = self.target_filename(basename, target_path=target_path, ext='.jpg', postfix=postfix)
        plt.savefig(
            filename,
            dpi=75,
            facecolor='w',
            edgecolor='w',
            bbox_inches='tight',
            pad_inches=0.05,
            format='jpg'
        )
        logit(f"{title} plot created as {filename} in {get_dt(t0)}.", color='yellow')
        plt.close() # Closing, dropping a plot
        return filename

    def image_resize(self, img, scale_factors, image_filename, overlay=None, resize_mode='downscale',
                     png_compression=0,
                     is_inspection=False, jpeg_settings: dict | None = None,
                     is_save_jp2=False, is_save=True):
        """
        Resize an image using either downscaling (with interpolation or local mean) or downsampling.

        Args:
            img (numpy.ndarray): The input image as a NumPy array.
            scale_factors (tuple): A tuple of (height_factor, width_factor) for resizing.
            image_filename (str): The filename to save the resized image.
            overlay (numpy.ndarray, optional): An optional overlay image to apply after resizing.
            png_compression (int): PNG Compression value for IMWRITE_PNG_COMPRESSION
            resize_mode (str, optional): The resizing mode. Options are:
                - 'downscale': Uses interpolation (cv2.INTER_AREA) for high-quality downscaling.
                - 'downsample': Selects every nth pixel for fast downsampling.
                - 'local_mean': Uses local mean averaging for downscaling.
                Default is 'downscale'.
            is_inspection (bool): Create inspection jpg file
            jpeg_settings (dict|None): jpeg settings
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
        self.log.debug(f'scale_factors: {scale_factors} mode: {resize_mode}')

        self.print_img_info(img, 'orig')
        # TODO: Why did we convert to 3d?
        # img = convert_to_3d(img)

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
                self.log.warning(f'Downscale failed, scaling as {min(scale_factors_int)}')
                resized_img = downscale_local_mean(img, min(scale_factors_int))
        elif resize_mode == 'downscale':
            # Calculate the new dimensions based on the scale factors
            new_height = int(img.shape[0] / scale_factors[0])
            new_width = int(img.shape[1] / scale_factors[1])
            new_dimensions = (new_width, new_height)

            # Resize the image using OpenCV's resize function with INTER_AREA interpolation
            msg_txt = 'resized'
            if scale_factors[0] != 1.0 or scale_factors[0] != 1.0:
                resized_img = cv2.resize(img, new_dimensions, interpolation=cv2.INTER_AREA)
            else:
                resized_img = img
                msg_txt = 'size preserved'
            self.log.debug(f'{msg_txt}: {img.shape} -> {resized_img.shape}')

        elif resize_mode == 'downsample':
            # Downsample by selecting every nth pixel
            resized_img = img[::int(scale_factors[0]), ::int(scale_factors[1])]
        else:
            raise ValueError(f"Invalid resize_mode: {resize_mode}. Choose 'downscale', 'downsample', or 'local_mean'.")

        # Normalize back to uint8 if the image is float
        if np.issubdtype(resized_img.dtype, np.floating):
            resized_img = (resized_img * 255 / resized_img.max()).astype(np.uint8)

        # Save as JPEG with 80% quality (add this line)
        # Using the inspection settings
        if is_inspection:
            jpeg_settings = jpeg_settings if jpeg_settings else {}
            images_keep = jpeg_settings.get('images_keep', 100)
            inspection_path = jpeg_settings.get('path', 'inspection_images/')
            quality = jpeg_settings.get('quality', 80)
            lower_percentile = jpeg_settings.get('lower_percentile', 1)
            upper_percentile = jpeg_settings.get('upper_percentile', 99)
            symlink_base_name = jpeg_settings.get('last_image_symlink_name', 'last_inspection_image.jpg')
            web_path = jpeg_settings.get('web_path', 'web/')

            # Ensure inspection_path exists
            self.make_dirs_safe(inspection_path, exist_ok=True)

            # Create proper path for JPEG file
            base_filename = os.path.basename(image_filename)  # Get just the filename

            # Split the symlink base name into stem + extension
            symlink_stem, symlink_ext = os.path.splitext(symlink_base_name)

            jp2_basename = base_filename.replace(".png", ".jp2")
            jp2_filename = os.path.join(inspection_path, jp2_basename)  # Full path

            # The normal mode would save the image as jpg, but jpg only supports 8-bit, so image gets useless.
            # Decision to only stay with stretch image
            save_modes = ["stretch", ]  # ["normal", "stretch"]
            for save_mode in save_modes:
                jpeg_basename = base_filename.replace(".png", f"_{save_mode}.jpg")
                jpeg_filename = os.path.join(inspection_path, jpeg_basename)  # Full path
                # Clear and explicit naming
                #  -> last_inspection_image_normal.jpg or last_inspection_image_stretch.jpg
                symlink_name = f"{symlink_stem}_{save_mode}{symlink_ext}" if len(save_modes) > 1 else symlink_base_name
                self.save_as_jpeg_with_stretch(resized_img, save_mode, jpeg_filename, quality, lower_percentile, upper_percentile)
                # Create symlink to the latest image
                self.create_symlink(web_path, jpeg_filename, symlink_name)

            # TODO: Revisit using of JPEG2000 - Disabled. Requested by Windell.
            # if self.save_as_jp2 and False:
            #     self.save_as_jp2(resized_img, jp2_filename,500) # Target ~10:1 compression

            # Create inspection histogram
            histogram_base = os.path.join(inspection_path, base_filename)
            histogram_filename = self.create_image_histogram(img, histogram_base, postfix="_image_histogram", title="Raw Image Histogram")  # adds : _histogram.jpg
            self.create_symlink(web_path, histogram_filename, 'last_inspection_histogram_raw_image.jpg')

            # Cleanup
            self.delete_trash(inspection_path, ext='.jpg', keep=images_keep)

        if overlay is not None:
            self.log.debug(f'Adding an overlay over the resized image: {overlay}')
            resized_img = self.overlay_raw(resized_img, scale_factors, overlay)

        self.print_img_info(resized_img, 'resized')

        # Save as PNG
        if is_save:
            success = cv2.imwrite(image_filename, resized_img, [int(cv2.IMWRITE_PNG_COMPRESSION), png_compression])
            if not success:
                logit(f"Failed to write image {image_filename}.", color="red")

        # Create symlink for inspection image pointing to a sd_card saved RAW image.
        if is_inspection:
            # Create symlink to the latest inspection image
            symlink_base_name = jpeg_settings.get('last_image_symlink_name', 'last_inspection_image.jpg')
            symlink_name = symlink_base_name.replace(".jpg", f".png")
            web_path = jpeg_settings.get('web_path', 'web/')
            self.create_symlink(web_path, image_filename, symlink_name)

            # Create symlink to latest histogram inspction image

        # Save as JP2
        # TODO: Decide and reinspect this saving the RAW as jp2
        if is_save_jp2:
            pass
            # jp2_filename = image_filename.replace(".png", ".jp2")
            # save_as_jp2(resized_img, jp2_filename, 400)  # Target ~10:1 compression

        # Get file size in MB (converted from bytes)
        with suppress(FileNotFoundError):
            file_size = os.path.getsize(image_filename) / (1024 * 1024)  # bytes to MB conversion
            self.log.debug(f'Saved resized image to path: {image_filename} compression: {png_compression} file_size: {file_size:.2f} Mb in {get_dt(t0)}.')

        return resized_img.shape


    def save_raws(self, img, ssd_path="", sd_card_path="", image_name="",
                  scale_factors=(16, 16), resize_mode='downscale',
                  raw_scale_factors=(8,8),  raw_resize_mode='downscale',
                  png_compression=0, is_save=True,
                  jpeg_settings: dict | None = None,
                  cfg=None):
        # Save original image to ssd
        img1 = img  # .copy()
        image_filename = f"{ssd_path}/{image_name}-raw.png"
        t0 = time.monotonic()
        # The IMWRITE_PNG_COMPRESSION - 0: The compression level (0 = no compression, 9 = maximum compression), see config.ini
        # Only save files in flight mode
        if is_save:
            # Save RAW Image
            raw_image_resized_shape = self.image_resize(img1, raw_scale_factors, image_filename, None,
                                               resize_mode=raw_resize_mode, png_compression=png_compression,
                                               is_inspection=False, is_save_jp2=True)
            # img1 = convert_dummy_to_mono(img1) # if RGB it will convert, else nothing orig image is returned
            # cv2.imwrite(image_filename, img1, [int(cv2.IMWRITE_PNG_COMPRESSION), png_compression])
            self.log.debug(f'Saved original image to ssd path: {image_filename} in {get_dt(t0)}.')

        # Save downscaled image to sd card
        # downscale_factors = (8, 8)
        # NOTE: The SD version of image will be CREATED for exchange with GUI!!! AND  deleted in .write if preflight mode
        image_resized_filename = f"{sd_card_path}/{image_name}-raw-ds.png"
        jpeg_settings = jpeg_settings if jpeg_settings else {}

        overlay_sd_card = None # Was: "Raw Image"
        image_resized_shape = self.image_resize(img1, scale_factors, image_resized_filename,
                                           overlay=overlay_sd_card,
                                           resize_mode=resize_mode, png_compression=png_compression,
                                           is_inspection=True, jpeg_settings=jpeg_settings,
                                           is_save=is_save)
        self.log.debug(f'Saved downscaled image to sd path: {image_resized_filename} in {get_dt(t0)}.')
        # RETURN only plain picklable types (strings, tuples of ints/floats)
        return (
            image_filename,
            tuple(getattr(img, 'shape', tuple(img.shape))),  # ensure tuple
            image_resized_filename,
            tuple(getattr(image_resized_shape, 'shape', tuple(image_resized_shape)))
        )

    def get_files(self, path: str, extension='.png') -> list:
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

    @staticmethod
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


    def delete_files(self, *file_paths_input: Union[str, List[Optional[str]]]) -> int:
        """
        Processes a variable number of arguments (files) or a single list argument
        containing paths/None, and attempts to delete each one.

        Logs successful deletions using log.debug. Logs failures using log.error.

        Args:
            *file_paths_input: A variable number of arguments. Can be single strings
                               (paths) or a single list containing paths/None.

        Returns:
            int: The count of files that were successfully deleted.
        """
        successful_deletions = 0
        paths_to_process: List[Optional[str]] = []

        # 1. Normalize input: Consolidate all arguments into a single list of strings/None
        if len(file_paths_input) == 1 and isinstance(file_paths_input[0], list):
            # Case 1: Called with a single list argument (e.g., delete_files(file_list))
            paths_to_process = file_paths_input[0]
        else:
            # Case 2: Called with multiple positional arguments (e.g., delete_files(f1, f2, f3))
            # *file_paths_input is a tuple of the arguments; we convert it to a list.
            paths_to_process = list(file_paths_input)

        # 2. Iterate and delete
        for file_path in paths_to_process:
            # Cast item to str or handle None explicitly to ensure robustness
            if file_path is None:
                continue

            # Ensure the item is treated as a string path
            file_path_str = str(file_path)

            # Use pathlib for robust and platform-independent path handling
            file_to_delete = Path(file_path_str)

            if not file_to_delete.exists():
                continue

            try:
                if file_to_delete.is_file():
                    os.remove(file_to_delete)
                    self.log.debug(f"Successfully deleted file: '{file_path_str}'")
                    successful_deletions += 1
                else:
                    self.log.error(f"Cannot delete: '{file_path_str}' is a directory, not a file.")

            except PermissionError:
                self.log.error(f"Permission denied: Cannot delete file '{file_path_str}'.")
            except OSError as e:
                self.log.error(f"An OS error occurred while deleting '{file_path_str}': {e}")

        return successful_deletions

    def make_dirs_safe(self, full_path: str, mode: int = 0o755, exist_ok: bool = True,
                       check_space: bool = False) -> bool:
        """
        Safely create directories with error handling and logging.

        Args:
            full_path: Directory path to create
            mode: Permission mode (0o755 default)
            exist_ok: Don't raise error if directory exists (True)
            check_space: Check disk space before creating (False)

        Returns:
            bool: True if successful, False on failure
        """
        # Check disk space first if requested
        if check_space and not self._check_disk_space(full_path):
            self.log.critical(f"No space left for: {full_path}")
            return False

        try:
            os.makedirs(full_path, mode=mode, exist_ok=exist_ok)
            self.log.info(f"Created directory: {full_path}")
            return True

        except FileExistsError:
            if not exist_ok:
                self.log.warning(f"Directory already exists: {full_path}")
            return True

        except PermissionError:
            self.log.error(f"Permission denied: {full_path}")
            return False

        except FileNotFoundError:
            self.log.critical(f"Path not found: {full_path}")
            return False

        except OSError as e:
            if e.errno == errno.ENOSPC:
                self.log.critical(f"No space left on device: {full_path}")
                return False
            elif e.errno == errno.EROFS:
                self.log.error(f"Read-only filesystem: {full_path}")
                return False
            elif e.errno == errno.ENAMETOOLONG:
                self.log.error(f"Path too long: {full_path}")
                return False
            else:
                self.log.error(f"Failed to create {full_path}: {e}")
                return False

        except Exception as e:
            self.log.error(f"Unexpected error creating {full_path}: {e}")
            return False

    def _check_disk_space(self, full_path: str, min_gb: float = 0.1) -> bool:
        """Check if there's enough disk space."""
        try:
            # Get the mount point for the path
            check_path = os.path.dirname(full_path) or full_path
            total, used, free = shutil.disk_usage(check_path)
            free_gb = free / (1024 ** 3)

            if free_gb < min_gb:
                self.log.warning(f"Low disk space: {free_gb:.1f}GB free at {full_path}")
                return False
            return True
        except Exception:
            # If we can't check space, assume it's okay
            return True

    def test_overlay_roi(self):
        # Ensure output directory exists
        os.makedirs("output", exist_ok=True)

        # Load the test image
        input_path = "output/test_image.png"
        image = cv2.imread(input_path)

        # Apply ROI overlay
        self.overlay_rois(image)
        overlay_image = image

        # Save the result
        output_path = "output/test_image_overlay.png"
        success = cv2.imwrite(output_path, overlay_image)


if __name__ == '__main__':
    ts_orig = '250106_094041.794214'
    ts = Utils.reparse_timestamp(ts_orig)
    print(f'{ts_orig} --> {ts}')

    class TestConfig:
        """Test CFG"""
        # Centroiding ROI clamp (after hysteresis mask) settings:
        roi_keep_frac_x = 0.80
        roi_keep_frac_y = 0.90

        # Autogain/autoexposure ROI (center-rectangle version)
        roi_frac_x = 0.60
        roi_frac_y = 0.60

        # Autogain/autoexposure ROI (composite circle + top/bottom strip version)
        roi_circle_diam_frac_w = 0.85
        roi_strip_frac_y = 0.05

        # Enables adding rois overlay
        overlay_rois = True

    u = Utils(cfg=TestConfig())
    u.test_overlay_roi()


# Last line
