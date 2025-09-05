# Use install.md for installing the PUEO Server/GUI and requirements.txt file with venv and pip
# pip install opencv-python
# pip install scipy
# pip install astropy
# pip install imageio
# pip install matplotlib
# pip install zwoasi
# pip install pyserial
# pip install pillow
# pip install python3-tk
# pip install pyusb
# pip install tqdm
# pip install

# Standard Imports
import logging
import traceback
from contextlib import suppress
# import argparse
import sys
import math
import time
import datetime
import os
import io
import pstats
import shutil
# from shutil import disk_usage
# import usb.core
# from os import listdir
from types import SimpleNamespace
from datetime import timezone
import threading
import multiprocessing
from multiprocessing.pool import AsyncResult
from tqdm import tqdm
import cProfile

# External imports
import cv2
# import imageio.v3 as iio
# from astropy import modeling
from scipy.optimize import curve_fit
from scipy.spatial.transform import Rotation
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Custom imports
from lib.astrometry import Astrometry
from lib.common import get_file_size, DroppingQueue
from lib.compute_star_centroids import compute_centroids_from_still
from lib.common import load_config, logit, current_timestamp, cprint, get_dt, save_to_json, save_to_excel
from lib.common import archive_folder, delete_folder
from lib.config import Config
from lib.versa_logic_api import VersaAPI
from lib.camera import PueoStarCamera, DummyCamera
from lib.focuser import Focuser
from lib.star_comm_bridge import StarCommBridge
from lib.utils import read_image_grayscale, read_image_BGR, display_overlay_info, save_raws, image_resize, timed_function, create_symlink
from lib.source_finder import global_background_estimation, local_levels_background_estimation, \
    median_background_estimation, sextractor_background_estimation, find_sources, find_sources_photutils, \
    select_top_sources, select_top_sources_photutils
# from lib.astrometry import do_astrometry, optimize_calibration_parameters
from lib.astrometry import Astrometry
from lib.start_tracker_body_rates import StarTrackerBodyRates
from lib.telemetry import Telemetry
from lib.commands import Command

# CONFIG - [GLOBAL]
config_file = 'config.ini'  # Note this file can only be changed HERE!!! IT is still config.ini, but just for ref.
dynamic_file = 'dynamic.ini'

__version__ = '1.00b'
__created_modified__ = '2024-10-22'
__last_modified__ = '2025-06-18'
__release_date__ = '2025-01-06'
__program__ = 'Pueo Star Tracker Server'
__program_short__ = 'pueo_star_camera_operation_code.py'
__author__ = 'Windell Egami, Milan Stubljar of Stubljar d.o.o. <info@stubljar.com>'


class asi:
    """
    Used only in this main file for constants. For the values used, note the main asi file is imported in
    lib/camera.py
    """
    ASI_GAIN = 0
    ASI_FLIP = 9
    ASI_EXPOSURE = 1
    # ASI_IMGTYPE
    ASI_IMG_RAW8 = 0
    ASI_IMG_RGB24 = 1
    ASI_IMG_RAW16 = 2
    ASI_IMG_Y8 = 3
    ASI_IMG_END = -1


class PueoStarCameraOperation:
    """Implements Pueo StarCamera Operation Main Server"""
    _status = 'Initializing'

    def __init__(self, cfg):

        self.log = logging.getLogger('pueo')
        self.cfg = cfg
        self.cfg._log = self.log    # Add loger to the config object

        # Params
        self.start_t0 = time.monotonic()
        self.status = 'Initializing'
        self.operation_enabled = self.cfg.run_autonomous
        self.img_cnt = 0
        self.solver = self.cfg.solver
        self._flight_mode = self.cfg.flight_mode
        self._chamber_mode = self.cfg.run_chamber

        self._level_filter = self.cfg.level_filter

        self.time_interval: int = self.cfg.time_interval
        self.cadence: float = self.time_interval / 1e6

        self.serial_utc_update = None
        self.serial_time_datum = None
        self.omega = None
        self.omega_x = None
        self.omega_y = None
        self.omega_z = None

        self.curr_img = None
        self.prev_img = None
        self.prev_image_name = None

        self.distortion_calibration_params = None

        self.max_focus_position = None
        self.min_focus_position = None
        self.return_partial_images = None
        self.info_file = None
        self.focus_image_path = None

        self.online_auto_gain_enabled = None
        self.update_calibration = None

        self.pixel_saturated_value = self.cfg.pixel_saturated_value_raw16
        self.desired_max_pix_value = int(0.85 * self.pixel_saturated_value)
        self.pixel_count_tolerance = int(0.5 * (self.pixel_saturated_value - self.desired_max_pix_value))

        # Raw Image
        self.curr_image_name = None
        self.curr_image_info = None

        self.curr_scaled_name = None
        self.curr_scaled_info = None

        # Final Overlay Image
        self.foi_name = None
        self.foi_info = None
        self.foi_scaled_name = None
        self.foi_scaled_info = None

        self.prev_star_centroids = None
        self.curr_time = None

        self.first_time = None
        self.prev_time = 0
        self.min_size = None
        self.max_size = None
        self.use_photoutils = None
        self.subtract_global_bkg = None
        self.fast = None
        self.bkg_threshold = None
        self.number_sources = None
        self.save_raw = None
        self.is_trail = None
        self.is_array = None
        self.include_angular_velocity = self.cfg.include_angular_velocity

        self.best_exposure_time = self.cfg.lab_best_exposure
        self.best_gain_value = self.cfg.lab_best_gain

        self.distortion = None
        self.stdev = None
        self.best_focus = None

        self.timestamp_fmt = "%y%m%d_%H%M%S.%f"  # File name timestamp friendly format

        self.image_list = []
        self.image_filename_list = []

        self.astrometry = {}
        self.prev_astrometry = {}
        self.curr_star_centroids = None
        self.contours_img = None

        self.telemetry_queue = DroppingQueue(maxsize=self.cfg.fq_max_size)
        self.positions_queue = DroppingQueue(maxsize=self.cfg.fq_max_size)

        # Astrometry
        self.astro = Astrometry(self.cfg.ast_t3_database, self.cfg)

        # Board API (VersaLogic)
        self.versa_api = VersaAPI()

        # Set Camera and Focuser Board GPIO pins to known state: LOW
        self.logit('Setting GPIO Pins For Focuser to False (LOW):', color='cyan')
        with suppress(RuntimeError):
            self.versa_api.set_pin(self.cfg.sbc_dio_focuser_pin, False)

        time.sleep(5)

        self.logit('Setting GPIO Pins For Camera to False (LOW):', color='cyan')
        with suppress(RuntimeError):
            self.versa_api.set_pin(self.cfg.sbc_dio_camera_pin, False)


        # Wait 2 secs in case Camera/Focuser got started.
        time.sleep(2)

        # Create camera object
        self.camera = PueoStarCamera(self.cfg)
        self.camera_dummy = DummyCamera(self.cfg)

        # Create focuser object
        self.focuser = Focuser(self.cfg)

        # Initialize Telemetry
        self.log.info('Starting telemetry.')
        self.telemetry = Telemetry(self.cfg, self.telemetry_queue)

        if self.operation_enabled:
            self.focuser.open_aperture()

        self.log.info('Starting pueo server.')
        # Creating Multiprocessing Pool
        self.log.info(f'Starting Multiprocessing Pool with {self.cfg.max_processes} processes.')
        self.pool = multiprocessing.Pool(processes=self.cfg.max_processes)  # Create the pool once

        # Initialise SERVER NOW
        # Pueo Star Camera Operation Code SERVER
        config = {
            'pueo_server_ip': self.cfg.pueo_server_ip,
            'server_ip': self.cfg.server_ip,
            'port': self.cfg.port,
            'retry_delay': self.cfg.retry_delay,
            'max_retry': self.cfg.max_retry,
            'msg_max_size': self.cfg.msg_max_size
        }
        config = SimpleNamespace(**config)
        self.server = StarCommBridge(is_client=False, config=config)
        self.focuser.server = self.server

        self.is_running = False
        server_thread = threading.Thread(target=self.server.start_server, args=(self,))
        server_thread.start()
        self.log.info('Pueo Server Started.')

    def __del__(self):
        """
        Clean up the pool when the object is deleted.
        """
        print('__del__')
        with suppress(Exception):
            self.log.info(f'Closing Multiprocessing Pool.')
            self.pool.close()
            self.pool.join()
        print('__del__.completed.')

    def logit(self, msg, level='info', color=None):
        """
        Redirect messages to client and console
        """
        logit(msg, color=color)
        with suppress(Exception):
            self.server.write(msg, level)

    @property
    def status(self):
        """Getter for current status"""
        return self._status

    @status.setter
    def status(self, value):
        """Setter for status with validation"""
        valid_statuses = ['Initializing', 'Ready', 'Error', 'Stopped', 'Initializing (autofocus)']
        if value not in valid_statuses:
            raise ValueError(f"Invalid status. Must be one of: {valid_statuses}")
        self._status = value
        logit(f"PUEO Server Status changed to: {self._status} [{get_dt(self.start_t0)}]", color='green')  # Optional logging

    def is_ready(self):
        """Check if server is ready"""
        return self._status == 'Ready'

    @staticmethod
    def get_daily_path(path: str,
                       fmt: str = '%Y-%m-%d',
                       create_if_missing: bool = True) -> str:
        """
        Appends current date to path and optionally creates directory.

        Args:
            path: Base directory path
            fmt: Date format string (default: YYYY-MM-DD)
            create_if_missing: Create directory if it doesn't exist (default: True)

        Returns:
            str: Full path with date appended

        Raises:
            OSError: If directory creation fails
        """
        # Generate dated path
        date_str = datetime.datetime.now().strftime(fmt)
        full_path = os.path.join(path.rstrip(os.sep), date_str)

        # Create directory if requested and doesn't exist
        if create_if_missing and not os.path.exists(full_path):
            os.makedirs(full_path, exist_ok=True)

        return full_path

    def save_capture_properties(self, filename, timestamp_string):
        settings = self.camera.get_control_values()
        with open(filename, 'w') as f:
            for k in sorted(settings.keys()):
                f.write('%s: %s\n' % (k, str(settings[k])))
            f.write(f'Timestamp: {timestamp_string}')
        self.logit(f'Capture properties saved to {filename}')

    @staticmethod
    def gauss(x, a, x0, sigma, offset):
        return a * np.exp(-(x - x0) ** 2 / (2 * sigma ** 2)) + offset

    @staticmethod
    def abs_line(x, a, h, c):
        return a * abs(x - h) + c

    @staticmethod
    def process_image_covariance(img):
        img1 = cv2.medianBlur(img, 3)
        laplacian = cv2.Laplacian(img1, cv2.CV_64F)
        score = laplacian.var()
        return score

    def plt_savefig(self, plt, mage_file, is_preserve=True):
        """Saves an image and sends it to the messenger."""
        self.logit(f'Saving image: {mage_file}')
        plt.savefig(mage_file)
        self.server.write(mage_file, data_type='image_file', is_preserve=is_preserve)

    def fit_best_focus(self, focus_image_path, focus_positions, focus_scores, diameter_scores, max_focus_position,
                       focus_method='sequence_contrast'):
        # TODO: Implement sequence_twostep focus_method
        # zip the two lists and sort
        list1, list2 = zip(*sorted(zip(focus_positions, focus_scores)))
        focus_positions, focus_scores = (list(t) for t in zip(*sorted(zip(list1, list2))))

        focus_dict = {
            'sequence_contrast': {
                'best_focus_pos': self.cfg.trial_focus_pos,
                'status': False
            },
            'sequence_diameter': {
                'best_focus_pos': self.cfg.trial_focus_pos,
                'status': False
            },
            'sequence_twostep': {
                'best_focus_pos': self.cfg.trial_focus_pos,
                'status': False
            },
        }

        # Autofucus by list of covariances

        trial_best_focus_pos = None
        if focus_method == 'sequence_contrast':
            try:
                fit_points = self.cfg.fit_points_init  # 100
                # post process all images.

                # initial conditions for the fit.
                mean0 = sum(np.multiply(focus_positions, focus_scores)) / sum(focus_scores)
                # mean0 = focus_positions[focus_scores.index(max(focus_scores))]

                sigma0 = math.sqrt(sum(np.multiply(focus_scores, (focus_positions - mean0) ** 2)) / sum(focus_scores))
                base_height0 = min(focus_scores)
                a0 = max(focus_scores)

                popt, pcov = curve_fit(self.gauss, focus_positions, focus_scores, p0=[a0, mean0, sigma0, base_height0])
                fit_points = np.linspace(min(focus_positions), max(focus_positions), fit_points)
                _ = popt[0]
                mean = popt[1]
                sigma = abs(popt[2])
                height = popt[3]

                plt.figure()
                plt.plot(focus_positions, focus_scores, 'b', label='Focus Data')
                plt.plot(fit_points, self.gauss(fit_points, *popt), 'r--', label='Gaussian fit')
                plt.legend()
                plt.xlabel('Focus Position, counts')
                plt.ylabel('Sequence Contrast/Covariance')
                plt.title(f'mean: {round(mean, 2)}, stdev: {round(sigma, 2)}, height: {round(height, 2)}')
                # TODO: Remove old commented out code
                # plt.savefig(focus_image_path + 'focus_score.png')
                self.plt_savefig(plt, focus_image_path + 'focus_score.png')
                trial_best_focus_pos = int(popt[1])  # Rounding to integer

                self.logit('Fitted parameters:')
                self.logit(f'a = {popt[0]} +- {np.sqrt(pcov[0, 0])}')
                self.logit(f'X_mean = {popt[1]} +- {np.sqrt(pcov[1, 1])}')
                self.logit(f'sigma = {popt[2]} +- {np.sqrt(pcov[2, 2])}')
                self.logit(f'height = {popt[3]} +- {np.sqrt(pcov[3, 3])}')
                # In range?
                if trial_best_focus_pos < 0 or trial_best_focus_pos > max_focus_position:
                    trial_best_focus_pos = self.cfg.trial_focus_pos
                    self.logit(f'Focus set to out of limits. So defaulting to {self.cfg.trial_focus_pos}.')
                else:
                    focus_dict['sequence_contrast']['best_focus_pos'] = trial_best_focus_pos
                    focus_dict['sequence_contrast']['status'] = True
            except Exception as e:
                self.log.error(f'Fitting Error: {e}')
                self.logit(f"There was an error with fitting: {e}")
                sigma = self.cfg.sigma_error_value
                # This needs to be read from a config file
                focus_dict['sequence_contrast']['best_focus_pos'] = self.cfg.trial_focus_pos
                focus_dict['sequence_contrast']['status'] = False
        elif focus_method == 'sequence_diameter':
            # Autofocus score by mean diameters
            best_focus_pos = 0
            try:
                fit_points = self.cfg.fit_points_init
                # post process all images.

                # initial conditions for the fit.
                a = 1
                h = 0.5 * (min(focus_positions) + max(focus_positions))
                c = min(diameter_scores)

                popt, pcov = curve_fit(self.abs_line, focus_positions, diameter_scores, p0=[a, h, c])
                fit_points = np.linspace(min(focus_positions), max(focus_positions), fit_points)
                a = popt[0]
                h = trial_best_focus_pos = int(popt[1])  # best_focus_pos
                c = popt[2]

                self.logit('Fitted parameters:')
                self.logit(f'a = {popt[0]} +- {np.sqrt(pcov[0, 0])}')
                self.logit(f'h (x offset) = {popt[1]} +- {np.sqrt(pcov[1, 1])}')
                self.logit(f'c (y offset) = {popt[2]} +- {np.sqrt(pcov[2, 2])}')

                plt.figure()
                plt.plot(focus_positions, diameter_scores, 'b', label='Focus Data')
                plt.plot(fit_points, self.abs_line(fit_points, *popt), 'r--', label='Absolute Line Fit')
                plt.legend()
                plt.xlabel('Focus Position, counts')
                plt.ylabel('Diameter')
                plt.title(f'x offset: {round(h, 2)}, y offset: {round(c, 2)}, slope: {round(a, 2)}')
                # TODO: Remove old commented out code
                # plt.savefig(focus_image_path + 'diameters_score.png')
                self.plt_savefig(plt, focus_image_path + 'diameters_score.png')

                # is in range
                if trial_best_focus_pos < 0 or trial_best_focus_pos > max_focus_position:
                    trial_best_focus_pos = self.cfg.trial_focus_pos
                    self.logit(f'Focus set to out of limits. So defaulting to {self.cfg.trial_focus_pos}.')
                else:
                    focus_dict['sequence_diameter']['best_focus_pos'] = h
                    focus_dict['sequence_diameter']['status'] = True
            except Exception as e:
                self.log.error(f'Fitting Error: {e}')
                self.logit(f"There was an error with fitting: {e}")
                sigma = self.cfg.sigma_error_value
                focus_dict['sequence_diameter']['best_focus_pos'] = self.cfg.trial_focus_pos
                focus_dict['sequence_diameter']['status'] = False
        if focus_method == 'sequence_contrast':  # and focus_dict['covariance']['status']:
            best_focus_pos = focus_dict['sequence_contrast']['best_focus_pos']
        elif focus_method == 'sequence_diameter':
            best_focus_pos = focus_dict['sequence_diameter']['best_focus_pos']
        elif focus_method == 'sequence_twostep':
            best_focus_pos = focus_dict['sequence_twostep']['best_focus_pos']

        return best_focus_pos, sigma

    def read_filename_focus_positions(self, focus_image_path, focus_positions, focus_scores):
        # get the path/directory
        # focus_positions = []
        for focus_image_filename in os.listdir(focus_image_path):
            # check if the image ends with png
            if focus_image_filename.endswith(".png"):
                self.logit(f'  file: {focus_image_filename} focus_lookup', color='yellow')
                focus_position = int(focus_image_filename.split(".")[0].split('_')[2].split('f')[1])
                focus_positions.append(focus_position)
                local_filename = focus_image_path + focus_image_filename
                focus_score = self.process_image_covariance(local_filename)
                focus_scores.append(focus_score)

    def store_image(self, img, filename):
        mode = None
        image = Image.fromarray(img, mode=mode)
        image.save(filename)
        # Images saved using cv2 are larger and saved slower.
        # cv2.imwrite(filename, img, [int(cv2.IMWRITE_PNG_COMPRESSION), self.cfg.png_compression])
        self.logit(f'  saved image: {filename}', color='yellow')

    def capture_timestamp_save(self, path, inserted_string, img = None):
        """Take image if not provided."""
        timestamp_string = current_timestamp(self.timestamp_fmt)
        camera_settings = self.camera.get_control_values()
        self.logit(f"exposure time (us) : {camera_settings['Exposure']}")
        self.logit(f"gain (cB) : {camera_settings['Gain']}")
        log_msg = 'Capturing image image.' if img is None else 'Using provided/existing image.'
        self.log.debug(log_msg)
        img = self.camera.capture() if img is None else img
        self.logit(f"Max pixel: {np.max(img)}, Min pixel: {np.min(img)}.")
        self.image_list.append(img)
        basename = f"{path}{timestamp_string}_{inserted_string}"
        self.image_filename_list.append(f'{basename}.png')
        filename = f"{basename}.txt"
        self.save_capture_properties(filename, timestamp_string)
        return img, basename

    def save_image(self, img, path, timestamp_string, inserted_string):
        filename = f'{path}{timestamp_string}_{inserted_string}.png'
        self.log.debug(f'Image filename: {filename}')
        self.store_image(img, filename)
        filename = f'{path}{timestamp_string}_{inserted_string}.txt'
        self.log.debug(f'Image properties filename: {filename}')
        self.save_capture_properties(filename, timestamp_string)

    def check_gain_routine(self, curr_img, desired_max_pix_value, pixel_saturated_value,
                           pixel_count_tolerance, save_path=None) -> bool:
        bins = np.linspace(0, pixel_saturated_value, self.cfg.autogain_num_bins)
        arr = curr_img.flatten()
        counts, bins = np.histogram(arr, bins=bins)
        n = len(counts)
        high_pix_value = max(arr)
        min_count = self.cfg.min_count  # min_count
        # find the next largest pixel value
        second_largest_pix_value = int(bins[0])
        for i in range(n - 1, -1, -1):
            # if arr[i] < high_pix_value:
            # TODO: Fix this routine as it is failing. Note the len(bins) = len(counts) + 1
            with suppress(IndexError):
                if bins[i+1] < high_pix_value and counts[i] >= min_count:
                    second_largest_pix_value = int(bins[i+1])
                    break
                else:
                    second_largest_pix_value = high_pix_value

        # Note this has already been done! Disabling for now!
        if False:
            plt.figure()
            plt.hist(bins[:-1], bins, weights=counts)
            plt.xlabel('Brightness, counts')
            plt.ylabel('Frequency, pixels')
            plt.title(f'hpv: {high_pix_value}, slp: {second_largest_pix_value}, lpv: {min(arr)}')
            plt.grid()
            # plt.show() # Showing only in TESTING phase!
            filename = f'{save_path}check_gain'
            self.save_fig(filename, plt)

        if high_pix_value == pixel_saturated_value:
            if second_largest_pix_value >= (desired_max_pix_value + pixel_count_tolerance):
                self.logit("Image is saturated.")
            elif second_largest_pix_value < (desired_max_pix_value - pixel_count_tolerance):
                self.logit("Image has hot pixels. Image not saturated.")
            return True
        else:
            self.logit("Image has no hot pixels.")

    # def check_pixed_count_diff(self, desired_max_pix_value, pixel_count_tolerance, high_pix_value):
        difference = np.subtract(np.int64(desired_max_pix_value), np.int64(high_pix_value))
        if high_pix_value > desired_max_pix_value + pixel_count_tolerance:
            self.logit("Counts too high.")
            self.logit(f"Pixel count difference: {difference}")
            return True
        elif high_pix_value < desired_max_pix_value - pixel_count_tolerance:
            self.logit("Counts too low.")
            self.logit(f"Pixel count difference: {difference}")
            return True
        else:
            # highest value is high enough.
            self.logit("Pixels counts are in range. Ending iterations.")
            self.logit(f"Pixel count difference: {difference}")
            return False

    def check_gain_exposure_routine(self, curr_img, desired_max_pix_value, pixel_saturated_value,
                                    pixel_count_tolerance) -> bool:
        bins = np.linspace(0, pixel_saturated_value, self.cfg.autogain_num_bins)
        arr = curr_img.flatten()
        counts, bins = np.histogram(arr, bins=bins)
        n = len(counts)
        high_pix_value = max(arr)
        min_count = 10  # min_count
        # find the next largest pixel value
        second_largest_pix_value = int(bins[0])
        for i in range(n - 1, -1, -1):
            # TODO: Fix this routine as it is failing. Note the len(bins) = len(counts) + 1
            with suppress(IndexError):
                if bins[i + 1] < high_pix_value and counts[i] >= min_count:
                    second_largest_pix_value = int(bins[i + 1])
                    break
                else:
                    second_largest_pix_value = high_pix_value

        plt.figure()
        plt.hist(bins[:-1], bins, weights=counts)
        plt.xlabel('Brightness, counts')
        plt.ylabel('Frequency, pixels')
        plt.title(f'hpv: {high_pix_value}, slp: {second_largest_pix_value}, lpv: {min(arr)}')
        plt.grid()
        # TODO: Showing only for TESTING purposes!
        # plt.show() # Showing only in TESTING phase!

        if high_pix_value == pixel_saturated_value:
            if second_largest_pix_value >= (desired_max_pix_value + pixel_count_tolerance):
                self.logit("Image is saturated.")
            elif second_largest_pix_value < (desired_max_pix_value - pixel_count_tolerance):
                self.logit("Image has hot pixels. Image not saturated.")
            return True
        else:
            self.logit("Image has no hot pixels.")
        difference = np.subtract(np.int64(desired_max_pix_value), np.int64(high_pix_value))
        if high_pix_value > desired_max_pix_value + pixel_count_tolerance:
            self.logit("Counts too high.")
            self.logit(f"Pixel count difference: {difference}")
            return True
        elif high_pix_value < desired_max_pix_value - pixel_count_tolerance:
            self.logit("Counts too low.")
            self.logit(f"Pixel count difference: {difference}")
            return True
        else:
            # highest value is high enough.
            self.logit("Pixels counts are in range. Ending iterations.")
            self.logit(f"Pixel count difference: {difference}")
            return False

    def save_fig(self, basename, plt):
        """Saves a matplotlib figure as a compressed JPEG image with optimized settings.

        This function saves the given matplotlib figure as a JPEG image with settings
        optimized for small file size while maintaining reasonable visual quality.
        The image is saved with the provided basename appended with '_histogram.jpg'.

        Args:
            basename (str): Base filename to use for the output image (without extension).
            plt (matplotlib.pyplot): The pyplot figure object to be saved.

        Side Effects:
            - Writes an image file to disk
            - Logs the save operation using self.logit()

        Optimization Notes:
            - Uses JPEG format with default quality (adjust via rcParams)
            - Sets DPI=50 (lower than screen standard) for smaller dimensions
            - tight bounding box (bbox_inches='tight') removes excess whitespace
            - Minimal padding (pad_inches=0.05) reduces empty space around plot

        Example:
            >>> save_fig('temperature_data', plt)
            # Saves as 'temperature_data_histogram.jpg'
        """
        filename = f'{basename}_histogram.jpg'

        self.logit(f'Saving histogram image as: {filename}')

        plt.savefig(
            filename,
            dpi=75,
            facecolor='w',
            edgecolor='w',
            bbox_inches='tight',
            pad_inches=0.05,
            format='jpg'
        )
        # self.server.write(filename, data_type='image_file', is_preserve=True)  # Will delete the image so saving it again!

        tmp_filename = f'{basename}_histogram_tmp.png'
        self.plt_savefig(plt, tmp_filename, is_preserve=False)


    def do_auto_gain_routine(self, auto_gain_image_path, initial_gain_value,
                             desired_max_pix_value, pixel_saturated_value, pixel_count_tolerance,
                             max_iterations=None):

        """For adhoc single image gain check use autonomous last image rather than creating serioes of images."""
        max_autogain_iterations = self.cfg.max_autogain_iterations if max_iterations is None else max_iterations
        t0 = time.monotonic()
        is_done = False
        loop_counts = 1
        new_gain_value = initial_gain_value
        if max_autogain_iterations > 1:
            # take image using these settings.
            self.camera.set_control_value(asi.ASI_GAIN, new_gain_value)
            self.logit(f'New Gain Value: {new_gain_value}')
            if self.focuser.aperture_position == 'closed':
                self.logit('Opening Aperture.')
                self.focuser.open_aperture()

        self.logit("Starting autogain routine.")
        high_pix_value = 0
        while not is_done:
            if loop_counts > max_autogain_iterations:
                is_done = True
                self.logit("Maximum iterations reached. Can't find solution. Ending cycle.")
                break
            self.logit("#" * 104)
            self.logit(f"Autogain Iteration: {loop_counts} / {max_autogain_iterations}")

            self.logit('Capturing image.')
            inserted_string = 'e'  + 'g' + str(new_gain_value)
            # Take image or use existing self.curr_img for single gain update
            img = self.curr_img if max_iterations is not None and max_iterations == 1 else None
            img, basename = self.capture_timestamp_save(auto_gain_image_path, inserted_string, img)
            self.log.debug('Image captured.')

            bins = np.linspace(0, pixel_saturated_value, self.cfg.autogain_num_bins)
            # Creating a histogram requires flatten + histogram
            arr = img.flatten()
            counts, bins = np.histogram(arr, bins=bins)
            # There is one counts less than bins, therefore we iterate over counts, not bins

            n = len(counts)
            high_pix_value = max(arr)
            min_count = self.cfg.min_count # [IMAGE] min_count
            # find the next largest pixel value
            second_largest_pix_value = int(bins[0])
            self.log.debug(f'bin: {len(bins)} counts: {len(counts)}')
            try:
                for i in range(n - 1, -1, -1):
                    if bins[i + 1] < high_pix_value and counts[i] >= min_count:
                        # second_largest_pix_value = arr[i]
                        second_largest_pix_value = int(bins[i + 1])
                        break
                    else:
                        second_largest_pix_value = high_pix_value
            except IndexError as e:
                self.log.error(e)
                raise IndexError(e)

            # TODO: Decide if creating a plot is required for autogain
            if self.cfg.return_partial_images or True:
                t0 = time.monotonic()
                plt.figure()
                plt.hist(bins[:-1], bins, weights=counts)
                plt.xlabel('Gain Brightness, counts')
                plt.ylabel('Frequency, pixels')
                plt.title(f'hpv: {high_pix_value}, slp: {second_largest_pix_value}, lpv: {min(arr)}')
                plt.grid()
                # plt.show() # Showing only in TESTING phase!
                self.save_fig(basename, plt)
                self.log.debug(f'Image plot created in {get_dt(t0)}.')

            image_saturated = True
            if high_pix_value == pixel_saturated_value:
                if second_largest_pix_value >= (desired_max_pix_value + pixel_count_tolerance):
                    self.logit("Image is saturated.")
                    image_saturated = True
                elif second_largest_pix_value < (desired_max_pix_value - pixel_count_tolerance):
                    self.logit("Image has hot pixels. Image not saturated.")
                    high_pix_value = second_largest_pix_value  # overwrite high pix value with valid count.
                    image_saturated = False
            else:
                self.logit("Image has no hot pixels.")
                image_saturated = False

            difference = np.subtract(np.int64(desired_max_pix_value), np.int64(high_pix_value))
            self.logit(f'Counts: desired_max_pix_value:{desired_max_pix_value} high_pix_value: {high_pix_value}')
            if high_pix_value > desired_max_pix_value + pixel_count_tolerance:
                self.logit("Counts too high.")
                self.logit(f"Pixel count difference: {difference}")
            elif high_pix_value < desired_max_pix_value - pixel_count_tolerance:
                self.logit("Counts too low.")
                self.logit(f"Pixel count difference: {difference}")
            else:
                # highest value is high enough.
                self.logit("Pixels counts are in range. Ending iterations.")
                self.logit(f"Pixel count difference: {difference}")
                is_done = False
                break
            old_gain_value = new_gain_value

            if not image_saturated:
                self.logit(f"Image not saturated.")
                #calculate gain adjustment
                new_gain_value = self.calculate_gain_adjustment(old_gain_value, high_pix_value, desired_max_pix_value)

                self.logit(f'Old/New Gain Value: {old_gain_value}/{new_gain_value}')
                self.camera.set_control_value(asi.ASI_GAIN, new_gain_value)
            else:
                self.logit(f"Image is saturated. Setting gain to halfway.")
                new_gain_value = int(0.5*(self.cfg.max_gain_setting-self.cfg.min_gain_setting))
                if new_gain_value < self.cfg.min_gain_setting:
                    self.logit(f"New gain too low. Setting gain={self.cfg.min_gain_setting}. Recommend decreasing exposure time.")
                    new_gain_value = self.cfg.min_gain_setting
                elif new_gain_value >= self.cfg.max_gain_setting:
                    self.logit(
                        f"New gain too high. Setting gain={self.cfg.max_gain_setting}. Recommend increasing exposure time.")
                    new_gain_value = self.cfg.max_gain_setting

                self.logit(f'Old/New Gain Value: {old_gain_value}/{new_gain_value}')
                self.camera.set_control_value(asi.ASI_GAIN, new_gain_value)
            # Increment the loop_counts
            loop_counts += 1
        self.logit("##########################Auto Gain Routine Summary Results: ###############################", color='green')
        self.logit(f'desired_max_pix_value: {desired_max_pix_value} [counts]')
        self.logit(f'largest count in image is: {high_pix_value} [counts]')
        self.logit(f'optimal_gain_value: {new_gain_value} [cB]')
        self.logit("#"*107, color='green')
        self.logit(f'do_auto_gain_routine completed in {get_dt(t0)}.', color='cyan')
        return new_gain_value

    def calculate_gain_adjustment(self, old_gain_value, high_pix_value, desired_max_pix_value):
        # Constants from config
        max_gain = self.cfg.max_gain_setting  # e.g., 570
        min_gain = self.cfg.min_gain_setting

        try:
            # Step 1: Calculate old gain in electrons/ADU
            old_electrons_per_ADU_gain = 6.0037 * math.exp(-0.013 * old_gain_value)
            old_ADU_per_electrons_gain = 1 / old_electrons_per_ADU_gain
            self.logit(f"old_electrons_per_ADU_gain: {old_electrons_per_ADU_gain}")
            self.logit(f"old_ADU_per_electrons_gain: {old_ADU_per_electrons_gain}")

            # Step 2: Calculate gain multiplier
            gain_multiplier = (desired_max_pix_value - self.cfg.pixel_bias) / (high_pix_value - self.cfg.pixel_bias)
            self.logit(f"gain_multiplier: {gain_multiplier}")

            # Step 3: Compute new gain values
            new_ADU_per_electrons_gain = gain_multiplier * old_ADU_per_electrons_gain
            new_electrons_per_ADU_gain = 1 / new_ADU_per_electrons_gain
            self.logit(f"new_electrons_per_ADU_gain: {new_electrons_per_ADU_gain}")

            # Step 4: Invert gain equation to compute gain setting
            new_gain_value = -76.9230769 * math.log(0.16656395 * new_electrons_per_ADU_gain)
            new_gain_value = int(round(new_gain_value))

            # Step 5: Clamp gain and warn if limits are hit
            if new_gain_value < min_gain:
                self.logit(
                    f"WARNING: Calculated gain {new_gain_value} is below minimum {min_gain}. Clamping to minimum.")
                new_gain_value = min_gain
            elif new_gain_value > max_gain:
                self.logit(
                    f"WARNING: Calculated gain {new_gain_value} exceeds maximum {max_gain}. Clamping to maximum.")
                new_gain_value = max_gain
            else:
                self.logit(f"Clamped new_gain_value: {new_gain_value}")

        except (ZeroDivisionError, ValueError) as e:
            self.logit(f"ERROR: Failed to calculate new gain due to: {e}")
            new_gain_value = old_gain_value  # fallback

        return new_gain_value

    def do_auto_exposure_routine(self, auto_exposure_image_path, initial_exposure_value,
                                 desired_max_pix_value, pixel_saturated_value, pixel_count_tolerance):
        t0 = time.monotonic()
        is_done = False
        loop_counts = 1
        new_exposure_value = initial_exposure_value
        # take image using these settings.
        self.camera.set_control_value(asi.ASI_EXPOSURE, new_exposure_value)
        self.logit(f'New Exposure Value: {new_exposure_value}')
        if self.focuser.aperture_position == 'closed':
            self.logit('Opening Aperture.')
            self.focuser.open_aperture()
        self.logit("Starting autoexposure routine.")
        high_pix_value = 0
        while not is_done:
            self.logit("#" * 104)
            self.logit(f"Autoexposure Iteration: {loop_counts}")

            if loop_counts > self.cfg.max_autoexposure_iterations:
                is_done = True
                self.logit("Maximum iterations reached. Can't find solution. Ending cycle.")
                break

            self.logit('Capturing image.')
            inserted_string = 'e'  + 'g' + str(new_exposure_value)
            img, basename = self.capture_timestamp_save(auto_exposure_image_path, inserted_string)
            bins = np.linspace(0, pixel_saturated_value, self.cfg.autoexposure_num_bins)
            arr = img.flatten()
            counts, bins = np.histogram(arr, bins=bins)
            # There is one counts less than bins, therefore we iterate over counts, not bins
            n = len(counts)
            high_pix_value = max(arr)
            # min_count = 10 # [IMAGE] min_count
            # find the next largest pixel value
            second_largest_pix_value = int(bins[0])
            self.log.debug(f'bin: {len(bins)} counts: {len(counts)}')
            try:
                for i in range(n - 1, -1, -1):
                    if bins[i] < high_pix_value and counts[i] >= self.cfg.min_count:
                        # second_largest_pix_value = arr[i]
                        second_largest_pix_value = int(bins[i])
                        break
                    else:
                        second_largest_pix_value = high_pix_value
            except IndexError as e:
                self.log.error(e)
                raise IndexError(e)
            plt.figure()
            plt.hist(bins[:-1], bins, weights=counts)
            plt.xlabel('Brightness, counts')
            plt.ylabel('Frequency, pixels')
            plt.title(f'hpv: {high_pix_value}, slp: {second_largest_pix_value}, lpv: {min(arr)}')
            plt.grid()
            # plt.show() # Showing only in TESTING phase!
            self.save_fig(basename, plt)

            image_saturated = True
            if high_pix_value == pixel_saturated_value:
                if second_largest_pix_value >= (desired_max_pix_value + pixel_count_tolerance):
                    self.logit("Image is saturated.")
                    image_saturated = True
                elif second_largest_pix_value < (desired_max_pix_value - pixel_count_tolerance):
                    self.logit("Image has hot pixels. Image not saturated.")
                    high_pix_value = second_largest_pix_value  # overwrite high pix value with valid count.
                    image_saturated = False
            else:
                self.logit("Image has no hot pixels.")
                image_saturated = False

            difference = np.subtract(np.int64(desired_max_pix_value), np.int64(high_pix_value))
            if high_pix_value > desired_max_pix_value + pixel_count_tolerance:
                self.logit("Counts too high.")
                self.logit(f"Pixel count difference: {difference}")
            elif high_pix_value < desired_max_pix_value - pixel_count_tolerance:
                self.logit("Counts too low.")
                self.logit(f"Pixel count difference: {difference}")
            else:
                # highest value is high enough.
                self.logit("Pixels counts are in range. Ending iterations.")
                self.logit(f"Pixel count difference: {difference}")
                is_done = False
                break
            old_exposure_value = new_exposure_value

            if not image_saturated:
                # calculate & make gain adjustment
                self.logit(f'Image not saturated.')
                new_exposure_value = (old_exposure_value/high_pix_value)*desired_max_pix_value
                if new_exposure_value < self.cfg.min_exposure_setting:
                    self.logit(f"New exposure too low. Setting exposure={self.cfg.min_exposure_setting}. Recommend decreasing gain.")
                    new_exposure_value = self.cfg.min_exposure_setting

                elif new_exposure_value >= self.cfg.max_exposure_setting:
                    self.logit(
                        f"New gain too high. Setting gain={self.cfg.max_exposure_setting}. Recomend increasing gain.")
                    new_exposure_value = self.cfg.max_exposure_setting

                self.logit(f'Old Exposure Value: {old_exposure_value}')
                self.logit(f'New Exposure Value: {new_exposure_value}')
                self.camera.set_control_value(asi.ASI_GAIN, new_exposure_value)
            else:
                self.logit(f'Image Saturated.')
                new_exposure_value = 0.5 * old_exposure_value
                if new_exposure_value < self.cfg.min_exposure_setting:
                    self.logit(
                        f"New exposure too low. Setting exposure={self.cfg.min_exposure_setting}. Recommend decreasing gain.")
                    new_exposure_value = self.cfg.min_exposure_setting
                self.logit(f'Old Exposure Value: {old_exposure_value}')
                self.logit(f'New Exposure Value: {new_exposure_value}')
                self.camera.set_control_value(asi.ASI_EXPOSURE, new_exposure_value)
            loop_counts = loop_counts + 1
        self.logit("##########################Auto Exposure Routine Summary Results: ###############################", color='green')
        self.logit(f'desired_max_pix_value: {desired_max_pix_value} [counts]')
        self.logit(f'largest count in image is: {high_pix_value} [counts]')
        self.logit(f'optimal_exposure_value: {new_exposure_value} [us]')
        self.logit("#"*107, color='green')
        self.logit(f'do_auto_exposure_routine completed in {get_dt(t0)}.', color='cyan')
        return new_exposure_value

    def get_centroid_diameters(self, img, is_array=True, log_file_path="log/test_log.txt", return_partial_images=True):
        # read image in array format. From camera
        if is_array:
            # Scale the 16-bit values to 8-bit (0-255) range
            scale_factor = float(2 ** self.cfg.pixel_well_depth) - 1
            scaled_data = ((img / scale_factor) * 255).astype(np.uint8)
            # Create a BGR image
            img_bgr = cv2.merge([scaled_data, scaled_data, scaled_data])
        else:
            img_bgr = read_image_BGR(img)
            img = read_image_grayscale(img)

        # save image. For debugging
        if return_partial_images:
            # Create partial result folder. For debugging
            # partial_results_path = './ partial_results'
            if not os.path.exists(self.cfg.partial_results_path):  # partial_results_path
                os.makedirs(self.cfg.partial_results_path)
            # save input image
            partial_image_path_tmpl = r'{partial_results_path}/{partial_image_name}'
            partial_image_path = partial_image_path_tmpl.format(partial_results_path=self.cfg.partial_results_path,
                                                                partial_image_name=self.cfg.partial_image_name)
            # cv2.imwrite("./partial_results/0 - Input Image.png", img_bgr)
            cv2.imwrite(partial_image_path, img_bgr)

        # source finder
        self.logit("--------Substract background--------")
        # global background estimation
        global_cleaned_img = global_background_estimation(img, return_partial_images, self.cfg.sigma_clipped_sigma)
        img = global_cleaned_img
        # local background estimation
        cleaned_img, background_img = local_levels_background_estimation(img=img, log_file_path=log_file_path,
                                                                         return_partial_images=return_partial_images,
                                                                         leveling_filter_downscale_factor=self.cfg.leveling_filter_downscale_factor,
                                                                         level_filter=self.level_filter,
                                                                         level_filter_type=self.cfg.level_filter_type)
        self.logit("--------Find sources--------")
        bkg_threshold = self.cfg.img_bkg_threshold  # img_bkg_threshold = 3.1
        masked_image, estimated_noise = find_sources(img, background_img, bkg_threshold, return_partial_images,
                                    self.cfg.local_sigma_cell_size,
                                    self.cfg.src_kernal_size_x, self.cfg.src_kernal_size_y, self.cfg.src_sigma_x,
                                    self.cfg.src_dst)
        self.logit("--------Select top sources--------")
        self.number_sources = self.cfg.img_number_sources  # img_number_source = 40
        min_size = self.cfg.img_min_size  # img_min_size = 20
        max_size = self.cfg.img_max_size  # img_max_size = 600
        masked_image, sources_mask, sources_contours = select_top_sources(img, masked_image,
                                                                          estimated_noise=estimated_noise, fast=False, # TODO: HARD CODED: estimated_noise=None, fast=True
                                                                          number_sources=self.number_sources,
                                                                          min_size=min_size, max_size=max_size,
                                                                          return_partial_images=return_partial_images,
                                                                          dilate_mask_iterations=self.cfg.dilate_mask_iterations)

        # TODO: Review this logic
        # The select_top_sources exited due to too many contours
        if masked_image is None:
            return [-1, ]
        # Compute source centroids
        self.logit("--------Compute source centroids--------")
        precomputed_star_centroids, contours_img = compute_centroids_from_still(masked_image, sources_contours,
                                                                                img=img_bgr,
                                                                                min_potential_source_distance=self.cfg.min_potential_source_distance,
                                                                                log_file_path=log_file_path,
                                                                                return_partial_images=return_partial_images)
        if precomputed_star_centroids.size == 0:
            return [-1, ]
        diameters = precomputed_star_centroids[:, -1]
        self.logit("--------Diameter List--------")
        self.logit(str(diameters))
        return diameters

    def do_autofocus_routine(self, focus_image_path, focus_start_pos, focus_stop_pos, focus_step_count,
                             max_focus_position, focus_method='sequence_contrast'):
        """
        focus_method: ['sequence_contrast', 'sequence_diameter', 'sequence_twostep']
        """
        # TODO: Implement sequence_twostep focus_method
        t0 = time.monotonic()
        focus_scores = []
        diameter_scores = []
        measured_focus_positions = []
        count = 1
        focus_positions = np.linspace(focus_start_pos, focus_stop_pos, focus_step_count)

        if self.focuser.aperture_position == 'closed':
            self.logit('Opening Aperture.')
            self.focuser.open_aperture()
        # Getting the images at different focus positions
        for i in range(len(focus_positions)):
            self.logit("#########################################################################################")
            self.logit(f"Iteration: {count} of {len(focus_positions)}.")
            inserted_string = 'f' + str(int(round(focus_positions[i])))
            measured_focus_positions.append(self.focuser.move_focus_position(focus_positions[i]))
            self.logit('Taking image')
            log_file_path = "log/test_log.txt"
            img, basename = self.capture_timestamp_save(focus_image_path, inserted_string)
            # We intend to calculate both scores using sequence_diameter and sequence_contrast
            diameters = self.get_centroid_diameters(img=img, is_array=True, log_file_path=log_file_path)
            # TODO: diameters will not be none but [-1], decide with next if...
            if diameters is None:
                self.logit(f"best focus fit failed. Setting focus position to {self.cfg.lab_best_focus}.")
                # When the routine fails, set the focus to lab_best_focus:
                self.focuser.move_focus_position(self.cfg.lab_best_focus)
                return self.cfg.lab_best_focus, self.cfg.stdev_error_value

            diameter_score = np.mean(diameters)
            self.logit(f'Diameter Focus score: {diameter_score}')
            self.logit(f'Length of diameters: {len(diameters)}')
            focus_scores.append(diameter_score)
            score = self.process_image_covariance(img)
            self.logit(f'Covariance Focus score: {score}')
            focus_scores.append(score)

            count = count + 1

        self.logit('Processing focus images.')
        # self.read_filename_focus_positions(focus_image_path, focus_positions, focus_scores)
        # self.logit(measured_focus_positions)
        # self.logit(focus_scores)
        try:
            trial_best_focus_pos, stdev = self.fit_best_focus(focus_image_path, focus_positions, focus_scores,
                                                              diameter_scores, max_focus_position, focus_method)
            # Calculate focus deviation range
            focus_min_pos = self.cfg.lab_best_focus * (1 - self.cfg.autofocus_max_deviation)
            focus_max_pos = self.cfg.lab_best_focus * (1 + self.cfg.autofocus_max_deviation)

            # Check if the calculated result of autofocus is within allowed autofocus_max_deviation
            if trial_best_focus_pos < focus_min_pos or trial_best_focus_pos > focus_max_pos:
                self.logit(
                    f'AutoFocus position out of allowed range of autofocus_max_deviation: {trial_best_focus_pos} allowed range: {focus_min_pos}..{focus_max_pos}')
                trial_best_focus_pos = self.cfg.lab_best_focus

            if trial_best_focus_pos < focus_start_pos:
                trial_best_focus_pos = focus_start_pos
                self.logit('Focus position too great. Setting to upper bound of focus range.')
            elif trial_best_focus_pos > focus_stop_pos:
                trial_best_focus_pos = focus_stop_pos
                self.logit('Focus position too small. Setting to lower bound of focus range.')
            if stdev > (focus_stop_pos + focus_start_pos) / 2:
                trial_best_focus_pos = self.cfg.trial_focus_pos
        except Exception as e:
            # TODO: one of the above values is none and we get an exception
            self.logit(f"best focus fit failed. Setting focus position to {self.cfg.trial_focus_pos}.")
            stdev = self.cfg.stdev_error_value
            trial_best_focus_pos = self.cfg.trial_focus_pos

            cprint(f'Error: {e}', 'red')
            logit(traceback.print_exception(e))
            self.log.error(traceback.format_exception(e))
            logit(e, color='red')

        # move the focus to the best found position.
        self.logit(f"Moving Focus to best focus position: {int(round(trial_best_focus_pos))}")
        self.focuser.move_focus_position(trial_best_focus_pos)
        # take final image at estimated best focus
        inserted_string = 'f' + str(int(round(trial_best_focus_pos)))
        self.capture_timestamp_save(focus_image_path, inserted_string)
        # No need to close anything
        # self.logit('Closing Aperture.')
        # self.focuser.close_aperture()
        self.logit(f'do_autofocus_routine completed in {get_dt(t0)}.', color='cyan')
        return trial_best_focus_pos, stdev

    def save_images(self, image_filename_list):
        ## Write images to disk##
        if self.is_flight:
            self.logit('Writing Images to disk.')
            for idx, filename in enumerate(image_filename_list):
                self.store_image(self.image_list[idx], image_filename_list[idx])
            self.logit('Finished writing images to disk.')
        else:
            self.logit('Skipping write Images to disk (preflight mode).', color='yellow')

    def camera_autofocus(self, cmd=None):
        t0 = time.monotonic()

        focus_method = cmd.focus_method if cmd else 'sequence_contrast'
        enable_autofocus = True
        # Autofocus.AutoGain
        enable_autogain = bool(cmd.enable_autogain) if cmd else self.cfg.enable_autogain_with_autofocus

        self.logit(f'Refocusing (take gain/focus sequences). Focus Method: {focus_method} Autogain Enabled: {enable_autogain}', color='cyan')
        # GUI: AutoGain True, AutoExposure: True ==> enable_autogain = True
        #   else: enable_autogain = False

        self.first_time = True
        # Run set focus/gain/exposure as per GUI
        if not enable_autofocus:
            self.logit(f'Autofocus routine disabled...')
            self.camera.set_control_value(asi.ASI_EXPOSURE,self.cfg.asi_exposure)  # asi_exposure = 9000 - GUI: Exposure Time [ms] * 1000
            self.camera.set_control_value(asi.ASI_GAIN,self.cfg.asi_gain)  # asi_gain = 100 - GUI: Gain [Cb] Centibels like decibels
            self.camera.set_control_value(asi.ASI_FLIP, self.cfg.asi_flip)  #
            self.camera.set_image_type(asi.ASI_IMG_RAW16)
            self.camera.set_roi(bins=self.cfg.roi_bins)

            self.min_focus_position, self.max_focus_position = self.focuser.home_lens_focus()
            curr_focus_position = self.focuser.get_focus_position()
            self.logit(f'Current focus position: {curr_focus_position}')
            curr_focus_position = self.focuser.move_focus_position(self.cfg.lab_best_focus)
            self.logit(f'Moved focus to: {curr_focus_position}')

        # Run autofocus routine
        else:  # == True:
            # autofocus and autogain camera maintanence
            self.image_list = []
            self.image_filename_list = []

            # set to default image dynamic range and resolution.
            self.camera.set_image_type(asi.ASI_IMG_RAW16)
            self.camera.set_roi(bins=self.cfg.roi_bins)

            #set definition of pixel saturated value bases on camera settings
            self.pixel_saturated_value = self.cfg.pixel_saturated_value_raw16

            #define the target max pixel value and the toleranace of this target
            self.desired_max_pix_value = int(0.90 * self.pixel_saturated_value)
            self.pixel_count_tolerance = int(2 * (self.pixel_saturated_value - self.desired_max_pix_value))

            # Open Aperture before taking images
            if self.focuser.aperture_position == 'closed':
                self.logit('Opening Aperture.', color='magenta')
                self.focuser.open_aperture()

            # create autgain sequence image path
            timestamp_string = current_timestamp(self.timestamp_fmt)
            auto_gain_image_path = self.cfg.auto_gain_image_path_tmpl.format(timestamp_string=timestamp_string)
            os.mkdir(auto_gain_image_path)

            #setting to minimum gain since starting from scratch and gain should increase to meet target.
            gain_value = self.cfg.min_gain_setting

            # first move to the lab best focus position.
            self.focuser.move_focus_position(self.cfg.lab_best_focus)
            self.logit('Moved to lab best focus position before running autogain.', color='yellow')

            # Find best gain value
            if enable_autogain:
                self.logit("Running auto gain before autofocus routine.", color='cyan')
                self.best_gain_value = self.do_auto_gain_routine(
                    auto_gain_image_path,
                    gain_value,
                    self.desired_max_pix_value,
                    self.pixel_saturated_value,
                    self.pixel_count_tolerance)
                self.logit(f"Auto gain done. Best gain: {self.best_gain_value}")
            else:
                self.logit("SKipping auto gain before autofocus routine.", color='cyan')
            ###############################
        # change to low resolution, low dynamic range##
        # take an image at full resolution and binning ==2
        self.camera.set_image_type(asi.ASI_IMG_RAW8)
        self.camera.set_roi(bins=self.cfg.roi_bins)

        # if self.arduino_serial and self.arduino_serial.isOpen():
        #get the current focuser position.
        if self.focuser.is_open():
            # Home the lens.
            self.min_focus_position, self.max_focus_position = self.focuser.home_lens_focus()
            curr_focus_position = self.focuser.get_focus_position()
            self.logit(f'Current focus position: {curr_focus_position}')
        else:
            # TODO. need to restart the USB port if cannot see the serial port.
            self.logit('Serial port not open.')

        # TODO: No autofocuser results in the None error in the line below.
        ## Doing initial autofocus run.##

        # Define the start and stop, count focus sequence
        # Command params:
        #             'start_position': 300,
        #             'stop_position': 400,
        #             'step_count': 5

        # Default values from config
        coarse_start_focus_pos = self.cfg.lab_best_focus - int(0.5*self.cfg.focus_tolerance_percentage/100 * (self.max_focus_position - self.min_focus_position))  # units of counts
        coarse_stop_focus_pos = self.cfg.lab_best_focus + int(0.5*self.cfg.focus_tolerance_percentage/100 * (self.max_focus_position - self.min_focus_position))  # units of counts
        coarse_focus_step_count = self.cfg.coarse_focus_step_count
        focus_params_src = 'config.ini'

        # Override with cmd if provided (API CLI/GUI call)
        if cmd:
            coarse_start_focus_pos = int(cmd.start_position)
            coarse_stop_focus_pos = int(cmd.stop_position)
            coarse_focus_step_count = int( cmd.step_count)
            focus_params_src = 'From API'

        logit(
            f'Autofocus params: src: {focus_params_src} '
            f'range: {coarse_start_focus_pos} -> {coarse_stop_focus_pos} '
            f'steps: {coarse_focus_step_count} ' 
            f'enable_autogain: {enable_autogain}',
            color='yellow'
        )

        ## Doing coarse focus adjustments##
        # Note there must be no : in the name for sake of using this as file/path name on linux/windows.
        # focus_image_path = '/home/windell/PycharmProjects/pueo_star_tracker/' + timestamp_string + '_coarse_focus_images/'
        # focus_image_path = '/home/windell/PycharmProjects/version_5/' + timestamp_string + '_coarse_focus_images/'
        # focus_image_path_tmpl = r'/home/windell/PycharmProjects/version_5/{timestamp_string}_coarse_focus_images/'

        #create the autofocus routine directory
        timestamp_string = current_timestamp("%y%m%d_%H%M%S.%f")
        self.focus_image_path = self.cfg.focus_image_path_tmpl.format(timestamp_string=timestamp_string)
        os.mkdir(self.focus_image_path)

        # Open Aperture before taking images
        if self.focuser.aperture_position == 'closed':
            self.logit('Opening Aperture.', color='magenta')
            self.focuser.open_aperture()

        self.logit("First pass autofocus routine...", color='cyan')

        # TODO: Recheck if autofocus should be done when running a program on its own...
        # best_focus = self.cfg.lab_best_focus
        self.best_focus, self.stdev = self.do_autofocus_routine(self.focus_image_path, coarse_start_focus_pos,
                                                                coarse_stop_focus_pos, coarse_focus_step_count,
                                                                self.max_focus_position,
                                                                focus_method)

        # change back to default image dynamic range and resolution.
        self.camera.set_image_type(asi.ASI_IMG_RAW16)
        self.camera.set_roi(bins=self.cfg.roi_bins)

        # Saving Autofocus images (in flight mode only)
        self.save_images(self.image_filename_list)

        #reinitialize the image_list and filename_lists
        self.image_list = []
        self.image_filename_list = []

        # change to high resolution and full dynamic rnage.
        # take an image at full resolution and binning ==2
        self.camera.set_image_type(asi.ASI_IMG_RAW16)
        self.camera.set_roi(bins=self.cfg.roi_bins)

        #define the 16 bit depth
        self.pixel_saturated_value = self.cfg.pixel_saturated_value_raw16

        #make autogain path
        timestamp_string = current_timestamp(self.timestamp_fmt)
        auto_gain_image_path = self.cfg.auto_gain_image_path_tmpl.format(timestamp_string=timestamp_string)
        os.mkdir(auto_gain_image_path)

        #set gain value to what is the best known currently
        gain_value = self.best_gain_value   #this is the best gain from the iterations

        # Find best gain value
        if enable_autogain:
            self.logit("Running auto gain routine after autofocus.", color='cyan')
            self.best_gain_value = self.do_auto_gain_routine(
                auto_gain_image_path,
                gain_value,
                self.desired_max_pix_value,
                self.pixel_saturated_value,
                self.pixel_count_tolerance)
            self.logit(f"Auto gain done. Best gain: {self.best_gain_value}")
            ##############################
        else:
            self.logit("SKipping auto gain after autofocus routine.", color='cyan')

        self.logit(f'camera autofocus completed in {get_dt(t0)}.', color='cyan')

    def camera_init(self):
        """
        Init camera as part of server startup:
            - run focushome
            - Set lab fucus
            - set lab gain - 120
            - set lab exposure - 100 ms
            - set close aperture
        """
        # Recalibrate focuser position
        self.min_focus_position, self.max_focus_position = self.focuser.home_lens_focus()

        # Set focus to lab_focus position
        self.focuser.move_focus_position(self.cfg.lab_best_focus)
        self.logit(f'Set camera focus to: {self.cfg.lab_best_focus} [counts]', color='blue')

        # Set camera gain to lab_best_gain
        self.camera.set_control_value(asi.ASI_GAIN, self.cfg.lab_best_gain)
        self.logit(f'Set camera gain to: {self.cfg.lab_best_gain} [cB]', color='blue')

        # Set camera exposure to lab_best_exposure
        self.camera.set_control_value(asi.ASI_EXPOSURE, self.cfg.lab_best_exposure)
        self.logit(f'Set camera exposure to: {self.cfg.lab_best_exposure} [us]', color='blue')

        # Close aperture
        self.focuser.close_aperture()

    def run(self):
        self.is_running = True

        # Initialise camera settings
        self.camera_init()

        if self.cfg.run_autofocus:
            self.log.info('Running startup autofocus.')
            self.status = 'Initializing (autofocus)'
            self.camera_autofocus()
        else:
            self.log.warning('Skipping startup autofocus. Set config::GENERAL::run_autofocus to True to enable.')

        # take an image with the ground best focus position and the new best focus position and choose the better of the two.
        # take images for image distortion correction.
        # calculate distortion
        # take an image with the ground best distortion parameters and the new parameters and choose the better of the two.
        # at this point, the camera should be ready for normal operation.

        # time_interval = 1000000 # Global
        prev_time = time.monotonic()
        command = ''
        # TODO: By default server should run automatically and operation_enabled should be True
        online_auto_gain_enabled = False
        self.prev_img = []
        self.curr_img = []
        self.omega_x = float('inf')
        self.omega_y = float('inf')
        self.omega_z = float('inf')
        self.omega = [self.omega_x, self.omega_y, self.omega_z]

        dt = datetime.datetime.now(timezone.utc)
        utc_time = dt.replace(tzinfo=timezone.utc)
        self.serial_utc_update = utc_time.timestamp()
        self.serial_time_datum = time.monotonic()

        # astrometry vars
        self.is_array = True
        self.is_trail = self.cfg.ast_is_trail
        self.save_raw = self.cfg.save_raw
        self.number_sources = self.cfg.ast_number_sources
        self.min_size = self.cfg.ast_min_size
        self.max_size = self.cfg.ast_max_size
        self.use_photoutils = self.cfg.ast_use_photoutils
        self.subtract_global_bkg = self.cfg.ast_subtract_global_bkg
        self.fast = self.cfg.ast_fast
        self.bkg_threshold = self.cfg.ast_bkg_threshold
        self.distortion_calibration_params = {"FOV": None, "distortion": None}

        # self.cfg.lab_fov = 10.79490900481796
        fov = self.cfg.lab_fov
        # self.cfg.lab_distortion_coefficient_1
        distortion_coefficient_1 = self.cfg.lab_distortion_coefficient_1  # -0.1
        distortion_coefficient_2 = self.cfg.lab_distortion_coefficient_2  # 0.1
        # default_distortion_calibration_params = {"FOV": 10.79490900481796, "distortion":[-.1, .1]}
        default_distortion_calibration_params = {"FOV": fov,
                                                 "distortion": [distortion_coefficient_1, distortion_coefficient_2]}

        # flag to update previously calculated distortion calibration parameters
        self.update_calibration = False

        self.return_partial_images = False
        # ssd_path = "ssd_path/"  # Note: is now global
        # sd_card_path = "sd_card_path/" # Note: is now global

        # read distortions calibration parameters
        # calibration_params_file = 'calibration_params_best.txt'
        try:
            with open(self.cfg.calibration_params_file, "r") as file:
                lines = file.readlines()
                for line in lines:
                    line = line.strip()
                    if line:
                        key, value = line.split(":")
                        self.distortion_calibration_params[key.strip()] = float(value.strip())
        except FileNotFoundError:
            self.logit("Error: Distortion calibration file not found.")
            self.distortion_calibration_params = default_distortion_calibration_params

        logit('Entering main operational loop:', color='green')
        self.status = 'Ready'
        while True:
            try:
                self.main_loop()
            except KeyboardInterrupt:
                self.logit('Server: Exiting!', level='error', color='red')
                self.telemetry.close()
                self.server.close()
                return

    def get_cpu_temp(self):
        try:
            with open("/sys/class/thermal/thermal_zone0/temp", "r") as cpu_file:
                temp_str = cpu_file.read()
                cpu_temp = int(temp_str) / 1000.0
                self.log.info(f'CPU Temp: {cpu_temp}')
                return cpu_temp
        except Exception as e:
            self.log.error(f"An error occurred reading CPU temperature: {e}")
        return None

    def autogain_maintanence(self, camera_settings):
        """
        Comment by Windell: dont only want to do this when in autonomous mode. It should be done whenever it is asked to.
        Checking for autonomous mode should happen earlier.
        """
        #make sure camera is set to correct settings for resolution and dynamic range.
        self.camera.set_image_type(asi.ASI_IMG_RAW16)
        self.camera.set_roi(bins=self.cfg.roi_bins)

        gain_value = camera_settings['Gain']
        timestamp_string = current_timestamp(self.timestamp_fmt)
        auto_gain_image_path = self.cfg.auto_gain_image_path_tmpl.format(timestamp_string=timestamp_string)
        os.mkdir(auto_gain_image_path)
        is_check_gain, et1 = timed_function(self.check_gain_routine,
                                            self.curr_img, self.desired_max_pix_value, self.pixel_saturated_value,
                                            self.pixel_count_tolerance, auto_gain_image_path)
        if is_check_gain:
            self.server.write("Performing autogain maintenance.")
            curr_gain_value = camera_settings['Gain']
            self.best_gain_valuem, et2 = timed_function(self.do_auto_gain_routine,
                                                        auto_gain_image_path, curr_gain_value,
                                                        self.desired_max_pix_value, self.pixel_saturated_value,
                                                        self.pixel_count_tolerance,
                                                        max_iterations=1)
            self.logit(f"Auto gain done. Old gain: {curr_gain_value} Best gain: {self.best_gain_value} Exec: {et1}, {et2} sec.")
            gain_value = self.best_gain_value
        else:
            self.logit(f"Auto gain done. Gain value is {gain_value}. No need to adjust. Exce time: {et1} sec.", color='green')

        return gain_value

    def info_add_image_info(self, camera_settings, curr_utc_timestamp):
        unique_pixel_values = np.unique(self.curr_img)  # Get distinct pixel values
        median_pixel_value = np.median(unique_pixel_values)  # Median of unique counts

        with open(self.info_file, "a", encoding='utf-8') as file:
            file.write("Image type : Single standard operating image\n")
            file.write(f"system timestamp : {curr_utc_timestamp}\n")
            # Removing the serial timestamp  -  not required.
            # file.write(f"serial timestamp : {curr_serial_utc_timestamp}\n")
            file.write(f"exposure time (us) : {camera_settings['Exposure']}\n")
            file.write(f"gain (cB) : {camera_settings['Gain']}\n")
            file.write(f"focus position : {self.focuser.focus_position}\n")
            file.write(f"aperture position : {self.focuser.aperture_pos}, {self.focuser.aperture_f_val}\n")
            file.write(f"min/max pixel value (counts) : {np.min(self.curr_img)} / {np.max(self.curr_img)}\n")
            file.write(f"mean/median pixel value (counts) : {np.mean(self.curr_img):.1f} / {median_pixel_value:.1f}\n")
            # file.write(f"median pixel value (counts) : {median_pixel_value:.1f}\n")
            # file.write(f"min pixel value (counts) : {np.min(self.curr_img)}\n")
            file.write(f"detector temperature : {camera_settings['Temperature']/10} °C\n")

            # write cpu temp
            cpu_temp = self.get_cpu_temp()
            file.write(f"CPU Temperature: {cpu_temp} °C\n")

            # write estimated calibration parameters to log
            file.write("estimated distortion parameters:\n")
            for key, value in self.distortion_calibration_params.items():
                # Write the key-value pair to the file
                file.write(f"{key}: {value}\n")
            else:
                file.write("Distortion calibration file not found.\n")

    def get_disk_usage(self, path: str = "/"):
        """
        Retrieves disk usage statistics for a given path on Linux and Windows.

        Args:
            path (str): The path or drive to check. Default is '/' for Linux. Use 'C:' or similar for Windows.

        Returns:
            dict: A dictionary containing total, used, and free space in bytes.
        """
        if os.name == 'posix':  # Linux/Unix/MacOS
            try:
                statvfs = os.statvfs(path)
                total_space = statvfs.f_frsize * statvfs.f_blocks
                free_space = statvfs.f_frsize * statvfs.f_bfree
                used_space = total_space - free_space
                return {
                    "total_space": total_space,
                    "used_space": used_space,
                    "free_space": free_space,
                }
            except Exception as e:
                raise OSError(f"Error retrieving disk usage for {path} on Linux/Unix: {e}")

        elif os.name == 'nt':  # Windows
            try:
                import ctypes
                path = 'C:' if path == '/' else path
                free_bytes_available = ctypes.c_ulonglong(0)
                total_number_of_bytes = ctypes.c_ulonglong(0)
                total_number_of_free_bytes = ctypes.c_ulonglong(0)

                ctypes.windll.kernel32.GetDiskFreeSpaceExW(
                    ctypes.c_wchar_p(path),
                    ctypes.pointer(free_bytes_available),
                    ctypes.pointer(total_number_of_bytes),
                    ctypes.pointer(total_number_of_free_bytes)
                )

                total_space = total_number_of_bytes.value
                free_space = total_number_of_free_bytes.value
                used_space = total_space - free_space

                return {
                    "total_space": total_space,
                    "used_space": used_space,
                    "free_space": free_space,
                }
            except Exception as e:
                raise OSError(f"Error retrieving disk usage for {path} on Windows: {e}")

        else:
            raise NotImplementedError("This function supports only Linux/Unix and Windows.")

    def info_add_astro_info(self):
        with open(self.info_file, "a", encoding='utf-8') as file:
            # image size
            file.write(f"raw file size: {sys.getsizeof(self.curr_img)} bytes\n")
            file.write(f"compressed file size: {os.path.getsize(self.cfg.ssd_path)} bytes\n")
            # available disk space - THIS IS UNIX!!! only
            # statvfs = os.statvfs('/')
            # available_space = statvfs.f_frsize * statvfs.f_bavail
            available_space = self.get_disk_usage()['free_space']

            available_space_gb = available_space / (1024 ** 3)
            file.write(f"Available disk space: {available_space_gb:.2f} GB\n")
            # astrometry result
            file.write("y coordinates, x coordinates, flux, std, fwhm:\n")
            # fwhm ~ Full Width Half Max
            with suppress(IndexError, TypeError):
                for i in range(len(self.curr_star_centroids)):
                    std = self.curr_star_centroids[i][3]
                    fwhm = self.curr_star_centroids[i][4]
                    file.write(f"{self.curr_star_centroids[i][0]},{self.curr_star_centroids[i][1]}, {self.curr_star_centroids[i][2]}, {std}, {fwhm}\n")
            file.write("\nAstrometry:\n")
            for key, value in self.astrometry.items():
                file.write(f"{key}: {value}\n")

            plate_scale = 0.0
            try:
                fov_deg = float(self.astrometry.get('FOV', 0.0))
                if fov_deg > 0 and hasattr(self, 'curr_img') and self.curr_img is not None:
                    plate_scale = (fov_deg / self.curr_img.shape[1]) * 3600  # arcsec/pixel
            except (TypeError, AttributeError, ValueError) as e:
                # Fallback to nominal FOV if astrometry fails
                nominal_fov = self.cfg.lab_fov  # degrees
                if hasattr(self, 'curr_img') and self.curr_img is not None:
                    plate_scale = (nominal_fov / self.curr_img.shape[1]) * 3600

            plate_scale_value = f'{plate_scale:.6f} arcsec/px'
            file.write(f"\nplate scale : {plate_scale_value}\n")
            if self.astrometry:
                # Added for overlay
                # TODO: HAve these captured differently!!!
                self.astrometry['PlateScale'] = plate_scale_value
                exposure = self.camera.get_control_values().get('Exposure', 'unknown')
                self.astrometry['ExposureTime'] = f'{exposure} us' # microsec

    def info_add_misc_info(self, omega_x, omega_y, omega_z, pk):
        with open(self.info_file, "a", encoding='utf-8') as file:
            file.write(f"Body rotation rate:\n")
            file.write(f"omegax: {omega_x}\n")
            file.write(f"omegay: {omega_y}\n")
            file.write(f"omegaz: {omega_z}\n")
            file.write("\n")
            # Do not print Covariance Matrix
            if False:
                file.write(f"Covariance Matrix:\n{pk} (deg/sec)^2")
                file.write("\n")
            if len(self.prev_img) != 0:
                file.write(f"name of previous image: {self.prev_image_name}\n")
                file.write(f"time difference between images: {self.curr_time - self.prev_time}")

    def info_add_angular_velocity_info(self, angular_velocity, delta_t_sec):
        """
        Add angular velocity information to the info file.

        Args:
            angular_velocity (dict): Dictionary containing 'roll_rate', 'az_rate', 'el_rate' in deg/s
            delta_t_sec (float): Time difference between solutions in seconds
        """
        with open(self.info_file, "a", encoding='utf-8') as file:
            file.write(f"Body rotation rate:\n")
            file.write(f"omega_roll: {angular_velocity.get('roll_rate', 'N/A')} deg/s\n")
            file.write(f"omega_az: {angular_velocity.get('az_rate', 'N/A')} deg/s\n")
            file.write(f"omega_el: {angular_velocity.get('el_rate', 'N/A')} deg/s\n")
            file.write(f"time difference between images\\solutions: {delta_t_sec} seconds\n")
            if hasattr(self, 'prev_image_name') and self.prev_image_name:
                file.write(f"name of previous image: {self.prev_image_name}\n")

    def do_astrometry(self, is_multiprocessing=True):
        """astro.do_astrometry wrapper function."""
        args = (
            self.curr_img,                              # img,
            self.is_array,                              # is_array=True,
            self.is_trail,                              # is_trail=True,
            self.use_photoutils,                        # use_photutils=False,
            self.subtract_global_bkg,                   # subtract_global_bkg=False,
            self.fast,                                  # fast=False,
            self.number_sources,                        # number_sources=20,
            self.bkg_threshold,                         # bkg_threshold=3.1,
            self.min_size,                              # min_size=4,
            self.max_size,                              # max_size=200,
            self.distortion_calibration_params,         # distortion_calibration_params=None,
            self.info_file,                             # log_file_path="log/test_log.txt",
            self.cfg.min_pattern_checking_stars,        # min_pattern_checking_stars=15,
            self.cfg.local_sigma_cell_size,             # local_sigma_cell_size=36,
            self.cfg.sigma_clipped_sigma,               # sigma_clipped_sigma=3.0,
            self.cfg.leveling_filter_downscale_factor,  # leveling_filter_downscale_factor=4,
            self.cfg.src_kernal_size_x,                 # src_kernal_size_x=3,
            self.cfg.src_kernal_size_y,                 # src_kernal_size_y=3,
            self.cfg.src_sigma_x,                       # src_sigma_x=1,
            self.cfg.src_dst,                           # src_dst=1,
            self.cfg.dilate_mask_iterations,            # dilate_mask_iterations=1,
            self.cfg.return_partial_images,             # return_partial_images=False,
            self.cfg.partial_results_path,              # partial_results_path="./partial_results",
            self.solver,                                # solver='solver1',
            self.level_filter,                          # level_filter: int = 9,
            self.cfg.ring_filter_type                   # ring_filter_type = 'mean'
        )

        if is_multiprocessing:
            # Run via multiprocessing pool
            return self.pool.apply_async(self.astro.do_astrometry, args=args)
        # Run directly
        return self.astro.do_astrometry(*args)

    @staticmethod
    def get_result(result):
        """Handles both AsyncResult and direct function call."""
        if isinstance(result, AsyncResult):
            return result.get()  # Retrieve result from AsyncResult
        return result  # Direct function result

    def camera_take_image(self, cmd=None, is_operation=False, is_test=False):
        """
        Take camera image and
        1. Save RAW and scale downs
        2. Astrometry Solve
        3. Calculate angular velocity
        4. Save Final and scaled image

        Uses multiprocessing if enabled.
        # TODO: Evaluate the use of multiprocessing.
        """
        print('\n')
        t0 = time.monotonic()
        if cmd is None:
            is_raw = False
        else:
            is_raw = cmd.mode == 'raw'  # cmd.mode == Raw
            self.solver = self.solver if is_raw else cmd.mode  # cmd.mode == solver1  or solver2
            self.cfg.set_dynamic(solver=self.solver)

        dt = datetime.datetime.now().isoformat()  # Get current ISO timestamp
        if is_raw:
            self.logit(f"Taking photo: raw: {is_raw} mode: {self.flight_mode} @{dt}", color='cyan')  # Timestamp at END
        else:
            self.logit(f"Taking photo: operation: {is_operation} solver: {self.solver} mode: {self.flight_mode} @{dt}", color='cyan')  # Timestamp at END

        self.curr_time = time.monotonic()
        dt = datetime.datetime.now(timezone.utc)
        utc_time = dt.replace(tzinfo=timezone.utc)
        curr_utc_timestamp = utc_time.timestamp()
        # curr_serial_utc_timestamp = self.serial_utc_update + time.monotonic() - self.serial_time_datum
        timestamp_string = dt.strftime(self.timestamp_fmt)  # Used for filename, ':' should not be used.

        self.info_file = f"{self.get_daily_path(self.cfg.final_path)}/log_{timestamp_string}.txt"

        if self.focuser.aperture_position == 'closed':
            self.focuser.open_aperture()

        # Set the ASI_IMG_RAW16
        self.camera.set_image_type(asi.ASI_IMG_RAW16)
        self.camera.set_roi(bins=self.cfg.roi_bins)

        # Capture/Take image
        if not is_test:
            t1 = time.monotonic()
            try:
                self.logit(f'Capturing image @{get_dt(t0)}.', color='green')
                self.curr_img = self.camera.capture()
                self.img_cnt += 1
                self.logit(f'Camera image captured in {get_dt(t1)} @{get_dt(t0)} shape: {self.curr_img.shape}.', color='green')
            except Exception as e:
                self.logit(f'Capture error: {e}', level='error', color='red')
                if not self.chamber_mode:
                    return

        # Chamber mode/test mode?
        if self.chamber_mode or is_test:
            self.curr_img = self.camera_dummy.capture()
            self.logit(f'Taking DUMMY capture from file: {self.camera_dummy.filename}', 'warning', color='blue')

        # Close aperture if not in autonomous mode (doing single take image)
        if not is_operation:
            self.focuser.close_aperture()

        # Get and log camera settings
        camera_settings = self.camera.get_control_values()
        self.logit(f"exposure time (us) : {camera_settings['Exposure']}")
        current_gain = camera_settings['Gain']
        self.logit(f"gain (cB) : {current_gain}")
        self.logit(f'curr image length: {len(self.curr_img)} prev image length: {len(self.prev_img)}')
        self.logit(f"Max pixel: {np.max(self.curr_img)}, Min pixel: {np.min(self.curr_img)}. ")

        # Write image info to log
        self.info_add_image_info(camera_settings, curr_utc_timestamp)

        # Periodic Autogain (in autonomous mode)
        # Single iteration keeps the camera in range.
        # TODO: If the autonomous mode is disabled it will lose track so needs to rerun the full autogain interactions again
        if is_operation and (self.img_cnt % self.cfg.autogain_update_interval) == 0 and not is_test:
            new_gain, gain_exec = timed_function(self.autogain_maintanence, camera_settings)
            self.logit(f'Single image autogain completed: interval: {self.cfg.autogain_update_interval} current gain: {current_gain} new gain: {new_gain} in {gain_exec:.4f} seconds.', color='cyan')

        # perform astrometry
        image_file = timestamp_string
        self.logit(f"curr image name : {image_file}")

        # Raw Image
        self.curr_image_name = None
        self.curr_image_info = None

        # Raw Image Scaled
        self.curr_scaled_name = None
        self.curr_scaled_info = None

        # Final Overlay Image
        self.foi_name = None
        self.foi_info = None

        # Final Overlay Image Scaled
        self.foi_scaled_name = None
        self.foi_scaled_info = None

        # Step 1: SAVE RAW Images
        if self.save_raw:
            # Use a multiprocessing Pool
            # Save image to SSD, SD card, inspection_images
            save_raws_result = self.pool.apply_async(
                save_raws,
                args=(self.curr_img,
                      self.get_daily_path(self.cfg.ssd_path), self.get_daily_path(self.cfg.sd_card_path),
                      image_file,
                      self.cfg.scale_factors, self.cfg.resize_mode,
                      self.cfg.raw_scale_factors, self.cfg.raw_resize_mode,
                      self.cfg.png_compression, self.is_flight,
                      self.cfg.inspection_settings
                      )
            )

            # Perform astrometry
            # TODO: Future Performance Enhancement and Cleanup, the RAWs should be done in a thread
            if False:
                self.logit(f'Fetching Astrometry (multiprocessing) @{get_dt(t0)}.', color='green')
                astrometry_result = None if is_raw else self.do_astrometry()

        # Step 3: Do Astrometry
        if not is_raw:
            self.logit(f'Fetching Astrometry (in main process) @{get_dt(t0)}.', color='green')
            self.curr_image_name = f"{self.cfg.ssd_path}/{image_file}-raw.png"
            astrometry_result = self.do_astrometry(is_multiprocessing=False)

        # Step 4: If RAWS wait for results from multiprocessing
        if self.save_raw:
            # Wait for both processes to complete
            save_raws_result.wait()
            # Get the save_raws return values (filenames)
            self.curr_image_name, self.curr_image_info, self.curr_scaled_name, self.curr_scaled_info = save_raws_result.get()
            # Send image to GUI/client
            self.server.write(f'Current image filename: {self.curr_image_name}')
            self.server.write(self.curr_scaled_name, data_type='image_file')

        if is_raw:
            # Only took the raw image, no solving
            self.logit(f'camera_take_image (RAW image/no solving) completed in {get_dt(t0)}.', color='green')
            return

        # Get the return values
        self.astrometry, self.curr_star_centroids, self.contours_img = self.get_result(astrometry_result)

        # TODO: Assumed the do_astrometry was success. None (no solution) needs to be handled.
        if self.astrometry is None:
            self.log.warning(f'No solution for astrometry.')
            self.server.write(f'Solving of image (do_astrometry) did not produce any solutions.', 'warning')
            return

        # Append additional image metadata log to a file.
        self.info_add_astro_info()

        # Calculate Angular velocity
        # Old implementation using images
        # self.calc_angular_velocity(curr_utc_timestamp)
        # New implementation using two astrometry results
        solutions_dt = self.curr_time - self.prev_time
        angular_velocity = {}
        self.omega = [float('nan')] * 3
        with suppress(KeyError, ValueError, TypeError):
            angular_velocity = self.compute_angular_velocity(self.astrometry, self.prev_astrometry, solutions_dt)
            keys_in_order = ["roll_rate", "az_rate", "el_rate"]
            self.omega = [angular_velocity[key] for key in keys_in_order]
          # Debug Variant
        if False:
            try:
                angular_velocity = self.compute_angular_velocity(self.astrometry, self.prev_astrometry, solutions_dt)
            except Exception as e:
                self.log.error(f"Failed to compute angular velocity: {e}")
                self.log.error(f"Exception type: {type(e).__name__}")
                self.log.error(f"Stack trace:\n{traceback.format_exc()}")
                angular_velocity = {}

        # Display overlay image
        # foi ~ Final Overlay Image
        if is_operation or True:
            self.log.debug('Adding overlay.')
            self.foi_name, self.foi_info, self.foi_scaled_name, self.foi_scaled_info = display_overlay_info(
                self.contours_img, timestamp_string,
                self.astrometry, self.omega, False,
                self.curr_image_name,
                self.get_daily_path(self.cfg.final_path),
                self.cfg.partial_results_path,
                self.cfg.scale_factors,
                self.cfg.resize_mode,
                self.cfg.png_compression,
                self.is_flight)
            # Create/update symlink to last info file
            if self.is_flight:
                create_symlink(self.cfg.web_path, self.foi_name, 'last_final_overlay_image.png')
                create_symlink(self.cfg.web_path, self.foi_scaled_name, 'last_final_overlay_image_downscaled.png')
            if self.foi_scaled_name is not None:
                self.server.write(self.foi_scaled_name, data_type='image_file', dst_filename=self.curr_scaled_name)
            else:
                cprint('Error no scaled image', color='red')
                self.log.error('foi_scaled_name was None, no downscaled image generated.')
            self.prev_time = self.curr_time
            self.prev_img = self.curr_img
            self.prev_astrometry = self.astrometry
            self.prev_star_centroids = self.curr_star_centroids
            self.prev_image_name = self.curr_image_name


        # Send the file to clients
        #  Flight computer ASTRO POSITION:
        _dt = datetime.datetime.fromtimestamp(curr_utc_timestamp)
        current_ts = _dt.isoformat()
        position = {
            'timestamp':  current_ts,
            'solver': self.solver,
            'astro_position': [None, None, None],
            'FOV': None,
            'RMSE': None,
            'RMS': [None, None, None],
            'sources': 0,
            'matched_stars': 0,
            'probability': 1,
            'angular_velocity': [None, None, None]  # Initialize empty angular velocity dict
        }
        if self.astrometry is not None:
            position['astro_position'] = [self.astrometry.get('RA'), self.astrometry.get('Dec'), self.astrometry.get('Roll')]
            position['FOV'] = self.astrometry.get('FOV')
            position['RMSE'] = self.astrometry.get('RMSE')
            position['RMS'] = [self.astrometry.get('RA_RMS'), self.astrometry.get('Dec_RMS'),  self.astrometry.get('Roll_RMS')]
            position['sources'] = self.astrometry.get('precomputed_star_centroids')
            position['matched_stars'] = self.astrometry.get('Matches')
            position['probability'] = self.astrometry.get('Prob')
            position['angular_velocity'] = [angular_velocity.get('roll_rate'), angular_velocity.get('az_rate'), angular_velocity.get('el_rate')]
        # Add position to the positions queue
        self.positions_queue.put(position)
        # Save to astro file
        save_to_json(position, self.cfg.astro_path)

        self.curr_scaled_name = self.curr_scaled_name or self.foi_scaled_name
        info_filename = f'{self.curr_scaled_name[:-4]}.txt'  # .png -> .txt
        self.server.write(self.info_file, data_type='info_file', dst_filename=info_filename)
        # Copy to sd_card and ssd_path
        if self.is_flight:
            shutil.copy2(self.info_file, self.get_daily_path(self.cfg.sd_card_path))
            shutil.copy2(self.info_file, self.get_daily_path(self.cfg.ssd_path))
        # Create/update symlink to last info file
        create_symlink(self.cfg.web_path, self.info_file, 'last_info_file.txt')

        self.logit(f'camera_take_image completed in {get_dt(t0)}.', color='green')

    def _radec_roll_to_rotation(self, ra_deg, dec_deg, roll_deg):
        """
        Convert RA, Dec, and Roll angles to a rotation matrix using standard astronomical transformations.

        This function constructs the rotation matrix that transforms from the International
        Celestial Reference Frame (ICRF) to the camera's body frame. The transformation
        follows the standard astronomical convention:

        1. Rotate by RA around the Z-axis (pointing toward North Celestial Pole)
        2. Rotate by -Dec around the Y-axis (negative because declination increases northward
           from the celestial equator, which corresponds to a negative rotation in the
           right-handed coordinate system where Y points east)
        3. Rotate by Roll around the X-axis (now aligned with the camera boresight)

        The resulting matrix R satisfies: v_camera = R @ v_icrf

        Args:
            ra_deg (float): Right Ascension in degrees
            dec_deg (float): Declination in degrees
            roll_deg (float): Roll angle in degrees

        Returns:
            numpy.ndarray: 3x3 rotation matrix representing the camera orientation
        """
        ra = np.deg2rad(ra_deg)
        dec = np.deg2rad(dec_deg)
        roll = np.deg2rad(roll_deg)

        # Standard astronomical sequence: R = R_roll(X) @ R_-dec(Y) @ R_ra(Z)
        R_z = Rotation.from_rotvec([0, 0, ra]).as_matrix()  # Rotate by RA around Z
        R_y = Rotation.from_rotvec([0, -dec, 0]).as_matrix()  # Rotate by -Dec around Y
        R_x = Rotation.from_rotvec([roll, 0, 0]).as_matrix()  # Rotate by Roll around X

        # Combine the rotations: First RA, then Dec, then Roll.
        return R_x @ R_y @ R_z

    def _decompose_rotation_to_local_axes(self, R1, R2):
        """
        Decompose the relative rotation between two orientations into camera-axis components.

        This function calculates the components of the angular velocity vector that would
        transform the camera from orientation R1 to orientation R2. The method involves:

        1. Computing the relative rotation matrix: R_rel = R2 @ R1.T
        2. Extracting the rotation vector from R_rel using matrix logarithm
        3. Projecting this rotation vector onto the camera's local axes (X, Y, Z) defined by R1

        The rotation vector represents the integrated angular displacement (ω * Δt) between
        the two orientations. Its projection onto the camera axes gives the components of
        this displacement along each axis.

        Args:
            R1 (numpy.ndarray): 3x3 rotation matrix representing initial camera orientation
            R2 (numpy.ndarray): 3x3 rotation matrix representing final camera orientation

        Returns:
            tuple: Components of the rotation vector along camera axes:
                - d_roll_rad: Component along Z-axis (roll/boresight direction) in radians
                - d_az_rad: Component along X-axis (azimuth/right direction) in radians
                - d_el_rad: Component along Y-axis (elevation/up direction) in radians
        """
        # Calculate the relative rotation matrix from frame1 to frame2
        R_rel = R2 @ R1.T  # This is R_{2<-1} = R2 * R1^{-1}

        # Extract rotation vector from relative rotation matrix
        # This vector represents the axis*angle rotation that transforms R1 to R2
        # Its magnitude is the rotation angle in radians, direction is rotation axis
        delta_theta_vec = Rotation.from_matrix(R_rel).as_rotvec()

        # The rotation vector magnitude should be < π for unambiguous interpretation
        rotation_angle = np.linalg.norm(delta_theta_vec)
        if rotation_angle > np.pi:
            # Handle large rotations by taking the complementary angle
            delta_theta_vec = delta_theta_vec * (1 - 2 * np.pi / rotation_angle)

        # Get the axes of the initial camera frame (R1)
        # The columns of R1 are the world axes expressed in the camera frame
        right = R1[:, 0]  # Camera's X-axis (azimuth/right direction)
        up = R1[:, 1]  # Camera's Y-axis (elevation/up direction)
        forward = R1[:, 2]  # Camera's Z-axis (roll/boresight direction)

        # Project the rotation vector onto the initial camera axes
        # These projections give the components of angular displacement along each axis
        d_roll = np.dot(delta_theta_vec, forward)  # Z-axis component
        d_az = np.dot(delta_theta_vec, right)  # X-axis component
        d_el = np.dot(delta_theta_vec, up)  # Y-axis component

        return d_roll, d_az, d_el

    def compute_angular_velocity(self, astrometry_curr, astrometry_prev, delta_t_sec):
        """
        Compute angular velocity between two astrometry solutions.

        This function calculates the instantaneous angular velocity of the camera
        between two orientations specified by their RA, Dec, and Roll values. The
        calculation involves:

        1. Converting both orientations to rotation matrices using standard
           astronomical transformations
        2. Computing the relative rotation between the orientations
        3. Decomposing this rotation into components along the camera's local axes
        4. Converting these displacement components to angular velocity by dividing
           by the time difference

        The resulting angular velocity vector components represent the camera's
        rotational speed around its own axes (roll, azimuth, elevation) during
        the time interval.

        Args:
            astrometry_curr (dict): Current astrometry solution with keys:
                'RA', 'Dec', 'Roll' (all in degrees)
            astrometry_prev (dict): Previous astrometry solution with keys:
                'RA', 'Dec', 'Roll' (all in degrees)
            delta_t_sec (float): Time difference between solutions in seconds

        Returns:
            dict: Dictionary containing angular velocities in deg/s for:
                - roll_rate: Rotation around camera's Z-axis (boresight/roll)
                - az_rate: Rotation around camera's X-axis (azimuth/right)
                - el_rate: Rotation around camera's Y-axis (elevation/up)

        Raises:
            KeyError: If required keys are missing from input dictionaries
            ValueError: If delta_t_sec is zero or negative
        """

        no_velocity = {
            "roll_rate": float('nan'),
            "az_rate": float('nan'),
            "el_rate": float('nan')
        }

        if delta_t_sec <= 0:
            raise ValueError("delta_t_sec must be positive")

        # Extract orientation parameters
        try:
            ra1, dec1, roll1 = (astrometry_prev['RA'], astrometry_prev['Dec'], astrometry_prev['Roll'])
            ra2, dec2, roll2 = (astrometry_curr['RA'], astrometry_curr['Dec'], astrometry_curr['Roll'])
        except KeyError as e:
            return no_velocity
            # raise KeyError(f"Missing required key in astrometry dict: {e}")

        # Combined validation in one block
        all_values = [ra1, dec1, roll1, ra2, dec2, roll2]

        if any(v is None or np.isnan(v) for v in all_values):
            invalid_type = "None" if any(v is None for v in all_values) else "NaN"
            self.log.warning(f"One or more orientation values are {invalid_type}, returning NaN angular velocity")
            return no_velocity

        # Check for identical orientations to avoid division by numerical noise
        atol = 1e-10
        if (np.isclose(ra1,ra2, atol=atol) and
                np.isclose(dec1, dec2, atol=atol) and
                np.isclose(roll1, roll2, atol=atol)):
            return no_velocity

        # Convert to rotation matrices
        R1 = self._radec_roll_to_rotation(ra1, dec1, roll1)
        R2 = self._radec_roll_to_rotation(ra2, dec2, roll2)

        # Decompose relative rotation into components of the rotation vector
        d_roll_rad, d_az_rad, d_el_rad = self._decompose_rotation_to_local_axes(R1, R2)

        # Convert components of rotation vector to angular velocities (rad/s to deg/s)
        angular_velocity = a_v= {
            "roll_rate": np.rad2deg(d_roll_rad) / delta_t_sec,
            "az_rate": np.rad2deg(d_az_rad) / delta_t_sec,
            "el_rate": np.rad2deg(d_el_rad) / delta_t_sec
        }

        # Write results to a file
        self.info_add_angular_velocity_info(angular_velocity, delta_t_sec)
        # Get current orientation for logging
        ra = astrometry_curr.get('RA', 'N/A')
        dec = astrometry_curr.get('Dec', 'N/A')
        roll = astrometry_curr.get('Roll', 'N/A')

        # Log the results
        self.log.debug(f'Angular velocity: {angular_velocity}')
        self.server.write(f'{self.curr_time}, RA: {ra}, DEC: {dec}, ROLL: {roll} [deg]')
        self.server.write(f'{self.curr_time}, Ω_roll: {a_v["roll_rate"]}, Ω_az: {a_v["az_rate"]}, Ω_el: {a_v["el_rate"]} [deg/sec]')
        self.log.debug(f'done')

        return angular_velocity

    def camera_run_autogain(self, cmd):
        """
        TODO: Check if using native zwoasi auto_exposure could be of use
        Here are the possible values for auto_exposure:
            'Exposure': Automatically adjusts the exposure time.
            'Gain': Automatically adjusts the gain.
            'Gamma': Automatically adjusts the gamma.
            'Brightness': Automatically adjusts the brightness.

        self.camera.auto_exposure() # Will use default values: ('Exposure', 'Gain')
          same as: self.camera.auto_exposure(('Exposure', 'Gain'))
        self.auto_exposure(auto=('Exposure',))
        """

        desired_max_pixel_value = cmd.desired_max_pixel_value
        # TODO: max_pixel_value NOT USED!!!
        self.logit('Commanded to run autogain.')
        ## Doing auto gain##
        timestamp_string = current_timestamp(self.timestamp_fmt)
        auto_gain_image_path = self.cfg.auto_gain_image_path_tmpl.format(timestamp_string=timestamp_string)
        os.mkdir(auto_gain_image_path)
        self.best_gain_value = self.do_auto_gain_routine(
            auto_gain_image_path, self.best_gain_value, desired_max_pixel_value,
            self.pixel_saturated_value, self.pixel_count_tolerance)
        self.logit(f"Autogain routine. Best gain: {self.best_gain_value}")

    def camera_run_autoexposure(self, cmd):
        desired_max_pixel_value = cmd.desired_max_pixel_value
        self.logit('Commanded to run autoexposure.')
        timestamp_string = current_timestamp(self.timestamp_fmt)
        auto_exposure_image_path = self.cfg.auto_gain_image_path_tmpl.format(timestamp_string=timestamp_string)
        os.mkdir(auto_exposure_image_path)
        self.best_exposure_time = self.do_auto_exposure_routine(
            auto_exposure_image_path, self.best_exposure_time,
            desired_max_pixel_value, self.pixel_saturated_value, self.pixel_count_tolerance)
        self.logit(f"Autoexposure routine. Best exposure: {self.best_exposure_time}")

    def camera_resume(self, cmd):
        self.logit(f"Command to resume star tracker operation. Setting solver: {cmd.solver} cadence: {cmd.cadence}s.", color='green')
        # if self.operation_enabled:
        #     return
        self.focuser.open_aperture()
        self.operation_enabled = True
        self.img_cnt = 0
        self.solver = cmd.solver
        self.astro.solver = self.solver  # Update also the astro instance
        self.cadence = float(cmd.cadence)
        self.time_interval = int(self.cadence * 1e6)  # Convert cadence [s] into time_interval [micros]
        self.cfg.set_dynamic(solver=self.solver, time_interval=self.time_interval, run_autonomous=self.operation_enabled)
        self.server.write(f"Operation resumed. Solver: {self.solver}  cadence: {self.cadence}s")

    def camera_pause(self):
        self.logit("Command to pause PUEO Star Tracker operation")
        self.operation_enabled = False
        time.sleep(1)
        if self.focuser.aperture_position == 'opened':
            self.focuser.close_aperture()
        self.cfg.set_dynamic(run_autonomous=self.operation_enabled)
        self.server.write("Operation paused.")

    def camera_run_autofocus(self, cmd):
        # TODO: Implement two different autofocus routines one for covariance and other for source_diameter autofocus type
        # Example: run_autofocus <focus_type> <start_focus_pos> <stop_focus_pos>
        # run_autofocus covariance <start_focus_pos> <stop_focus_pos>
        # run_autofocus two_step <focus_coefficient>

        # focus_type is one of 'covariance', 'source_diameter', 'two_step'
        self.logit('Commanded to refocus (take focus sequence).')

        focus_method = cmd.focus_method

        # focus_method ['sequence_contrast', 'sequence_diameter', 'sequence_twostep']
        # start_position
        # stop_position
        # stepcount
        # focus_coefficient

        # Prepare camera for taking images
        if self.focuser.aperture_position == 'closed':
            self.focuser.open_aperture()

        self.camera.set_image_type(asi.ASI_IMG_RAW8)
        self.camera.set_roi(bins=self.cfg.roi_bins)

        # TODO: Implement the third routine for two_step method, also add the param: focus_coefficient to do_autofocus_routine
        timestamp_string = current_timestamp("%y%m%d_%H%M%S.%f")
        self.focus_image_path = self.cfg.focus_image_path_tmpl.format(timestamp_string=timestamp_string)
        os.mkdir(self.focus_image_path)

        # Recalibrate focuser position
        self.min_focus_position, self.max_focus_position = self.focuser.home_lens_focus()

        self.best_focus, self.stdev = self.do_autofocus_routine(self.focus_image_path, int(cmd.start_position),
                                                                int(cmd.stop_position),
                                                                # self.cfg.fine_focus_step_count,
                                                                int(cmd.step_count),
                                                                self.max_focus_position,
                                                                cmd.focus_method)

        self.logit(f"Manual best focus: {self.best_focus} stddev: {self.stdev} method: {focus_method}")

    def camera_set_aperture_position(self, cmd):
        aperture_position = int(cmd.aperture_position)
        f_val = self.focuser.f_stops[aperture_position] if len(self.focuser.f_stops) > aperture_position else 'f??'
        self.logit(f'Changing focuser aperture to: {aperture_position} [{f_val}]', color='blue')
        self.server.write(f'Changing focuser aperture to: {aperture_position} [{f_val}]')
        self.focuser.move_aperture_absolute(aperture_position)
        self.cfg.set_dynamic(lab_best_aperture_position=aperture_position)
        pos = f_val = None
        with suppress(Exception):
            pos, f_val = self.focuser.get_aperture_position()
        self.server.write(f'Focuser position: {pos} [{f_val}]')

    def camera_get_aperture_position(self):
        # returns pos, f_val
        return self.focuser.get_aperture_position()

    def camera_set_exposure_time(self, cmd):
        manual_exposure_time = cmd.exposure_time
        self.logit(f'Changing camera exposure to: {manual_exposure_time} [us]')
        self.server.write(f"Changing camera exposure to: {manual_exposure_time}")
        self.camera.set_control_value(asi.ASI_EXPOSURE, int(manual_exposure_time))  # units microseconds,
        self.cfg.set_dynamic(lab_best_exposure=int(manual_exposure_time))  # Update dynamic lab_best_exposure

    def camera_set_gain(self, cmd):
        gain_setting = cmd.gain
        self.camera.set_control_value(asi.ASI_GAIN, int(gain_setting))
        self.logit(f'Changing camera gain to: {gain_setting} [cB]')
        self.cfg.set_dynamic(lab_best_gain=int(gain_setting))  # Update dynamic lab_best_gain

    def camera_get_values(self):
        r = self.camera.get_control_values()
        self.server.write('-- Camera values --')
        for k, v in r.items():
            self.server.write(f'  {k}: {v}')

        self.server.write('-- Focuser values --')
        self.server.write(f'  Min position: {self.min_focus_position}')
        self.server.write(f'  Max position: {self.max_focus_position}')
        self.server.write(f'  Focus position: {self.focuser.focus_position}')
        self.server.write(f'  Aperture position: {self.focuser.aperture_position}')
        self.server.write('-- Server values --')
        self.server.write(f'  CPU Temperature: {self.get_cpu_temp()} °C')

    def camera_set_focus_position(self, cmd):
        focus_position = int(cmd.focus_position)
        self.cfg.set_dynamic(lab_best_focus=focus_position)  # Update dynamic lab_best_focus
        self.logit(f'Changing focus to: {focus_position} [counts]')
        focus_position = self.focuser.move_focus_position(focus_position)
        self.logit(f'New Focus Position: {focus_position}')

    def camera_delta_focus_position(self, cmd):
        delta_focus = cmd.focus_delta
        self.logit(f'Adjusting focus by: {delta_focus} [us]')
        curr_focus_position = self.focuser.get_focus_position()
        new_focus_position = curr_focus_position + int(delta_focus)
        adjusted_focus_position = self.focuser.move_focus_position(new_focus_position)
        self.cfg.set_dynamic(lab_best_focus=adjusted_focus_position)  # Update dynamic lab_best_focus
        self.logit(f'Actual adjusted focus position: {adjusted_focus_position}')

    def camera_sample_distortion(self):
        self.logit('Sampling_distortion.')
        # optimize calibration parameters
        self.distortion_calibration_params = self.astro.optimize_calibration_parameters(
            self.is_trail,
            self.number_sources,
            self.bkg_threshold,
            self.min_size, self.max_size,
            calibration_images_dir="../data/calibration_images",
            log_file_path="log/calibration_log.txt",
            calibration_params_path="calibration_params.txt",
            update_calibration=self.update_calibration,
            min_pattern_checking_stars=self.cfg.min_pattern_checking_stars,
            local_sigma_cell_size=self.cfg.local_sigma_cell_size,
            sigma_clipped_sigma=self.cfg.sigma_clipped_sigma,
            leveling_filter_downscale_factor=self.cfg.leveling_filter_downscale_factor,
            src_kernal_size_x=self.cfg.src_kernal_size_x,
            src_kernal_size_y=self.cfg.src_kernal_size_y,
            src_sigma_x=self.cfg.src_sigma_x,
            src_dst=self.cfg.src_dst,
            dilate_mask_iterations=self.cfg.dilate_mask_iterations,
            scale_factors=self.cfg.scale_factors,
            resize_mode=self.cfg.resize_mode,
            level_filter=self.level_filter,
            ring_filter_type=self.cfg.ring_filter_type
        )
        self.server.write("Checking distortion.")

    def camera_enable_distortion_correction(self, cmd):
        """
        For now let this be a tuple and use pincushin/barrel distortion.
        """
        # distortion = self.distortion_calibration_params  # for now let this be a tuple and use pincushin/barrel distortion.
        # TODO: Check distortion_calibration_params with Windell
        self.distortion_calibration_params = {"FOV": cmd.fov,
                                              "distortion": [cmd.distortion_coefficient_1,
                                                             cmd.distortion_coefficient_2]}
        self.distortion = self.distortion_calibration_params

    def camera_disable_distortion_correction(self):
        self.distortion = None

    def camera_input_gyro_rates(self, cmd):
        # TODO: Check with Windell
        body_x = cmd.omega_x
        body_y = cmd.omega_y
        body_z = cmd.omega_z
        self.log.info(f'Reported gyro rates X:{body_x} Y:{body_y} Z:{body_z}')

    def camera_gyro_rates(self, cmd):
        pass
        # old code
        # bodyx = float(command.split(" ")[1])
        # bodyy = float(command.split(" ")[2])
        # bodyz = float(command.split(" ")[3])
        # self.logit(f'Reported gyro rates X:{bodyx} Y:{bodyy} Z:{bodyz}')

    def camera_update_time(self, cmd):
        # TODO: Implement update_time
        new_time = cmd.new_time  # A timestamp
        dt = datetime.datetime.now(timezone.utc)
        # Calculate the time difference as a timedelta object
        delta_td = dt - new_time

        # Convert the timedelta to seconds
        delta_t = delta_td.total_seconds()

        utc_time = dt.replace(tzinfo=timezone.utc)
        curr_utc_timestamp = utc_time.timestamp()
        serial_utc_update = curr_utc_timestamp + delta_t
        serial_time_datum = time.monotonic()
        self.logit(f"Given UTC update via serial. {delta_t}")

    def camera_power_cycle(self, cmd):
        """Powercycle and reinitialise camera/focuser via power/on/off via SBC VersaLogic API using GPIO"""
        power_cycle_wait_s = self.cfg.power_cycle_wait / 1000

        self.log.info('Camera Power Cycle')
        with suppress(RuntimeError):
            self.versa_api.cycle_pin(self.cfg.sbc_dio_camera_pin, self.cfg.sbc_dio_cycle_delay, self.cfg.sbc_dio_default)
        self.camera.power_cycle(power_cycle_wait_s)

        self.log.info('Focuser Power Cycle')
        with suppress(RuntimeError):
            self.versa_api.cycle_pin(self.cfg.sbc_dio_focuser_pin, self.cfg.sbc_dio_cycle_delay, self.cfg.sbc_dio_default)
        self.focuser.power_cycle(power_cycle_wait_s)

    def camera_home_lens(self, cmd):
        c_min = self.min_focus_position
        c_max = self.max_focus_position
        self.min_focus_position, self.max_focus_position = self.focuser.home_lens_focus()
        self.logit(
            f'Home Lens completed: new min/max positions: {self.min_focus_position}/{self.max_focus_position} old: {c_min}/{c_max}')

    def camera_check_lens(self, cmd):
        result = self.focuser.check_lens_focus()
        self.logit(f'Check Lens completed: results: {result}')
        return result

    def camera_set_level_filter(self, cmd):
        level = cmd.level
        self.level_filter = int(level)
        self.logit(f'Changing star detection level filter to: {level} pixels')
        self.cfg.set_dynamic(level_filter=self.level_filter)

    @property
    def level_filter(self):
        """Getter for level_filter (no arguments allowed)."""
        return self._level_filter

    @level_filter.setter
    def level_filter(self, value: bool):
        """Setter for level_filter."""
        self._level_filter = value

    @property
    def chamber_mode(self):
        """Getter for chamber_mode (no arguments allowed)."""
        return self._chamber_mode

    @chamber_mode.setter
    def chamber_mode(self, value: bool):
        """Setter for chamber_mode."""
        self._chamber_mode = value

    @property
    def flight_mode(self):
        """Getter for flight_mode (no arguments allowed)."""
        return self._flight_mode

    @flight_mode.setter
    def flight_mode(self, value):
        """Setter for flight_mode (validates input)."""
        allowed_modes = {'preflight', 'flight'}  # Use lowercase for consistency
        if value.lower() not in allowed_modes:
            raise ValueError(f"Invalid flight mode: {value}. Must be one of {allowed_modes}")
        self._flight_mode = value.lower()  # Normalize to lowercase

    @property
    def is_flight(self) -> bool:
        """Determine if the Pueo is in flight mode.

        Returns:
            bool: True if the current mode is 'flight', False otherwise.

        Example:
            >>> if self.is_flight:
            ...     print("PUEO is in flight mode")
        """
        return self.flight_mode == 'flight'

    def wrapup_profiler(self, profiler):
        """Save profiling results"""
        profile_file = './logs/profile_stats'

        # Create a stream for the profiling results
        s = io.StringIO()
        ps = pstats.Stats(profiler, stream=s).sort_stats('cumulative')

        # Print the profiling results
        ps.print_stats()
        profiling_output = s.getvalue()
        self.log.info(profiling_output)

        # Save as Text
        with open(f"{profile_file}.txt", "w") as txt_file:
            txt_file.write(profiling_output)

        stats_dict = ps.stats

        # Save as JSON - Serialisation problem
        # json_data = {str(k): v for k, v in stats_dict.items()}  # Convert keys to strings for JSON compatibility
        # save_to_json(json_data, f"{profile_file}.json")

        # Save as Excel
        data = []
        for func, (cc, nc, tt, ct, callers) in stats_dict.items():
            data.append([str(func), cc, nc, tt, ct])

        df = pd.DataFrame(data, columns=["Function", "Calls", "Primitive Calls", "Total Time", "Cumulative Time"])
        df.to_excel(f"{profile_file}.xlsx", index=False)

        self.log.info(f"Profiling results saved to {profile_file}.txt, {profile_file}.json, and {profile_file}.xlsx")

    def run_test(self):
        tm = time.monotonic()
        self.chamber_mode = False # Note
        solvers = ['solver1', 'solver2', 'solver3']
        # solvers = ['solver1', ]

        # Prepare partial_results folder

        partial_results = 'partial_results'
        is_profile = True

        profiler = cProfile.Profile()
        results = {}

        test_data = {
            # 'sigma': 2.5,   # Ranging from 2.0 - 5.1 step 0.1
            'max_size': 50,
            'binning': False,
            'return_binned': False,
            'use_binned_for_star_candidates': False,
            'detect_hot_pixels': False,
        }

        def run_all_solvers(cedar_sigma=None):
            for solver in solvers:
                files = self.camera_dummy.files
                for _ in tqdm(range(files), desc=f'Testing {solver}', ncols=100):
                    cmd = Command()
                    cmd.take_image(mode=solver)
                    delete_folder('partial_results')
                    t0 = time.monotonic()
                    try:
                        status = 'Success'
                        error = None
                        self.astrometry = {}
                        if is_profile:
                            # Start profiling
                            profiler.enable()
                            self.camera_take_image(cmd, is_operation=True, is_test=True)
                            # Stop profiling
                            profiler.disable()
                        else:
                            self.camera_take_image(cmd, is_operation=True)
                        error = None
                    except Exception as e:
                        status = 'Error'
                        error = ''
                        cprint(f'Error: {e}', 'red')
                        traceback.print_exception(e)
                        errors = [str(exc).strip() for exc in traceback.format_exception(e)]
                        error = "\n".join(errors)
                        self.log.error(error)
                        logit(e, color='red')
                        if is_profile:
                            with suppress(ValueError):
                                profiler.disable()
                    dt = time.monotonic() - t0
                    filename = self.camera_dummy.filename

                    result = {
                        'category': 'day_in_life',
                        'runtime': dt,
                        'raw_img_name': self.curr_image_name,
                        'raw_img_shape': self.curr_image_info,
                        'raw_img_size': self.camera_dummy.file_size,

                        'raw_ds_name': self.curr_scaled_name,
                        'raw_ds_shape': self.curr_scaled_info,
                        'raw_ds_size': get_file_size(self.curr_scaled_name),

                        'foi_name': self.foi_name,
                        'foi_shape': self.foi_info,
                        'foi_size': get_file_size(self.foi_name),

                        'foi_ds_name': self.foi_scaled_name,
                        'foi_ds_shape': self.foi_scaled_info,
                        'foi_ds_size': get_file_size(self.foi_scaled_name),

                        'astrometry': self.astrometry,
                        'solved': 'Yes' if self.astrometry and self.astrometry.get('RA', '') is not None else 'No',
                        # 'cedar_sigma': self.astro.cedar.sigma,
                        'status': status,
                        'error': error
                    }

                    sigma = None # Sigma is part of cedar detect search/solve
                    sigma_txt = f'-sigma-{sigma}' if sigma else ''
                    ext = os.path.basename(filename)[-4:]
                    filename = f'{os.path.basename(filename)[:-4]}{sigma_txt}{ext}'
                    if filename in results:
                        results[filename][solver] = result
                    else:
                        results[filename] = {solver: result}
                    # Capture
                    save_to_json(results, 'logs/results.json')

                    # Create Archive with results
                    is_archive = True
                    if is_archive:
                        save_to_json({solver: result}, f'{partial_results}/4.0 - results.json')
                        if self.foi_name and os.path.exists(self.foi_name):
                            shutil.copy2(self.foi_name, partial_results)
                        archive = f'{os.path.basename(filename)[:-4]}.zip'
                        archive_folder(f'output/{archive}', 'partial_results/')

        def float_range(start, stop, step):
            """Generates a range of float values."""
            while start < stop:
                yield start
                start += step

        # for sigma in float_range(2.5, 5.0, 0.1):
        # for sigma in float_range(2.0, 5.1, 0.1):
        #     sigma = round(sigma, 1)

        self.astro.test = True
        self.astro.test_data = test_data
        sigma = None
        # self.astro.test_data['sigma'] = sigma
        run_all_solvers(sigma)

        save_to_excel(results, 'logs/results.xlsx')
        self.wrapup_profiler(profiler)
        logit(f'Test Run Completed successfully in {get_dt(tm)}.', color='green')

    def main_loop(self):
        if self.cfg.run_test:
            self.run_test()
            raise KeyboardInterrupt()

        # Do we have updated config?
        # Read it
        if self.operation_enabled:
            # self.cfg.reload()
            self.camera_operation()
        else:
            # We only sleep in non-operation mode
            time.sleep(1)

    def camera_operation(self):
        """Controls timed camera capture operations with precise interval timing.

        This method manages periodic image capture according to a configured time interval.
        If not enough time has passed since the last capture, it sleeps for the remaining
        duration before returning. Ensures precise timing between consecutive captures.

        The timing logic:
        1. Checks if operation is enabled
        2. Calculates elapsed time since last capture (μs)
        3. If interval not met: sleeps remaining time
        4. If interval met: triggers new capture and updates timing reference

        Note:
            Uses monotonic time for drift-resistant interval calculations.

        Side Effects:
            - May sleep to maintain timing precision
            - Calls camera_take_image() when interval expires
            - Updates self.prev_time after successful captures

        Returns:
            None: Always returns None, either after sleeping or capturing
        """
        if not self.operation_enabled:
            return

        # Take new image every self.cfg.time_interval microseconds
        self.curr_time = time.monotonic()
        elapsed = int((self.curr_time - self.prev_time) * 1e6)  # microseconds

        if elapsed < self.time_interval:
            # Calculate remaining time in seconds and sleep
            remaining_time = (self.time_interval - elapsed) / 1e6  # convert to seconds
            time.sleep(max(0, remaining_time))  # ensure non-negative sleep time

        self.curr_time = time.monotonic()
        self.camera_take_image(is_operation=True)
        self.prev_time = self.curr_time  # Critical: Update timing reference AFTER capture

def init():
    """
    Initializes core settings and configuration for the application.

    This function handles:
    - Loading the configuration and setting up logging.
    - Initializing global variables, including the error log path and test mode.
    - Ensuring the error log is cleared at startup.

    Globals:
        TEST (bool): Indicates if the application is running in test mode.
        ERROR_LOG (str): Path to the error log file.

    """
    global config_file

    program_name = f'{__program__} v{__version__}'
    logit(program_name, attrs=['bold'], color='cyan')

    log = load_config(name='pueo', config_file=config_file,
                      log_file_name_token="server_log")  # Initialise logging, config.ini, ...
    log.debug('Main - star_comm_bridge')


if __name__ == "__main__":
    init()
    cfg1 = Config(f'conf/{config_file}', f'conf/{dynamic_file}')
    server = PueoStarCameraOperation(cfg1)
    server.run()

# Last line of this fine script, well almost
