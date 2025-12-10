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
# from datetime import datetime, timezone, timedelta

# External imports
import cv2
# import imageio.v3 as iio
# from astropy import modeling
from scipy.optimize import curve_fit
from PIL import Image
import matplotlib
matplotlib.use('Agg')   # non-GUI backend — renders to files only
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Custom imports
from lib.astrometry import Astrometry
from lib.common import get_file_size, DroppingQueue
from lib.common import init_common, load_config, logit, current_timestamp, cprint, get_dt, save_to_json, save_to_excel
from lib.common import archive_folder, delete_folder
from lib.config import Config
from lib.versa_logic_api import VersaAPI
from lib.camera import PueoStarCamera, DummyCamera
from lib.compute import Compute
from lib.focuser import Focuser
from lib.star_comm_bridge import StarCommBridge
from lib.utils import Utils
from lib.source_finder import SourceFinder
from lib.telemetry import Telemetry
from lib.commands import Command
from lib.fs_monitor import FSMonitor, MonitorCfg

# CONFIG - [GLOBAL]
config_file = 'config.ini'  # Note this file can only be changed HERE!!! IT is still config.ini, but just for ref.
dynamic_file = 'dynamic.ini'

__version__ = '1.00b'
__created_modified__ = '2024-10-22'
__last_modified__ = '2025-09-27'
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
        self.utils = Utils(self.cfg, self.log, self)  # Passing SERVER for parent access.
        self.sf = SourceFinder(self.cfg, self.log, self)  # Passing SERVER for parent access.
        # Params
        self.start_t0 = time.monotonic()
        self.status = 'Initializing'
        self.operation_enabled = self.cfg.run_autonomous
        self.img_cnt = 0 # Capture image counter!
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

        self.curr_img_dtc = datetime.datetime.now(timezone.utc)

        self.distortion_calibration_params = None

        self.max_focus_position = None
        self.min_focus_position = None
        self.return_partial_images = None
        self.info_file = self.prev_info_file = None
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
        self.foi_name = self.prev_foi_name = None
        self.foi_info = None
        self.foi_scaled_name = self.prev_foi_scaled_name = None
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

        # Example: 251130_142512.123456
        # self.timestamp_fmt = "%y%m%d_%H%M%S.%f"  # File name timestamp friendly format

        # Changed filename timestamps:
        #  - very readable
        #  - visually close to standard ISO
        #  - still lexicographically sortable
        #  - includes T + Z so it's clearly UTC
        #  - safe on Windows, Linux, macOS
        #  - no ambiguity
        # Example: 2025-11-30T14-25-12.123456Z
        self.timestamp_fmt = "%Y-%m-%dT%H-%M-%S.%fZ"  # 6 decimals = microseconds

        self.image_list = []
        self.image_filename_list = []

        self.astrometry = {}
        self.prev_astrometry = {}
        self.curr_star_centroids = None
        self.contours_img = None

        self._autogain_thread = None  # Initialize as None

        self.telemetry_queue = DroppingQueue(maxsize=self.cfg.fq_max_size)
        self.positions_queue = DroppingQueue(maxsize=self.cfg.fq_max_size)

        # Astrometry
        self.astro = Astrometry(self.cfg.ast_t3_database, self.cfg, self.log, server=self)

        # Create/update symlink to astro.json file
        self.utils.create_symlink(self.cfg.web_path, self.cfg.astro_path, 'astro.json')

        # Board API (VersaLogic)
        self.versa_api = VersaAPI()

        # Set Camera and Focuser Board GPIO pins to known state: LOW
        self.logit('Setting GPIO Pins For Focuser to False (LOW):', color='cyan')
        with suppress(RuntimeError):
            self.versa_api.set_pin(self.cfg.sbc_dio_focuser_pin, False)

        # Wait 5 secs after setting the on-off camera/focuser pins.
        time.sleep(5)

        self.logit('Setting GPIO Pins For Camera to False (LOW):', color='cyan')
        with suppress(RuntimeError):
            self.versa_api.set_pin(self.cfg.sbc_dio_camera_pin, False)

        # Wait 2 secs in case Camera/Focuser got started.
        time.sleep(2)

        # Create compute object

        self.compute = Compute(self.log)

        # Create monitor object
        self.monitor = FSMonitor(self.cfg, self.log)
        warning_pct = self.monitor.cfg.warning_pct
        critical_pct = self.monitor.cfg.critical_pct
        # Add paths:
        # Add root for disk-only checks as an example
        root = os.path.abspath(os.sep) if os.name != "nt" else os.path.splitdrive(os.getcwd())[0] + os.sep
        self.monitor.add_path("root", root, warning_pct=warning_pct, critical_pct=critical_pct, fs_type="internal")
        self.monitor.add_path("ssd", self.cfg.ssd_path, warning_pct=warning_pct, critical_pct=critical_pct, fs_type="data")
        self.monitor.add_path("sd_card", self.cfg.sd_card_path, warning_pct=warning_pct, critical_pct=critical_pct, fs_type="data")
        # Start monitoring
        self.monitor.run()

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
            f.write('--- Capture Properties ---\n')
            for k in sorted(settings.keys()):
                f.write(f'{k}: {settings[k]}\n')
            f.write(f'Timestamp: {timestamp_string}\n')
            f.write('\n')  # blank line after this block
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

    def focus_score_starfield_edge(self, img, top_frac=0.08):
        """
        Edge-contrast focus score for star fields (Laplacian version, no top-fraction).

        Steps:
        - Convert to float and normalize
        - Light Gaussian blur to suppress noise
        - Laplacian (second derivative)
        - Score = mean of |Laplacian| over the whole image
        """
        # 1) to float
        img_f = img.astype(np.float32, copy=False)

        # 2) normalize to [0, 1] by max (safe for 8- or 16-bit)
        m = float(img_f.max())
        if m <= 0 or not np.isfinite(m):
            return 0.0
        img_f /= m

        # 3) light smoothing
        img_smooth = cv2.GaussianBlur(img_f, (3, 3), 0)

        # 4) Laplacian (second derivative)
        lap = cv2.Laplacian(img_smooth, cv2.CV_32F, ksize=3)
        vals = np.abs(lap).ravel()  # magnitude of second derivative

        if vals.size == 0:
            return 0.0

        # 5) simple pooling: mean of all magnitudes (no top_frac)
        return float(vals.mean())

    def robust_avg_diameter(self, diameters, min_px=1.5, max_px=30.0, trim_frac=0.10):
        """
        Robust mean of star diameters:
        - drop non-positive and out-of-range diameters
        - sigma-clip outliers
        - trim upper/lower tails
        """
        d = np.asarray(diameters, dtype=float).ravel()
        d = d[np.isfinite(d)]

        # basic range filter (throw away obvious garbage)
        d = d[(d >= min_px) & (d <= max_px)]
        if d.size == 0:
            return None

        # sigma-clipping around median
        med = np.median(d)
        mad = 1.4826 * np.median(np.abs(d - med))
        if mad > 0:
            keep = np.abs(d - med) <= 3.0 * mad
            if np.any(keep):
                d = d[keep]

        if d.size < 5:
            return float(np.mean(d))

        # trim tails
        d = np.sort(d)
        k = int(round(trim_frac * d.size))
        if k > 0 and 2 * k < d.size:
            d = d[k:-k]

        return float(np.mean(d))

    def plt_savefig(self, plt, image_file, is_preserve=True):
        """Saves an image and sends it to the messenger."""
        self.logit(f'Saving image: {image_file}')
        plt.savefig(image_file)
        plt.close()
        self.server.write(image_file, data_type='image_file', is_preserve=is_preserve)

    def fit_best_focus(self, focus_image_path, focus_positions, focus_scores, diameter_scores,
                       max_focus_position, focus_method='sequence_contrast'):
        """
        Best-focus estimation from a focus scan.

        Logic:
          - Always try BOTH:
              * Gaussian fit on edge/contrast scores (focus_scores).
              * V-fit (abs_line) on diameter_scores.
          - If BOTH fits work, use the V-fit (diameter-based).
          - If only ONE fit works, use that one.
          - If NEITHER fit works, fall back to cfg.lab_best_focus and
            return cfg.sigma_error_value as stdev.

        Returns
        -------
        best_focus_pos : int
        stdev          : float (width/uncertainty from chosen fit,
                                 or cfg.sigma_error_value if no fit succeeded)
        """
        sigma_error = float(self.cfg.sigma_error_value)

        # Sort focus_positions & focus_scores like original code did
        list1, list2 = zip(*sorted(zip(focus_positions, focus_scores)))
        focus_positions, focus_scores = (list(t) for t in zip(*sorted(zip(list1, list2))))

        focus_positions = np.array(focus_positions, dtype=float)
        focus_scores = np.array(focus_scores, dtype=float)
        diameter_scores = np.array(diameter_scores, dtype=float)

        fit_points = self.cfg.fit_points_init  # Already an int

        # ------------------------------------------------------------------
        # 1) Gaussian fit on edge/contrast scores (existing "sequence_contrast")
        # ------------------------------------------------------------------
        gauss_best_pos = None
        gauss_stdev = None

        try:
            # initial conditions for the Gaussian fit
            mean0 = sum(np.multiply(focus_positions, focus_scores)) / sum(focus_scores)
            sigma0 = math.sqrt(
                sum(np.multiply(focus_scores, (focus_positions - mean0) ** 2)) /
                sum(focus_scores)
            )
            base_height0 = float(focus_scores.min())
            a0 = float(focus_scores.max())

            p0 = [a0, mean0, sigma0, base_height0]

            popt, pcov = curve_fit(self.gauss, focus_positions, focus_scores, p0=p0)

            fit_x = np.linspace(focus_positions.min(), focus_positions.max(), fit_points)
            mean = popt[1]
            sigma = abs(popt[2])
            height = popt[3]

            # Save plot
            plt.figure()
            plt.plot(focus_positions, focus_scores, 'b', label='Focus Data')
            plt.plot(fit_x, self.gauss(fit_x, *popt), 'r--', label='Gaussian fit')
            plt.legend()
            plt.xlabel('Focus Position, counts')
            plt.ylabel('Sequence Contrast/Covariance')
            plt.title(f'mean: {round(mean, 2)}, stdev: {round(sigma, 2)}, height: {round(height, 2)}')
            self.plt_savefig(plt, focus_image_path + 'focus_score.png')

            self.logit('Gaussian fit (edge) parameters:')
            self.logit(f'a = {popt[0]} +- {np.sqrt(pcov[0, 0])}')
            self.logit(f'X_mean = {popt[1]} +- {np.sqrt(pcov[1, 1])}')
            self.logit(f'sigma = {popt[2]} +- {np.sqrt(pcov[2, 2])}')
            self.logit(f'height = {popt[3]} +- {np.sqrt(pcov[3, 3])}')

            trial_best = int(round(mean))
            if 0 <= trial_best <= max_focus_position:
                gauss_best_pos = trial_best
                gauss_stdev = float(sigma)
            else:
                self.logit(
                    f'Gaussian best focus {trial_best} out of limits 0..{max_focus_position}; '
                    'ignoring Gaussian fit.'
                )
        except Exception as e:
            self.log.error(f'Gaussian fitting Error: {e}')
            self.logit(f"There was an error with Gaussian fitting: {e}")

        # ------------------------------------------------------------------
        # 2) V-fit on diameters using abs_line (existing "sequence_diameter")
        # ------------------------------------------------------------------
        vfit_best_pos = None
        vfit_stdev = None

        try:
            a0 = 1.0
            h0 = 0.5 * (focus_positions.min() + focus_positions.max())
            c0 = float(diameter_scores.min())

            p0_v = [a0, h0, c0]

            popt_v, pcov_v = curve_fit(self.abs_line, focus_positions, diameter_scores, p0=p0_v)

            fit_x_v = np.linspace(focus_positions.min(), focus_positions.max(), fit_points)
            a = popt_v[0]
            h = popt_v[1]  # V-vertex (focus position)
            c = popt_v[2]

            trial_best_v = int(round(h))

            self.logit('V-fit (diameter) parameters:')
            self.logit(f'a (slope) = {popt_v[0]} +- {np.sqrt(pcov_v[0, 0])}')
            self.logit(f'h (x offset) = {popt_v[1]} +- {np.sqrt(pcov_v[1, 1])}')
            self.logit(f'c (y offset) = {popt_v[2]} +- {np.sqrt(pcov_v[2, 2])}')

            plt.figure()
            plt.plot(focus_positions, diameter_scores, 'b', label='Focus Data')
            plt.plot(fit_x_v, self.abs_line(fit_x_v, *popt_v), 'r--', label='Absolute Line Fit')
            plt.legend()
            plt.xlabel('Focus Position, counts')
            plt.ylabel('Diameter')
            plt.title(f'x offset: {round(h, 2)}, y offset: {round(c, 2)}, slope: {round(a, 2)}')
            self.plt_savefig(plt, focus_image_path + 'diameters_score.png')

            if 0 <= trial_best_v <= max_focus_position:
                vfit_best_pos = trial_best_v
                # uncertainty in h as a proxy for stdev in focus units
                vfit_stdev = float(np.sqrt(pcov_v[1, 1]))
            else:
                self.logit(
                    f'V-fit best focus {trial_best_v} out of limits 0..{max_focus_position}; '
                    'ignoring V-fit.'
                )
        except Exception as e:
            self.log.error(f'V-fit (diameter) Error: {e}')
            self.logit(f"There was an error with V-fitting: {e}")

        # ------------------------------------------------------------------
        # 3) Decide which result to use
        # ------------------------------------------------------------------
        v_ok = vfit_best_pos is not None
        gauss_ok = gauss_best_pos is not None

        if v_ok and gauss_ok:
            self.logit(
                f"fit_best_focus: both fits OK; using V-fit (diameter) at pos={vfit_best_pos}, "
                f"Gaussian (edge) best at pos={gauss_best_pos} for comparison."
            )
            return int(vfit_best_pos), float(vfit_stdev)

        if v_ok:
            self.logit(
                f"fit_best_focus: using V-fit (diameter) at pos={vfit_best_pos}; "
                "Gaussian fit not usable."
            )
            return int(vfit_best_pos), float(vfit_stdev)

        if gauss_ok:
            self.logit(f"fit_best_focus: V-fit failed; using Gaussian (edge) fit at pos={gauss_best_pos}.")
            return int(gauss_best_pos), float(gauss_stdev)

        # ------------------------------------------------------------------
        # 4) Final fallback: lab_best_focus + sigma_error_value
        # ------------------------------------------------------------------
        self.logit(
            "fit_best_focus: both V-fit and Gaussian fit failed; "
            f"falling back to cfg.lab_best_focus={self.cfg.lab_best_focus} "
            f"and sigma_error_value={sigma_error}."
        )
        return int(self.cfg.lab_best_focus), sigma_error

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
            save_fig('temperature_data', plt)
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
        if self.cfg.enable_gui_data_exchange:
            tmp_filename = f'{basename}_histogram_tmp.png'
            self.plt_savefig(plt, tmp_filename, is_preserve=False)


    def do_auto_gain_routine(self, auto_gain_image_path, initial_gain_value,
                             desired_max_pix_value, pixel_saturated_value, pixel_count_tolerance,
                             max_iterations=None):
        """
        Autogain: adjust camera GAIN using a brightness control metric.

        Control strategy:
          * Primary metric: MAX pixel value in the image (drives bright star peaks).
          * If image is saturated (max == pixel_saturated_value):
              → fall back to 99.9th percentile of UNCLIPPED pixels (arr < pixel_saturated_value),
                to keep using a meaningful value instead of the pinned ADC ceiling.

        Behavior:
          * On every iteration, we capture a fresh image through capture_timestamp_save().
          * Compute:
              - max pixel value
              - min pixel value
              - p99.9 over all pixels (for logging)
              - count/fraction of saturated pixels
          * If not saturated → control_value = max pixel.
          * If saturated → control_value = 99.9th percentile of unclipped pixels (or max as last resort).
          * If |control_value - desired| <= pixel_count_tolerance → converged → stop.
          * Otherwise, adjust gain in the correct direction (up for dark, down for bright)
            by at least autogain_min_gain_step, until we converge or hit max_autogain_iterations.

        Saving policy:
          * Images are saved ONLY by capture_timestamp_save(), using auto_gain_image_path
            and inserted_string. This function assumes that the returned `basename` is the
            full base path (dir + filename root) for that image.

          * If self.cfg.return_partial_images is True:
              - Save histogram text file every iteration.
              - Save per-iteration histogram plot every iteration (same base as image).
              - Also save a FINAL histogram plot for the last image (basename + "_final").
          * If self.cfg.return_partial_images is False:
              - Do NOT save histogram text or plots during the loop.
              - After the loop, save ONLY the FINAL histogram text file and FINAL
                histogram plot for the last image (basename + "_final").
        """

        # 1) Iteration control
        max_autogain_iterations = (
            self.cfg.max_autogain_iterations if max_iterations is None else max_iterations
        )
        save_partials = bool(self.cfg.return_partial_images)

        t0 = time.monotonic()
        loop_counts = 1

        # 2) Clamp initial gain into allowed range
        min_gain = int(self.cfg.min_gain_setting)
        max_gain = int(self.cfg.max_gain_setting)
        new_gain_value = int(initial_gain_value)
        if new_gain_value < min_gain:
            self.logit(
                f"Initial gain {new_gain_value} < min_gain_setting={min_gain}. Clamping.",
                color='yellow'
            )
            new_gain_value = min_gain
        elif new_gain_value > max_gain:
            self.logit(
                f"Initial gain {new_gain_value} > max_gain_setting={max_gain}. Clamping.",
                color='yellow'
            )
            new_gain_value = max_gain

        if max_autogain_iterations > 1:
            # take image using these settings.
            self.camera.set_control_value(asi.ASI_GAIN, new_gain_value)
            self.logit(f'Initial (New) Gain Value: {new_gain_value}')
            if self.focuser.aperture_position == 'closed':
                self.logit('Opening Aperture.')
                self.focuser.open_aperture()

        # Camera setting required mode
        self.camera.set_image_type(asi.ASI_IMG_RAW16)
        # self.camera.set_roi(bins=self.cfg.roi_bins)

        self.logit("Starting autogain routine.")
        high_pix_value = 0

        # To remember the final iteration histogram for the final plot/text
        last_basename = None  # full base path from capture_timestamp_save
        last_bins = None
        last_counts = None
        last_high_pix_value = None
        last_low_pix_value = None
        last_control_value = None
        last_metric_source = None

        # Minimum gain step (ensures we don't stall if controller returns same value)
        min_step = int(self.cfg.autogain_min_gain_step)

        # Threshold for "heavily saturated" fraction of pixels at ADC limit
        sat_frac_thresh = 1e-4  # 0.01%

        # 3) Main loop
        while True:
            if loop_counts > max_autogain_iterations:
                self.logit("Maximum iterations reached. Can't find solution. Ending cycle.")
                break

            self.logit("#" * 104)
            self.logit(f"Autogain Iteration: {loop_counts} / {max_autogain_iterations}")

            self.logit('Capturing image.')
            inserted_string = 'e' + 'g' + str(new_gain_value)

            # Take image or use existing self.curr_img for single gain update
            img = self.curr_img if max_iterations is not None and max_iterations == 1 else None
            img, basename = self.capture_timestamp_save(auto_gain_image_path, inserted_string, img)
            self.log.debug('Image captured.')
            # `basename` is the base path for this image; we reuse it for histogram files.

            # Flatten once
            arr, et2 = self.utils.timed_function(img.flatten)

            # ----------------- Brightness metrics -----------------
            high_pix_value, et4 = self.utils.timed_function(np.max, arr)
            low_pix_value = int(arr.min())
            p999_all = float(np.percentile(arr, 99.9))  # for logging only

            num_at_sat = int((arr >= pixel_saturated_value).sum())
            sat_frac = num_at_sat / max(arr.size, 1)
            unclipped = arr[arr < pixel_saturated_value]

            # Control metric selection
            if high_pix_value < pixel_saturated_value or num_at_sat == 0:
                # Not saturated (or no pixels at ADC limit) → use max pixel directly
                control_value = float(high_pix_value)
                metric_source = "max"
                is_saturated = False
            else:
                # Saturated: use 99.9 percentile of *unclipped* pixels if possible
                if unclipped.size > 0:
                    control_value = float(np.percentile(unclipped, 99.9))
                    metric_source = "p99.9_unclipped"
                else:
                    # Pathological: everything is clipped; fall back to max
                    control_value = float(high_pix_value)
                    metric_source = "max_all_clipped"
                is_saturated = True

            # Log iteration stats including control metric and saturation info
            self.logit(
                f'Iteration stats: iter={loop_counts}, gain={new_gain_value} [cB], '
                f'max={high_pix_value} [ADU], '
                f'control={control_value:.1f} [ADU] ({metric_source}), '
                f'p99.9_all={p999_all:.1f}, '
                f'min={low_pix_value} [ADU], '
                f'sat_pixels={num_at_sat} ({sat_frac:.2e} of image)'
            )

            # ----------------- Histogram (for plotting / logging) -----------------
            bins, et1 = self.utils.timed_function(
                np.linspace, 0, pixel_saturated_value, self.cfg.autogain_num_bins
            )
            (counts, bins), et3 = self.utils.timed_function(np.histogram, arr, bins=bins)

            self.log.debug(
                f'Exec np.linspace/img.flatten/np.histogram/np.max in sec: '
                f'{et1}/{et2}/{et3}/{et4} arr dtype: {arr.dtype}'
            )
            self.log.debug(f'bin: {len(bins)} counts: {len(counts)}')

            # Save data needed to make final histogram after loop
            last_basename = basename
            last_bins = bins
            last_counts = counts
            last_high_pix_value = high_pix_value
            last_low_pix_value = low_pix_value
            last_control_value = control_value
            last_metric_source = metric_source

            # ----------------- Optional histogram figure + text for partials -----------------
            if save_partials:
                # Histogram figure (more "professional" style)
                t_plot0 = time.monotonic()
                fig, ax = plt.subplots()

                ax.hist(
                    bins[:-1],
                    bins,
                    weights=counts,
                    histtype="stepfilled",
                    linewidth=1.5,
                    alpha=0.8,
                )

                ax.set_xlabel("Pixel value [counts]")
                ax.set_ylabel("Frequency [pixels]")

                ax.set_title(
                    f"Autogain Histogram (iter={loop_counts}, gain={new_gain_value} cB)\n"
                    f"control={control_value:.1f} ({metric_source}), max={high_pix_value}, "
                    f"min={low_pix_value}, target={desired_max_pix_value}±{pixel_count_tolerance}"
                )

                # Mark the target and tolerance band
                ax.axvline(desired_max_pix_value, linestyle="--", linewidth=1.5)
                ax.axvline(
                    desired_max_pix_value - pixel_count_tolerance,
                    linestyle=":",
                    linewidth=1.0,
                )
                ax.axvline(
                    desired_max_pix_value + pixel_count_tolerance,
                    linestyle=":",
                    linewidth=1.0,
                )

                # Lightly tighten x-limits around data
                x_min = max(
                    0,
                    min(low_pix_value, desired_max_pix_value - 3 * pixel_count_tolerance),
                )
                x_max = min(
                    pixel_saturated_value * 1.05,
                    max(high_pix_value, desired_max_pix_value + 3 * pixel_count_tolerance),
                )
                ax.set_xlim(x_min, x_max)

                ax.tick_params(direction="in", top=True, right=True)

                fig.tight_layout()
                # Save histogram PNG next to the image (same basename, different extension)
                self.save_fig(basename, plt)
                self.log.debug(f"Histogram plot created in {get_dt(t_plot0)}.")

                # Histogram text file for this iteration (same directory as image)
                txt_filename = basename + "_histogram.txt"
                try:
                    with open(txt_filename, 'w') as f:
                        f.write("--- Histogram Data (Iteration) ---\n")
                        f.write(f"# Basename: {basename}\n")
                        f.write(f"# Iteration: {loop_counts}\n")
                        f.write(f"# Gain [cB]: {new_gain_value}\n")
                        f.write(f"# Desired max pixel value [counts]: {desired_max_pix_value}\n")
                        f.write(f"# Pixel tolerance [counts]: {pixel_count_tolerance}\n")
                        f.write(f"# ADC saturation value [counts]: {pixel_saturated_value}\n")
                        f.write(f"# High pixel value (max): {high_pix_value}\n")
                        f.write(f"# Control value [counts]: {control_value:.3f}\n")
                        f.write(f"# Control source: {metric_source}\n")
                        f.write(f"# 99.9th percentile (all pixels): {p999_all:.3f}\n")
                        f.write(f"# Saturated pixels: {num_at_sat} ({sat_frac:.3e} of image)\n")
                        f.write(f"# Low pixel value (min): {low_pix_value}\n")
                        f.write(f"# Number of bins: {len(bins)}\n")
                        f.write(f"# Total pixels: {np.sum(counts)}\n")
                        f.write("# Bin_Left_Edge,Bin_Right_Edge,Count\n")
                        for i in range(len(counts)):
                            f.write(f"{bins[i]:.1f},{bins[i + 1]:.1f},{counts[i]}\n")
                        f.write("\n")
                    self.log.debug(f'Histogram data saved to {txt_filename}')
                except Exception as e:
                    self.log.error(f'Failed to save Histogram data to {txt_filename}: {e}')

            # ----------------- Saturation diagnostics ONLY (still use control metric) -----------------
            heavily_saturated = is_saturated and (sat_frac > sat_frac_thresh)
            if heavily_saturated:
                self.logit(
                    "Image is heavily saturated (significant fraction at ADC limit); "
                    "using unclipped 99.9th percentile as control metric."
                )
            elif is_saturated:
                self.logit(
                    "Image has some saturated pixels; using unclipped 99.9th percentile as control metric."
                )
            else:
                self.logit("Image not saturated; using max pixel as control metric.")

            # ----------------- Control metric + convergence test -----------------
            difference = float(desired_max_pix_value) - float(control_value)

            self.logit(
                f'Counts: desired_max_pix_value:{desired_max_pix_value} '
                f'control_value: {control_value:.1f} ({metric_source})'
            )

            # Convergence: |control_value - target| <= tolerance
            if abs(difference) <= float(pixel_count_tolerance):
                self.logit("Pixel counts (control metric) are in range. Ending iterations.")
                self.logit(f"Pixel count difference: {difference:.1f}")
                break

            # We are NOT converged → gain must change this iteration
            if control_value > desired_max_pix_value + pixel_count_tolerance:
                self.logit("Counts too high (control metric above target).")
                counts_too_high = True
            else:
                self.logit("Counts too low (control metric below target).")
                counts_too_high = False
            self.logit(f"Pixel count difference: {difference:.1f}")

            old_gain_value = int(new_gain_value)

            # ----------------- Compute new gain -----------------
            self.logit(
                f"Image autogain step using control metric '{metric_source}' "
                f"(heavily_saturated={heavily_saturated})."
            )

            raw_new_gain = self.calculate_gain_adjustment(
                old_gain_value, control_value, desired_max_pix_value
            )

            # Enforce directionality + minimum step so we don't stall
            if control_value > desired_max_pix_value + pixel_count_tolerance:
                # too bright → MUST move gain DOWN
                if raw_new_gain >= old_gain_value:
                    raw_new_gain = old_gain_value - min_step
            elif control_value < desired_max_pix_value - pixel_count_tolerance:
                # too dim → MUST move gain UP
                if raw_new_gain <= old_gain_value:
                    raw_new_gain = old_gain_value + min_step

            # Round and clamp to valid range
            new_gain_value = int(round(raw_new_gain))

            # If the gain rails out at min/max, recommend exposure adjustment in the log.
            if new_gain_value < min_gain:
                self.logit(
                    f"Requested gain {new_gain_value} < min_gain_setting={min_gain}. "
                    f"Clamping.",
                    color='yellow'
                )
                new_gain_value = min_gain

                # Recommendation based on brightness state
                if counts_too_high:
                    self.logit(
                        "Gain has railed at MIN and image is still too bright. "
                        "Recommend DECREASING exposure time.",
                        color='red'
                    )
                else:
                    self.logit(
                        "Gain has railed at MIN but image is too dim. "
                        "Recommend INCREASING exposure time (check configuration).",
                        color='red'
                    )

            elif new_gain_value > max_gain:
                self.logit(
                    f"Requested gain {new_gain_value} > max_gain_setting={max_gain}. "
                    f"Clamping.",
                    color='yellow'
                )
                new_gain_value = max_gain

                if counts_too_high:
                    self.logit(
                        "Gain has railed at MAX and image is too bright. "
                        "Recommend DECREASING exposure time.",
                        color='red'
                    )
                else:
                    self.logit(
                        "Gain has railed at MAX and image is still too dim. "
                        "Recommend INCREASING exposure time.",
                        color='red'
                    )

            self.logit(f'Old/New Gain Value: {old_gain_value}/{new_gain_value}')
            self.camera.set_control_value(asi.ASI_GAIN, new_gain_value)

            loop_counts += 1

        # 4) Final histogram text & plot (ALWAYS saved for final image)
        if last_basename is not None and last_bins is not None and last_counts is not None:
            # Final histogram text: if partials are disabled, this is the ONLY histogram TXT
            if not save_partials:
                txt_filename = last_basename + "_final_histogram.txt"
                try:
                    with open(txt_filename, 'w') as f:
                        f.write("--- Histogram Data (FINAL) ---\n")
                        f.write(f"# Basename: {last_basename}\n")
                        f.write(f"# Final gain [cB]: {int(new_gain_value)}\n")
                        f.write(f"# Desired max pixel value [counts]: {desired_max_pix_value}\n")
                        f.write(f"# Pixel tolerance [counts]: {pixel_count_tolerance}\n")
                        f.write(f"# ADC saturation value [counts]: {pixel_saturated_value}\n")
                        f.write(f"# High pixel value (max): {last_high_pix_value}\n")
                        f.write(f"# Control value [counts]: {last_control_value:.3f}\n")
                        f.write(f"# Control source: {last_metric_source}\n")
                        f.write(f"# Low pixel value (min): {last_low_pix_value}\n")
                        f.write(f"# Number of bins: {len(last_bins)}\n")
                        f.write(f"# Total pixels: {np.sum(last_counts)}\n")
                        f.write("# Bin_Left_Edge,Bin_Right_Edge,Count\n")
                        for i in range(len(last_counts)):
                            f.write(f"{last_bins[i]:.1f},{last_bins[i + 1]:.1f},{last_counts[i]}\n")
                        f.write("\n")
                    self.log.debug(f'FINAL histogram data saved to {txt_filename}')
                except Exception as e:
                    self.log.error(f'Failed to save FINAL histogram data to {txt_filename}: {e}')

            # Final histogram plot: always saved with "_final" suffix, same base as image
            try:
                fig, ax = plt.subplots()

                ax.hist(
                    last_bins[:-1],
                    last_bins,
                    weights=last_counts,
                    histtype="stepfilled",
                    linewidth=1.5,
                    alpha=0.8,
                )

                ax.set_xlabel("Pixel value [counts]")
                ax.set_ylabel("Frequency [pixels]")

                ax.set_title(
                    f"FINAL Autogain Histogram (gain={int(new_gain_value)} cB)\n"
                    f"control={last_control_value:.1f} ({last_metric_source}), "
                    f"max={last_high_pix_value}, min={last_low_pix_value}, "
                    f"target={desired_max_pix_value}±{pixel_count_tolerance}"
                )

                # Target + tolerance band
                ax.axvline(desired_max_pix_value, linestyle="--", linewidth=1.5)
                ax.axvline(
                    desired_max_pix_value - pixel_count_tolerance,
                    linestyle=":",
                    linewidth=1.0,
                )
                ax.axvline(
                    desired_max_pix_value + pixel_count_tolerance,
                    linestyle=":",
                    linewidth=1.0,
                )

                # Clean x-limits based on final data
                x_min = max(
                    0,
                    min(last_low_pix_value, desired_max_pix_value - 3 * pixel_count_tolerance),
                )
                x_max = min(
                    pixel_saturated_value * 1.05,
                    max(
                        last_high_pix_value,
                        desired_max_pix_value + 3 * pixel_count_tolerance,
                    ),
                )
                ax.set_xlim(x_min, x_max)

                ax.tick_params(direction="in", top=True, right=True)

                fig.tight_layout()
                # Save final histogram PNG next to the last image
                self.save_fig(last_basename + "_final", plt)
                self.log.debug(f"Final histogram plot saved for {last_basename}")
            except Exception as e:
                self.log.error(f'Failed to save final histogram plot: {e}')

        # 5) Summary
        self.logit(
            "##########################Auto Gain Routine Summary Results: "
            "###############################",
            color='green'
        )
        self.logit(f'desired_max_pix_value: {desired_max_pix_value} [counts]')

        if last_control_value is not None:
            self.logit(
                f'final control_value ({last_metric_source}): '
                f'{last_control_value:.1f} [counts]'
            )
        if last_high_pix_value is not None:
            self.logit(f'final max_pixel_value: {last_high_pix_value} [counts]')

        self.logit(f'optimal_gain_value: {int(new_gain_value)} [cB]')
        self.logit("#" * 107, color='green')
        self.logit(f'do_auto_gain_routine completed in {get_dt(t0)}.', color='cyan')

        return int(new_gain_value)

    def software_gain_from_metric(
            self,
            bright_value: int | None,
            old_gain_value: int,
            min_gain_step: int | None = None,
    ) -> int:
        """
        Convert a brightness metric (e.g. sw_bright_value from source_finder)
        into a new gain value using calculate_gain_adjustment().

        bright_value:
            Brightness metric in ADU (e.g. 99.5th percentile of masked star pixels).
        old_gain_value:
            Current camera gain (cB).
        min_gain_step:
            Minimum |Δgain| required to actually change the gain.
            If None, uses cfg.software_autogain_min_step (if present) or 1.

        Returns
        -------
        int
            New gain value (or old_gain_value if no change / invalid input).
        """

        cfg = self.cfg

        # If metric is missing, bail out
        if bright_value is None:
            self.logit("software_gain_from_metric: bright_value is None; keeping current gain.")
            return int(old_gain_value)

        desired_max_pix_value = self.desired_max_pix_value
        pixel_count_tolerance = self.pixel_count_tolerance

        # Already good enough?
        diff = desired_max_pix_value - float(bright_value)
        self.logit(
            f"software_gain_from_metric: bright={bright_value}, "
            f"desired={desired_max_pix_value}, tol={pixel_count_tolerance}, diff={diff:.1f}"
        )
        if abs(diff) <= float(pixel_count_tolerance):
            self.logit("software_gain_from_metric: within tolerance; no gain change.")
            return int(old_gain_value)

        # Use existing analytic gain mapping
        try:
            new_gain = self.calculate_gain_adjustment(
                old_gain_value=old_gain_value,
                high_pix_value=float(bright_value),
                desired_max_pix_value=float(desired_max_pix_value),
            )
        except Exception as e:
            self.logit(
                f"software_gain_from_metric: calculate_gain_adjustment failed ({e}); "
                "keeping current gain."
            )
            return int(old_gain_value)

        # Clamp to allowed range
        new_gain = int(max(cfg.min_gain_setting, min(cfg.max_gain_setting, int(new_gain))))

        # Choose min step
        if min_gain_step is None:
            min_gain_step = self.cfg.software_autogain_min_step
        else:
            min_gain_step = int(min_gain_step)

        # Deadband in gain space
        step = new_gain - int(old_gain_value)
        if abs(step) < min_gain_step:
            self.logit(
                f"software_gain_from_metric: gain change {step} < {min_gain_step}; skipping."
            )
            return int(old_gain_value)

        # Actually set the camera gain here
        try:
            camera_settings = self.camera.get_control_values()
            self.camera.set_control_value(asi.ASI_GAIN, int(new_gain))
            current_gain = camera_settings['Gain']
            self.logit(f"Software adjust NEW camera gain (cB) : {current_gain}")
        except Exception as e:
            self.logit(
                f"software_gain_from_metric: failed to set camera gain ({e}); "
                "keeping old gain."
            )
            return int(old_gain_value)

        self.logit(f"software_gain_from_metric: gain {old_gain_value} -> {new_gain}")
        return int(new_gain)

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

            high_pix_value = max(high_pix_value, 1e-6)

            # Step 2: Calculate gain multiplier
            gain_multiplier = desired_max_pix_value / high_pix_value
            self.logit(f"gain_multiplier: {gain_multiplier}")

            # Step 3: Compute new gain values
            # TODO: Rename the vars with upper letters; Don't use upper case at all cost!
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

            if self.cfg.return_partial_images:
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

    def get_centroid_diameters(self, img, is_array=True,
                               log_file_path="log/test_log.txt",
                               return_partial_images=True):
        """
        Autofocus helper.

        Here `img` is expected to be a *binary sources mask* (True/False,
        0/1, or 0/255). We compute a contour-based diameter for each blob
        directly from the mask geometry (no background subtraction or
        source_finder inside this function).

        Diameter definition (area-equivalent):
            d_eq = 2 * sqrt(A / pi)
        where A is the contour area in pixels².
        """
        # --- Normalize to a uint8 mask (0 or 255) ---
        if is_array:
            mask = img.astype(np.uint8)
        else:
            # if a filename is ever passed, read as grayscale
            mask = self.utils.read_image_grayscale(img)

        # Ensure binary 0/255
        if mask.max() <= 1:
            mask = (mask > 0).astype(np.uint8) * 255

        # --- Find blobs in the mask ---
        contours, _ = cv2.findContours(
            mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        if not contours:
            self.logit("Focus Routine: no contours found in mask; returning [-1].")
            return [-1, ]

        min_size = self.cfg.img_min_size
        max_size = self.cfg.img_max_size

        diameters = []
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < min_size or area > max_size:
                continue

            # Diameter of contour: use area-equivalent diameter
            # d_eq = 2 * sqrt(A / pi)
            equiv_radius = np.sqrt(area / np.pi)
            d = 2.0 * float(equiv_radius)
            diameters.append(d)

        if not diameters:
            self.logit("Focus Routine: all contours rejected by size limits; returning [-1].")
            return [-1, ]

        diameters = np.array(diameters, dtype=float)
        self.logit("--------Focus Routine: Diameter List (mask-based, area-equivalent)--------")
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

            d = int(self.cfg.local_sigma_cell_size)
            if d < 3:
                raise ValueError("d must be >= 3")

            # --- First leveling pass ---
            downscale_factor = self.cfg.leveling_filter_downscale_factor
            if downscale_factor > 1:
                downsampled_img = cv2.resize(
                    img,
                    (img.shape[1] // downscale_factor, img.shape[0] // downscale_factor),
                    interpolation=cv2.INTER_AREA
                )
                d_small = max(3, d // downscale_factor)  # shrink kernel accordingly
            else:
                downsampled_img = img
                d_small = d

            initial_local_levels = self.sf.ring_mean_background_estimation(downsampled_img, d_small)

            # Upscale back with interpolation
            self.logit("Upscaling initial local levels.")
            if downscale_factor > 1:
                initial_local_levels = cv2.resize(
                    initial_local_levels,
                    (img.shape[1], img.shape[0]),
                    interpolation=cv2.INTER_LINEAR
                )

            # Subtract background the first time
            cleaned_img = self.sf.subtract_background(img, initial_local_levels)

            # --- Second leveling pass on cleaned image ---
            downscale_factor = self.cfg.leveling_filter_downscale_factor
            if downscale_factor > 1:
                downsampled_img = cv2.resize(
                    cleaned_img,
                    (cleaned_img.shape[1] // downscale_factor, cleaned_img.shape[0] // downscale_factor),
                    interpolation=cv2.INTER_AREA
                )
                d_small = max(3, d // downscale_factor)
            else:
                downsampled_img = cleaned_img
                d_small = d

            final_local_levels = self.sf.ring_mean_background_estimation(downsampled_img, d_small)

            # Upscale back with interpolation
            if downscale_factor > 1:
                final_local_levels = cv2.resize(
                    final_local_levels,
                    (img.shape[1], img.shape[0]),
                    interpolation=cv2.INTER_LINEAR
                )

            # Estimate noise for image
            sigma_g = self.sf.estimate_noise_pairs(cleaned_img, sep=int(self.cfg.noise_pair_sep_full))
            estimated_noise = np.full_like(cleaned_img, sigma_g, dtype=np.float32)

            # Build leveled residual
            residual_img = self.sf.subtract_background(cleaned_img, final_local_levels)

            # --- Threshold using a single noise-scaled threshold (no hysteresis) ---
            self.logit("Creating mask using single-threshold (k=5.0, no hysteresis) for autofocus.")
            k_single = 5.0  # single k-sigma threshold for autofocus

            sources_mask, sources_mask_u8 = self.sf.threshold_with_noise(
                residual_img,
                sigma_map=estimated_noise,
                k_high=k_single,  # same threshold for high/low
                k_low=k_single,
                use_hysteresis=False,  # disable hysteresis for autofocus
                min_area=int(self.cfg.hyst_min_area),
                close_kernel=int(self.cfg.hyst_close_kernel),
                sigma_gauss=float(self.cfg.hyst_sigma_gauss),
                sigma_floor=float(self.cfg.hyst_sigma_floor),
            )

            # Apply mask to cleaned image for diameter measurement
            masked_image = cv2.bitwise_and(cleaned_img, cleaned_img, mask=sources_mask_u8)

            # Mask-based equivalent diameters
            equiv_diameters = self.get_centroid_diameters(masked_image, is_array=True)

            if (
                    equiv_diameters is None or
                    (hasattr(equiv_diameters, "__len__") and len(equiv_diameters) == 1 and float(
                        equiv_diameters[0]) < 0)
            ):
                self.logit("No valid diameters at this focus position; assigning penalty diameter score.")
                diameter_score = 1e6
                self.logit('Mean star diameter: n/a (penalty).')
                self.logit('Median star diameter: n/a (penalty).')
                self.logit('Number of diameters: 0')
            else:
                mean_d = float(np.mean(equiv_diameters))
                median_d = float(np.median(equiv_diameters))

                # Use median for autofocus score (more robust)
                diameter_score = median_d

                self.logit(f'Mean star diameter: {mean_d:.3f} px')
                self.logit(f'Median star diameter (used for V-fit): {median_d:.3f} px')
                self.logit(f'Number of diameters: {len(equiv_diameters)}')

            # Store for diameter-based autofocus (sequence_diameter)
            diameter_scores.append(diameter_score)

            # Edge-contrast metric on cleaned background-subtracted image
            edge_score = self.focus_score_starfield_edge(cleaned_img, top_frac=1.0)
            self.logit(f'Edge-contrast Focus score: {edge_score:.6g}')

            # Store for contrast-based autofocus (sequence_contrast)
            focus_scores.append(edge_score)

            count += 1

        self.logit('Processing focus images.')
        try:
            trial_best_focus_pos, stdev = self.fit_best_focus(
                focus_image_path, focus_positions, focus_scores,
                diameter_scores, max_focus_position, focus_method
            )

            # Calculate focus deviation range
            focus_min_pos = self.cfg.lab_best_focus * (1 - self.cfg.autofocus_max_deviation)
            focus_max_pos = self.cfg.lab_best_focus * (1 + self.cfg.autofocus_max_deviation)

            # Check if the calculated result of autofocus is within allowed autofocus_max_deviation
            if trial_best_focus_pos < focus_min_pos or trial_best_focus_pos > focus_max_pos:
                self.logit(
                    f'AutoFocus position out of allowed range of autofocus_max_deviation: '
                    f'{trial_best_focus_pos} allowed range: {focus_min_pos}..{focus_max_pos}'
                )
                trial_best_focus_pos = self.cfg.lab_best_focus

            # Clamp to scan range with correct messages
            if trial_best_focus_pos < focus_start_pos:
                trial_best_focus_pos = focus_start_pos
                self.logit('Focus position too small. Setting to lower bound of focus range.')
            elif trial_best_focus_pos > focus_stop_pos:
                trial_best_focus_pos = focus_stop_pos
                self.logit('Focus position too great. Setting to upper bound of focus range.')

            # Only apply this stdev sanity check if we actually had a real fit
            if hasattr(self.cfg, "sigma_error_value") and stdev != self.cfg.sigma_error_value:
                if stdev > (focus_stop_pos + focus_start_pos) / 2:
                    self.logit(
                        f'Autofocus stdev={stdev:.3f} too large; '
                        f'falling back to trial_focus_pos={self.cfg.trial_focus_pos}.'
                    )
                    trial_best_focus_pos = self.cfg.trial_focus_pos

        except Exception as e:
            # Catch any unexpected failure from fit_best_focus / post-processing
            self.logit(f"best focus fit failed. Setting focus position to {self.cfg.trial_focus_pos}.")
            stdev = self.cfg.stdev_error_value
            trial_best_focus_pos = self.cfg.trial_focus_pos

            cprint(f'Error: {e}', 'red')
            logit(traceback.print_exception(e))
            self.log.error(traceback.format_exception(e))
            logit(e, color='red')

        # Move the focus to the best found position.
        self.logit(f"Moving Focus to best focus position: {int(round(trial_best_focus_pos))}")
        self.focuser.move_focus_position(trial_best_focus_pos)

        # Take final image at estimated best focus
        inserted_string = 'f' + str(int(round(trial_best_focus_pos)))
        self.capture_timestamp_save(focus_image_path, inserted_string)

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

            #define the target max pixel value and the tolerance of this target
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
                                                                self.max_focus_position, focus_method='sequence_contrast')

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
        # Keep cfg.exposure_time_s in sync (seconds)
        # TODO: Fix exposure_time_s, this is not a CONFIG value shall be defined at correct app level
        self.cfg.exposure_time_s = float(self.cfg.lab_best_exposure) / 1e6


        # Close aperture
        self.focuser.close_aperture()

    def run(self):
        """Run main PUEO execution entry and run loop."""
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
        # TODO: REMOVE after TEST!!!
        if False:
            duration = 60  # seconds
            for _ in tqdm(range(duration), desc=f"Test delay {duration}", unit="sec"):
                time.sleep(1)

        self.status = 'Ready'
        while True:
            try:
                self.main_loop()
            except KeyboardInterrupt:
                self.logit('Server: Exiting!', level='error', color='red')
                self.telemetry.close()
                self.server.close()
                return

    def autogain_maintenance(self, camera_settings):
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
        is_check_gain, et1 = self.utils.timed_function(self.check_gain_routine,
                                            self.curr_img, self.desired_max_pix_value, self.pixel_saturated_value,
                                            self.pixel_count_tolerance, auto_gain_image_path)
        if is_check_gain:
            self.server.write("Performing autogain maintenance.")
            curr_gain_value = camera_settings['Gain']
            self.best_gain_value, et2 = self.utils.timed_function(self.do_auto_gain_routine,
                                                        auto_gain_image_path, curr_gain_value,
                                                        self.desired_max_pix_value, self.pixel_saturated_value,
                                                        self.pixel_count_tolerance,
                                                        max_iterations=1)
            self.logit(f"Auto gain done. Old gain: {curr_gain_value} Best gain: {self.best_gain_value} Exec: {et1}, {et2} sec.")
            gain_value = self.best_gain_value
        else:
            self.logit(f"Auto gain done. Gain value is {gain_value}. No need to adjust. Exce time: {et1} sec.", color='green')

        return gain_value

    def _run_autogain_maintenance(self, camera_settings, current_gain):
        """Wrapper function to run autogain maintenance and log results."""
        new_gain, gain_exec = self.utils.timed_function(self.autogain_maintenance, camera_settings)
        self.logit(
            f'Single image autogain completed: interval: {self.cfg.autogain_update_interval} current gain: {current_gain} new gain: {new_gain} in {gain_exec:.4f} seconds.',
            color='cyan')

    def info_add_image_info(self, camera_settings, curr_utc_timestamp):
        """
        Append image metadata and statistics to info file with optimized performance.

        This function efficiently calculates image statistics and writes metadata to
        a log file, with special optimizations for large uint16 images (4000x3000+).

        Args:
            camera_settings (dict): Camera configuration parameters including
                exposure, gain, and temperature.
            curr_utc_timestamp (float): UTC timestamp of image capture.

        Performance optimizations:
        - Uses np.min/max on flattened array views instead of full arrays
        - Calculates mean without creating temporary arrays
        - Replaces np.unique + np.median with np.percentile for median estimation
        - Buffers file writes to reduce I/O overhead
        - Precomputes values outside of write loop
        """
        t0 = time.monotonic()
        # Precompute all image statistics in a single efficient pass
        img_flat = self.curr_img.reshape(-1)  # Create view, no copy

        # Calculate min/max - fastest method for uint16
        min_val = np.min(img_flat)
        max_val = np.max(img_flat)

        # Calculate mean - optimized for large arrays
        mean_val = np.mean(img_flat, dtype=np.float32)  # Use float32 for speed

        # Estimate median using percentile (much faster than unique+median)
        # For typical astronomical images, this is sufficiently accurate
        median_estimate = np.percentile(img_flat, 50)

        # Precompute temperature values
        detector_temp = camera_settings['Temperature'] / 10
        cpu_temp = self.telemetry.get_cpu_temp()

        # Prepare all file content in memory first to minimize I/O
        file_content = [
            "Image type : Single standard operating image\n",
            f"system timestamp : {curr_utc_timestamp}\n",
            f"exposure time (us) : {camera_settings['Exposure']}\n",
            f"gain (cB) : {camera_settings['Gain']}\n",
            f"focus position : {self.focuser.focus_position}\n",
            f"aperture position : {self.focuser.aperture_pos}, {self.focuser.aperture_f_val}\n",
            f"min/max pixel value (counts) : {min_val} / {max_val}\n",
            f"mean/median pixel value (counts) : {mean_val:.1f} / {median_estimate:.1f}\n",
            f"detector temperature : {detector_temp} °C\n",
            f"CPU Temperature: {cpu_temp} °C\n"
        ]

        # Add distortion parameters if available
        if hasattr(self, 'distortion_calibration_params') and self.distortion_calibration_params:
            file_content.append("estimated distortion parameters:\n")
            for key, value in self.distortion_calibration_params.items():
                file_content.append(f"{key}: {value}\n")
        else:
            file_content.append("Distortion calibration file not found.\n")

        # Write all content in a single I/O operation
        with open(self.info_file, "a", encoding='utf-8', buffering=8192) as file:
            file.write("\n=== IMAGE METADATA ===\n")

            # Compute start, mid, and end times (UTC)
            exp_us = int(camera_settings.get("Exposure", 0) or 0)
            t_start = datetime.datetime.fromtimestamp(curr_utc_timestamp, tz=timezone.utc)
            t_mid = t_start + datetime.timedelta(microseconds=exp_us // 2)
            t_end = t_start + datetime.timedelta(microseconds=exp_us)

            file.write(f"capture_start_utc : {t_start.isoformat()}\n")
            file.write(f"mid_exposure_utc : {t_mid.isoformat()}\n")
            file.write(f"capture_end_utc : {t_end.isoformat()}\n")
            file.write(f"exposure_time_us : {exp_us}\n")

            file.writelines(file_content)
            file.write("\n")

        self.log.debug(f'Saved image info in {get_dt(t0)}.')

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
            # === FILES & STORAGE ===
            file.write("\n=== FILES & STORAGE ===\n")
            file.write(f"raw file size: {sys.getsizeof(self.curr_img)} bytes\n")
            file.write(f"compressed file size: {os.path.getsize(self.cfg.ssd_path)} bytes\n")
            available_space = self.get_disk_usage()['free_space']
            available_space_gb = available_space / (1024 ** 3)
            file.write(f"available disk space: {available_space_gb:.2f} GB\n")
            file.write("\n")

            # === DETECTIONS (sorted by flux) ===
            file.write("=== DETECTIONS (sorted by flux) ===\n")
            file.write("y,x,flux,std,fwhm\n")  # CSV header for easy parsing
            with suppress(IndexError, TypeError):
                for i in range(len(self.curr_star_centroids)):
                    std = self.curr_star_centroids[i][3]
                    fwhm = self.curr_star_centroids[i][4]
                    file.write(
                        f"{self.curr_star_centroids[i][0]},{self.curr_star_centroids[i][1]},{self.curr_star_centroids[i][2]},{std},{fwhm}\n")
            file.write("\n")

            # === ASTROMETRY ===
            file.write("=== ASTROMETRY ===\n")
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
                gain = self.camera.get_control_values().get('Gain', 'unknown')
                self.astrometry['ExposureTime'] = f'{exposure} us' # microsecond
                self.astrometry['Gain'] = f'{gain} cB' # centibels

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

    def info_add_angular_velocity_info(self, omega_xyz_deg_s, delta_t_sec, eci_los_vec=None):
        """
        Append camera angular velocity (deg/s) and optional ECI LOS vector to the info file.

        Args:
            omega_xyz_deg_s: iterable/list/np.array of [omega_x, omega_y, omega_z] in deg/s (camera frame)
            delta_t_sec: float Δt between solutions in seconds
            eci_los_vec: optional iterable [eci_x, eci_y, eci_z] (unit vector)
        """
        ox, oy, oz = [float(v) for v in omega_xyz_deg_s]
        with open(self.info_file, "a", encoding='utf-8') as file:
            file.write("Camera rotation rate:\n")
            file.write(f"omega_x: {ox:.6f} deg/s\n")
            file.write(f"omega_y: {oy:.6f} deg/s\n")
            file.write(f"omega_z: {oz:.6f} deg/s\n")
            file.write(f"time difference between images\\solutions: {delta_t_sec:.6f} seconds\n")
            if hasattr(self, 'prev_image_name') and self.prev_image_name:
                file.write(f"name of previous image: {self.prev_image_name}\n")
            if eci_los_vec is not None:
                ex, ey, ez = [float(v) for v in eci_los_vec]
                file.write("ECI LOS unit vector:\n")
                file.write(f"eci_x: {ex:.9f}\n")
                file.write(f"eci_y: {ey:.9f}\n")
                file.write(f"eci_z: {ez:.9f}\n")

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

    def compute_angular_velocity(self):
        """Compute camera-frame angular velocity via SO(3) and log ECI LOS."""

        # Δt
        delta_t_sec = float(self.curr_time - self.prev_time)

        # Camera-frame angular rates [deg/s] from consecutive (RA,Dec,Roll)
        # This uses your SO(3)-based helper already in this file.
        omega_x, omega_y, omega_z = self.compute_camera_rates_from_astrometry()  # returns deg/s

        # Save in self.omega as [x,y,z] in deg/s (camera frame)
        self.omega = [float(omega_x), float(omega_y), float(omega_z)]

        # For logging: current orientation
        ra = self.astrometry.get('RA', 'N/A')
        dec = self.astrometry.get('Dec', 'N/A')
        roll = self.astrometry.get('Roll', 'N/A')

        # ECI LOS unit vector for this (RA,Dec,Roll)
        R_e_c = self._radec_roll_to_R_eci_from_cam(ra, dec, roll)  # columns are cam axes in ECI
        z_eci = R_e_c[:, 2]  # +Z_cam LOS in ECI

        # Write detailed info file line (camera rates + ECI LOS)
        self.info_add_angular_velocity_info(self.omega, delta_t_sec, eci_los_vec=z_eci)

        # Console/server logs
        self.log.debug(f'Camera angular velocity (deg/s): {self.omega}')
        self.server.write(f'{self.curr_time}, RA: {ra}, DEC: {dec}, ROLL: {roll} [deg]')
        self.server.write(f'{self.curr_time}, Ωx: {omega_x:.6f}, Ωy: {omega_y:.6f}, Ωz: {omega_z:.6f} [deg/s]')
        self.server.write(
            f'{self.curr_time}, ECI LOS: eci_x={z_eci[0]:.9f}, eci_y={z_eci[1]:.9f}, eci_z={z_eci[2]:.9f}')

        return {'omega_x': float(omega_x), 'omega_y': float(omega_y), 'omega_z': float(omega_z)}

    def _radec_roll_to_R_eci_from_cam(self, ra_deg, dec_deg, roll_deg):
        # Inputs are degrees
        α = np.deg2rad(float(ra_deg));
        δ = np.deg2rad(float(dec_deg));
        ρ = np.deg2rad(float(roll_deg))

        # +Z_cam as LOS in ECI
        z_eci = np.array([np.cos(δ) * np.cos(α), np.cos(δ) * np.sin(α), np.sin(δ)], dtype=float)

        # Orthonormal x/y basis ⟂ z_eci
        ref = np.array([0, 0, 1.0], dtype=float)
        if abs(z_eci @ ref) > 0.98:
            ref = np.array([1.0, 0, 0], dtype=float)
        x0 = np.cross(ref, z_eci);
        x0 /= (np.linalg.norm(x0) + 1e-12)
        y0 = np.cross(z_eci, x0);
        y0 /= (np.linalg.norm(y0) + 1e-12)

        # Apply roll about boresight
        x_eci = x0 * np.cos(ρ) + y0 * np.sin(ρ)
        y_eci = -x0 * np.sin(ρ) + y0 * np.cos(ρ)

        # Columns = camera axes expressed in ECI
        return np.column_stack([x_eci, y_eci, z_eci])

    def compute_camera_rates_from_astrometry(self):
        """
        Angular velocity in the CAMERA frame [deg/s] via SO(3) log map between
        consecutive (RA, Dec, Roll) solutions. Camera frame == body frame here.
        """
        # Pull consecutive solutions and Δt
        ra1 = self.astrometry.get('RA');
        dec1 = self.astrometry.get('Dec');
        roll1 = self.astrometry.get('Roll')
        ra0 = self.prev_astrometry.get('RA');
        dec0 = self.prev_astrometry.get('Dec');
        roll0 = self.prev_astrometry.get('Roll')
        dt = max(1e-6, float(self.curr_time - self.prev_time))

        # ECI<-Cam at t1 and t0
        R_e_c1 = self._radec_roll_to_R_eci_from_cam(ra1, dec1, roll1)
        R_e_c0 = self._radec_roll_to_R_eci_from_cam(ra0, dec0, roll0)

        # Since camera==body, rates are expressed in camera frame.
        # Relative rotation (t0->t1) expressed in camera frame at t0
        R_rel = R_e_c0.T @ R_e_c1

        # Log map: ω*dt = vee(log(R_rel))  — robust axis-angle
        tr = np.clip(np.trace(R_rel), -1.0, 3.0)
        angle = np.arccos(0.5 * (tr - 1.0))
        if angle < 1e-12:
            S = 0.5 * (R_rel - R_rel.T)
            wdt = np.array([S[2, 1], S[0, 2], S[1, 0]])
        else:
            S = 0.5 * (R_rel - R_rel.T)
            axis = np.array([S[2, 1], S[0, 2], S[1, 0]]) / (np.sin(angle) + 1e-12)
            wdt = angle * axis

        return np.degrees(wdt / dt)  # [ωx, ωy, ωz] in deg/s (camera frame)

    def adjust_exposure_gain(self, astrometry):
        """
        Single-step combined autogain (fine knob: GAIN) + autoexposure (coarse knob: EXPOSURE).

        Exposure units (internal): microseconds (µs).
        Exposure units (logging): milliseconds (ms).
        Gain units: camera gain setting (cB).

        All configuration comes from [CAMERA]:

            self.cfg.min_gain_setting
            self.cfg.max_gain_setting
            self.cfg.camera_exposure_min_us
            self.cfg.camera_exposure_max_us

            self.cfg.camera_exposure_lab_best_us   # used when exposure is "locked"

            self.cfg.autogain_enable
            self.cfg.autogain_desired_max_pixel_value
            self.cfg.autogain_mid_gain_setting
            self.cfg.autogain_min_gain_step
            self.cfg.autogain_max_exp_factor_up    # if both up & down are 1.0 → exposure lock
            self.cfg.autogain_max_exp_factor_down  # "
            self.cfg.autogain_use_masked_p999      # 0=never, 1=auto, 2=force
            self.cfg.autogain_min_mask_pixels
        """

        # --- 0. Master switch ---
        if self.cfg.autogain_mode == 'off':
            self.logit(f"auto gain/exposure: autogain_mode: {self.cfg.autogain_mode}, skipping.", color='yellow')
            return
        else:
            self.logit(f"auto gain/exposure: autogain_mode: {self.cfg.autogain_mode}, desired_max_pixel_value: {self.cfg.autogain_desired_max_pixel_value}.", color='green')

        # --- 1. Read p99.9 metrics from astrometry dict ---
        # NOTE: currently still coming from the legacy p999_* keys;
        # you will update astrometry wiring separately.
        p999_original = astrometry.get('p999_original')
        p999_masked = astrometry.get('p999_masked_original')
        n_mask_pixels = astrometry.get('n_mask_pixels')  # optional

        # --- Decide whether to use masked or unmasked p999 ---

        # Mapping autogain_use_masked_p999 string to int
        mode_map = {"never": 0, "auto": 1, "force": 2}

        # Read the config value (string)
        mode_str = str(self.cfg.autogain_use_masked_p999).lower().strip()

        mode = mode_map.get(mode_str, -1)

        min_mask_pixels = self.cfg.autogain_min_mask_pixels

        # Mapping mode (int) to full mode name (str):
        mode_str_map = {
            0: "never_masked",
            1: "auto_masked_if_enough_pixels",
            2: "force_masked_when_valid",
        }
        mode_str = mode_str_map.get(mode, f"unknown_mode_{mode}")
        if mode == -1:
            self.log.warning(f"Unknown mode: {mode_str}")

        use_masked = False

        if p999_masked is not None and p999_masked > 0:
            if mode == 2:
                # FORCE mode: always use masked p999 if it exists and is positive
                use_masked = True
            elif mode == 1:
                # AUTO mode: only use masked p999 if there are enough pixels in the mask
                if (n_mask_pixels is not None) and (n_mask_pixels >= min_mask_pixels):
                    use_masked = True
            # mode == 0 -> never use masked

        if use_masked:
            high_pix_value = p999_masked
            src_name = "p999_masked_original"
        else:
            high_pix_value = p999_original
            src_name = "p999_original"

        desired_max_pix_value = self.cfg.autogain_desired_max_pixel_value

        # --- Sanity check on brightness metrics and target ---
        # NOTE: high_pix_value == 0.0 is allowed (means "no signal yet", very dark).
        if (
                high_pix_value is None or
                desired_max_pix_value is None or
                desired_max_pix_value <= 0
        ):
            self.logit(f"autogain/autoexposure: status:HUNTING, action:invalid_p999", color='cyan')
            self.logit(f"  src: {src_name} mode: {mode}({mode_str}) use_masked: {use_masked} n_mask_pixels: {n_mask_pixels}",  color='yellow')
            self.logit(f"  p999: {high_pix_value}, target: {desired_max_pix_value}", color='yellow')
            self.logit(f"Note: no_change_applied", color='white')
            return

        # --- 2. Current camera settings ---
        camera_settings = self.camera.get_control_values()
        old_exposure = camera_settings['Exposure']  # µs
        old_gain_value = camera_settings['Gain']  # cB

        # --- 3. Limits and knobs from [CAMERA] ---

        # Gain hardware limits
        max_gain_hw = self.cfg.max_gain_setting
        min_gain_hw = self.cfg.min_gain_setting

        # Exposure limits (µs)
        exp_min = self.cfg.camera_exposure_min_us
        exp_max = self.cfg.camera_exposure_max_us

        # Autogain behavior parameters
        mid_gain = self.cfg.autogain_mid_gain_setting
        min_gain_step = self.cfg.autogain_min_gain_step
        max_exp_factor_up = self.cfg.autogain_max_exp_factor_up
        max_exp_factor_down = self.cfg.autogain_max_exp_factor_down

        # --- 3a. Exposure lock detection and lab_best exposure ---

        # max_exp_factor_up == max_exp_factor_down == 1.0 → lock exposure to lab_best
        # exposure_lock = (
        #         abs(max_exp_factor_up - 1.0) < 1e-6 and
        #         abs(max_exp_factor_down - 1.0) < 1e-6
        # )

        exposure_lock = self.cfg.autogain_mode == 'gain'

        lab_best_exposure = None
        if exposure_lock:
            lab_best_exposure = self.cfg.lab_best_exposure
            if lab_best_exposure > exp_max:
                lab_best_exposure = exp_max
            elif lab_best_exposure < exp_min:
                lab_best_exposure = exp_min
            lab_best_100us = lab_best_exposure / 100.0
            lab_best_exposure = int(round(lab_best_100us) * 100)

        # --- 4. GAIN as fine knob ---

        proposed_gain_value = self.calculate_gain_adjustment(
            old_gain_value,
            high_pix_value,
            desired_max_pix_value
        )

        too_dark = high_pix_value < desired_max_pix_value
        too_bright = high_pix_value > desired_max_pix_value

        # Initial brightness classification
        if too_bright:
            brightness_state = "TOO_BRIGHT"
        elif too_dark:
            brightness_state = "TOO_DARK"
        else:
            brightness_state = "NEAR_TARGET"

        # Gain "railed in wrong direction"
        railed_high_and_dark = (proposed_gain_value >= max_gain_hw) and too_dark
        railed_low_and_bright = (proposed_gain_value <= min_gain_hw) and too_bright
        railed_in_wrong_direction = railed_high_and_dark or railed_low_and_bright

        # If exposure is locked, never trigger coarse exposure branch
        if exposure_lock:
            railed_in_wrong_direction = False

        # --- Summary state we'll fill then log once ---

        action = "none"  # "fine_gain", "coarse_exposure", "no_change_small_delta", "lock_exp_only"
        ratio = None  # desired / measured brightness
        exp_factor = 1.0
        new_gain_value = old_gain_value
        new_exposure_value = old_exposure
        changed = False

        # --- Case A: gain can fix it (or exposure is locked) ---
        if not railed_in_wrong_direction:
            delta_gain = proposed_gain_value - old_gain_value

            if abs(delta_gain) < min_gain_step:
                action = "no_change_small_delta"

                if exposure_lock and (lab_best_exposure is not None) and (old_exposure != lab_best_exposure):
                    action = "lock_exp_only"
                    new_exposure_value = lab_best_exposure
                    new_gain_value = old_gain_value
                    changed = True
            else:
                action = "fine_gain"
                new_gain_value = proposed_gain_value

                if exposure_lock and (lab_best_exposure is not None):
                    new_exposure_value = lab_best_exposure
                else:
                    new_exposure_value = old_exposure

                changed = True

        else:
            # --- Case B: gain railed → adjust exposure and adjust gain (UNLOCKED ONLY) ---
            action = "coarse_exposure"

            # Use a floor so we don't divide by zero; keep actual high_pix_value for logs.
            high_for_ratio = max(high_pix_value, 1.0)
            ratio = desired_max_pix_value / float(high_for_ratio)  # >1 => too dark, <1 => too bright

            if ratio > 1.0:
                exp_factor = min(ratio, max_exp_factor_up)
            else:
                exp_factor = max(ratio, max_exp_factor_down)

            new_exposure_value = old_exposure * exp_factor

            # Clamp to exposure range
            if new_exposure_value > exp_max:
                new_exposure_value = exp_max
            elif new_exposure_value < exp_min:
                new_exposure_value = exp_min

            # Snap exposure to nearest 100 µs
            exposure_100us = new_exposure_value / 100.0
            new_exposure_value = int(round(exposure_100us) * 100)

            # Gain behavior at exposure limits:
            # - If too dark and already at max exposure → rail gain high and keep it there.
            # - If too bright and already at min exposure → rail gain low and keep it there.
            # - Otherwise → recenter to mid_gain to give room for fine adjustments.
            if too_dark and new_exposure_value >= exp_max:
                new_gain_value = max_gain_hw
            elif too_bright and new_exposure_value <= exp_min:
                new_gain_value = min_gain_hw
            else:
                new_gain_value = int(round(mid_gain))

            changed = True

        # --- Decide HUNTING vs CONVERGED ---
        status = "CONVERGED" if not changed else "HUNTING"
        if status == "CONVERGED":
            brightness_state = "NEAR_TARGET"

        # --- Helper to build ASCII bars + percentages (scaled down to 45 slots) ---

        def _make_bar_and_pct(val, vmin, vmax, slots=45):
            """
            Create a status bar like (--------|------------------------------)
            and a percentage (0–100, with 2 decimal places) showing where `val`
            lies between vmin and vmax.
            """
            try:
                if vmax <= vmin:
                    return "(---------------------------------------------)", 0.0
                frac = (val - vmin) / float(vmax - vmin)
            except Exception:
                return "(---------------------------------------------)", 0.0

            if frac < 0.0:
                frac = 0.0
            elif frac > 1.0:
                frac = 1.0

            idx = int(round(frac * (slots - 1)))
            left = "-" * idx
            right = "-" * (slots - 1 - idx)
            pct = frac * 100.0
            return f"({left}|{right})", pct

        # Build bars using the applied values (after clamping/rounding)
        exp_bar, exp_pct = _make_bar_and_pct(new_exposure_value, exp_min, exp_max)
        gain_bar, gain_pct = _make_bar_and_pct(new_gain_value, min_gain_hw, max_gain_hw)

        exp_pct_str = f"{exp_pct:.2f}%"
        gain_pct_str = f"{gain_pct:.2f}%"

        # For logging: convert exposures to milliseconds
        old_exp_ms = old_exposure / 1000.0
        new_exp_ms = new_exposure_value / 1000.0
        exp_min_ms = exp_min / 1000.0
        exp_max_ms = exp_max / 1000.0

        # --- Build verbose summary log (4 lines) ---

        ratio_str = f"{ratio:.3f}" if ratio is not None else "n/a"
        exp_factor_str = f"{exp_factor:.3f}" if action == "coarse_exposure" else "1.000"

        new_gain_value = int(round(new_gain_value))
        new_exposure_value = int(round(new_exposure_value))

        # Logging MUST be single line
        self.logit(
            f"autogain/autoexposure: status:{status}, brightness:{brightness_state}, "
            f"action:{action}, exp_lock:{exposure_lock}",
            color='yellow'
        )

        self.logit(
            f"src:{src_name}, mode:{mode}({mode_str}), use_masked:{use_masked}, "
            f"n_mask_pixels:{n_mask_pixels}, "
            f"p999:{high_pix_value:.1f}, target:{desired_max_pix_value:.1f}, "
            f"ratio:{ratio_str}, exp_factor:{exp_factor_str}",
            color='yellow'
        )

        self.logit(
            f"gain_old:{old_gain_value} cB, gain_new:{new_gain_value} cB, "
            f"exp_old:{old_exp_ms:.3f} ms, exp_new_ms:{new_exp_ms:.3f} ms, "
            f"epn_new:{new_exposure_value}, "
            f"gain_range:[{min_gain_hw},{max_gain_hw}] cB, "
            f"exp_range:[{exp_min_ms:.3f},{exp_max_ms:.3f}] ms, "
            f"changed:{changed}",
            color='yellow'
        )

        self.logit(
            f"bars: exp:[{exp_min_ms:.3f} ms]{exp_bar}[{exp_max_ms:.3f} ms], "
            f"pos:{new_exp_ms:.3f} ms,{exp_pct_str}, "
            f"gain:[{min_gain_hw} cB]{gain_bar}[{max_gain_hw} cB], "
            f"pos:{new_gain_value} cB,{gain_pct_str}",
            color='yellow'
        )

        # --- Apply settings if they changed ---
        if changed:
            self.camera.set_control_value(asi.ASI_GAIN, new_gain_value)
            self.cfg.set_dynamic(lab_best_gain=new_gain_value)
            if self.cfg.autogain_mode == 'both':
                self.camera.set_control_value(asi.ASI_EXPOSURE, new_exposure_value)
                self.cfg.set_dynamic(lab_best_exposure=new_exposure_value)

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

        # Check Filesystem Monitor and change to preflight if any of the monitored paths has reached critical level
        if self.monitor.enabled and self.monitor.is_critical and self.is_flight:
            self.flight_mode = self.flight_mode # Check will be done within the setter!!!
            self.cfg.set_dynamic(flight_mode=self.flight_mode)
            self.logit(f"Changing flight_mode to PREFLIGHT. Filesystem CRITICAL level reached.", color='red')  # Timestamp at END

        dt = datetime.datetime.now().isoformat()  # Get current ISO timestamp
        if is_raw:
            self.logit(f"Taking photo: raw: {is_raw} mode: {self.flight_mode} @{dt}", color='cyan')  # Timestamp at END
        else:
            self.logit("######################################## NEW CYCLE ########################################", color='cyan')
            self.logit(f"Taking photo: operation: {is_operation} solver: {self.solver} mode: {self.flight_mode} @{dt}", color='cyan')  # Timestamp at END

        self.curr_time = time.monotonic()
        # Default: will be updated just in time before capture!!!
        dtc, utc_time, curr_utc_timestamp = self.utils.get_current_utc_timestamp()

        if self.focuser.aperture_position == 'closed':
            self.focuser.open_aperture()

        # Set the ASI_IMG_RAW16
        # self.log.debug(f'Setting image type/roi @{get_dt(t0)}.')
        # self.camera.set_image_type(asi.ASI_IMG_RAW16)
        # self.camera.set_roi(bins=self.cfg.roi_bins)
        self.camera.ensure_image_type(asi.ASI_IMG_RAW16)
        self.camera.ensure_image_roi(bins=self.cfg.roi_bins)
        self.log.debug(f'Ensured image type/roi @{get_dt(t0)}.')

        # Capture/Take image
        if not is_test:
            t1 = time.monotonic()
            try:
                self.logit(f'Not testing. Taking image using real camera.')
                self.logit(f'Capturing image @{get_dt(t0)}.', color='green')
                # HW Autogain Request:
                #   In autonomous mode, every n-th image if hw_autogain_recalibration_interval > 0
                self.camera._hw_autogain_requested = (
                        self.operation_enabled and
                        self.cfg.hw_autogain_recalibration_interval != self.camera.HW_AUTOGAIN_DISABLED and
                        (self.img_cnt % self.cfg.hw_autogain_recalibration_interval == 0)
                )
                # Context manager for a hardware autogain cycle.
                with self.camera.hw_autogain_cycle():
                    dtc, utc_time, curr_utc_timestamp = self.utils.get_current_utc_timestamp()
                    self.curr_time = time.monotonic()
                    self.curr_img = self.camera.capture()
                self.curr_img_dtc = dtc
                self.img_cnt += 1
                self.logit(f'Camera image captured in {get_dt(t1)} @{get_dt(t0)} shape: {self.curr_img.shape} capture timestamp: {curr_utc_timestamp}.', color='green')
            except Exception as e:
                self.logit(f'Capture error: {e}', level='error', color='red')
                if not self.chamber_mode:
                    return

        # Set defaults:
        #  Flight computer ASTRO POSITION:
        position = {
            'timestamp':  utc_time.isoformat(),
            'solver': self.solver,
            'solver_name': self.astro.get_solver_name(self.solver),
            'astro_position': [None, None, None],
            'FOV': None,
            'RMSE': None,
            'RMS': [None, None, None],
            'sources': 0,
            'matched_stars': 0,
            'probability': float(0.0),
            'angular_velocity': [None, None, None]  # Initialize empty angular velocity dict
        }

        # curr_serial_utc_timestamp = self.serial_utc_update + time.monotonic() - self.serial_time_datum
        timestamp_string = dtc.strftime(self.timestamp_fmt)  # Used for filename, ':' should not be used.

        # Chamber mode/test mode?
        if self.chamber_mode or is_test:
            self.logit(f'Testing. Using image files.')
            self.curr_img = self.camera_dummy.capture()
            self.logit(f'Taking DUMMY capture from file: {self.camera_dummy.filename}', 'warning', color='blue')

        # Close aperture if not in autonomous mode (doing single take image)
        if not is_operation:
            self.focuser.close_aperture()

        # Get and log camera settings, also save it to utils for histogram rendering
        camera_settings = self.utils.camera_settings = self.camera.get_control_values()
        self.logit(f"camera exposure time (us) : {camera_settings['Exposure']}")
        current_gain = camera_settings['Gain']
        self.logit(f"camera gain (cB) : {current_gain}")
        self.logit(f'curr image length: {len(self.curr_img)} prev image length: {len(self.prev_img)}')
        self.logit(f"Max pixel: {np.max(self.curr_img)}, Min pixel: {np.min(self.curr_img)}. ")

        # Write image info to log
        self.info_file = f"{self.get_daily_path(self.cfg.final_path)}/log_{timestamp_string}.txt"
        self.info_add_image_info(camera_settings, curr_utc_timestamp)

        # Perform astrometry
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
                self.utils.save_raws,
                args=(self.curr_img,
                      self.get_daily_path(self.cfg.ssd_path), self.get_daily_path(self.cfg.sd_card_path),
                      image_file,
                      self.cfg.scale_factors, self.cfg.resize_mode,
                      self.cfg.raw_scale_factors, self.cfg.raw_resize_mode,
                      self.cfg.png_compression, self.is_flight,
                      self.cfg.inspection_settings,
                      self.cfg
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
            if self.cfg.enable_gui_data_exchange:
                self.server.write(f'Current image filename: {self.curr_image_name}')
                self.server.write(self.curr_scaled_name, data_type='image_file')

        if is_raw:
            # Only took the raw image, no solving
            self.logit(f'camera_take_image (RAW image/no solving) completed in {get_dt(t0)}.', color='green')
            return position

        # Get the return values
        self.astrometry, self.curr_star_centroids, self.contours_img = self.get_result(astrometry_result)

        # TODO: Assumed the do_astrometry was success. None (no solution) needs to be handled.
        if self.astrometry is None:
            self.log.warning(f'No solution for astrometry.')
            self.server.write(f'Solving of image (do_astrometry) did not produce any solutions.', 'warning')
            return position
        else:
            self.logit(f"p999_masked_image value: {self.astrometry.get('p999_masked_original', '-')}")
            self.logit(f"p999_original image value: {self.astrometry.get('p999_original', '-')}")

            # Enabled via self.cfg.autogain_enabled
            self.adjust_exposure_gain(self.astrometry)

        # Append additional image metadata log to a file.
        self.info_add_astro_info()

        # Calculate Angular velocity
        angular_velocity = {}
        self.omega = [float('nan')] * 3
        with suppress(KeyError, ValueError, TypeError):
            angular_velocity = self.compute_angular_velocity()

        # Display overlay image
        # foi ~ Final Overlay Image
        if is_operation or True:
            self.log.debug('Adding overlay.')
            self.foi_name, self.foi_info, self.foi_scaled_name, self.foi_scaled_info = self.utils.display_overlay_info(
                self.contours_img, timestamp_string,
                self.astrometry, self.omega, False,
                self.curr_image_name,
                self.get_daily_path(self.cfg.final_path),
                self.cfg.partial_results_path,
                self.cfg.foi_scale_factors,
                self.cfg.foi_resize_mode,
                self.cfg.png_compression,
                self.is_flight)


            # Create/update symlink to last foi file
            self.utils.create_symlink(self.cfg.web_path, self.foi_scaled_name, 'last_final_overlay_image_downscaled.png')

            if self.cfg.enable_gui_data_exchange and self.foi_scaled_name is not None:
                self.server.write(self.foi_scaled_name, data_type='image_file', dst_filename=self.curr_scaled_name)
            # else:
            #     cprint('Error no scaled image', color='red')
            #     self.log.error('foi_scaled_name was None, no downscaled image generated.')

            # Delete prev files (output folder), keeping only Last one.
            deleted_cnt = self.utils.delete_files(self.prev_foi_name, self.prev_foi_scaled_name, self.prev_info_file)
            if deleted_cnt:
                self.log.debug(f'Cleanup deleted {deleted_cnt} files (foi, info file).')

            self.prev_time = self.curr_time
            self.prev_img = self.curr_img
            self.prev_astrometry = self.astrometry
            self.prev_star_centroids = self.curr_star_centroids
            self.prev_image_name = self.curr_image_name

            self.prev_foi_name = self.foi_name
            self.prev_foi_scaled_name = self.foi_scaled_name
            self.prev_info_file = self.info_file

        # Send the file to clients
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
        if self.cfg.enable_gui_data_exchange:
            info_filename = f'{self.curr_scaled_name[:-4]}.txt'  # .png -> .txt
            self.server.write(self.info_file, data_type='info_file', dst_filename=info_filename)
        # Copy info file to sd_card and ssd_path
        if self.is_flight:
            shutil.copy2(self.info_file, self.get_daily_path(self.cfg.sd_card_path))
            shutil.copy2(self.info_file, self.get_daily_path(self.cfg.ssd_path))
        # Create/update symlink to last info file
        self.utils.create_symlink(self.cfg.web_path, self.info_file, 'last_info_file.txt')

        self.logit(f'camera_take_image completed in {get_dt(t0)}.', color='green')
        return position

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
        self.camera.disable_hw_autogain() # Disable autogain
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
        pos, f_val = self.focuser.move_aperture_absolute(aperture_position)
        self.cfg.set_dynamic(lab_best_aperture_position=aperture_position)
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
        # TODO: DO NOT DO THIS HERE THIS WAY!!! Also update cfg.exposure_time_s in seconds for streak ω estimation
        # TODO: exposure_time_s is defined in config but its not a config var!!! Fix thsi!
        try:
            self.cfg.exposure_time_s = float(manual_exposure_time) / 1e6
        except Exception:
            self.cfg.exposure_time_s = 0.0

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
        self.server.write(f'  CPU Temperature: {self.telemetry.get_cpu_temp()} °C')

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

    def camera_power_switch(self, cmd):
        """Switch the Power to the target device"""
        device = str(cmd.device)
        power = cmd.power.lower() == 'on' # True: On, False: Off
        switch_txt = 'On' if power else 'Off'

        invert = not self.cfg.sbc_dio_default
        value = power ^ invert  # ~ ^ XOR
        # Switch: True = ON, False = OFF
        # If sbc_dio_default is False, logic is inverted

        color = 'green' if power else 'red'
        self.logit(f'{device.title()} Power Switch: {switch_txt}', color=color)
        hi_low = "HIGH" if value else "LOW"
        self.logit(f'Setting GPIO Pins For {device.title()} to {value} ({hi_low}):', color='cyan')
        if device == 'camera':
            with suppress(RuntimeError):
                self.versa_api.set_pin(self.cfg.sbc_dio_camera_pin, value)
        elif device == 'focuser':
            with suppress(RuntimeError):
                self.versa_api.set_pin(self.cfg.sbc_dio_focuser_pin, value)
        else:
            raise ValueError('Invalid device, specified.')


    def camera_home_lens(self, cmd):
        c_min = self.min_focus_position
        c_max = self.max_focus_position
        self.min_focus_position, self.max_focus_position = self.focuser.home_lens_focus()
        self.logit(f'Home Lens completed: new min/max positions: {self.min_focus_position}/{self.max_focus_position} old: {c_min}/{c_max}')

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

        # Check if FileSystem is critical!!
        # Force switch to preflight
        if self.monitor.is_critical:
            value = 'preflight'

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
    init_common(log)
    log.debug('Main - star_comm_bridge')


def main():
    """Main, wrapped for server startup for standalone or profiled execution."""
    init()
    cfg1 = Config(f'conf/{config_file}', f'conf/{dynamic_file}')
    server = PueoStarCameraOperation(cfg1)
    server.run()


if __name__ == "__main__":
    main()

# Last line of this fine script, well almost
