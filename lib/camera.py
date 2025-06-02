# Standard imports
from contextlib import suppress
import datetime
import usb.core
import time
from time import sleep
from datetime import timezone
import sys
import os
import logging

# External imports
from zwoasi import Camera
import zwoasi as asi
from unittest.mock import MagicMock, patch
import cv2
import numpy as np

# Custom imports
from lib.utils import read_image_grayscale
from lib.common import logit

# '/usr/lib/x86_64-linux-gnu/libASICamera2.so'
lib_asi_file = '~/ASIStudio/libASICamera2.so'
lib_asi = os.path.abspath(os.path.expanduser(lib_asi_file))

with suppress(FileNotFoundError, OSError):
    asi.init(lib_asi)

class DummyCamera:
    """A mock camera interface for testing and development purposes.

    This class simulates the behavior of a real camera by returning fixed, hardcoded
    values and test images. It was created to enable testing and development of
    camera-related functionality without requiring access to physical hardware.

    Typical use cases include:
    - Running/debugging code on development workstations
    - Automated testing in CI pipelines
    - Prototyping camera-related features

    The test images are loaded from the directory specified by `images_dir`.

    Note: This is a testing utility - for production use, replace with a real camera
    implementation.
    Use at your own discretion.

    Attributes:
        images_dir (str): Path to the directory containing test images. Defaults to 'test_images'.
        image_idx (int): Index of the current image being served.
        image_cnt (int): Total count of available test images.
        filename (str): Path to the currently loaded image file.
    """
    images_dir = 'test_images'
    image_idx = 0
    image_cnt = 0
    filename = None

    def __init__(self, camera=None):
        """Initialize the dummy camera instance.

        Args:
            camera (object, optional): If provided, replaces the camera's methods with
                dummy implementations. Defaults to None.
        """
        self.log = logging.getLogger('pueo')
        self.log.warning(f'Initializing DUMMY Camera')
        self.camera = camera
        if camera:
            camera.capture = self.capture
            camera.get_camera_property = self.get_camera_property
            camera.set_camera_defaults = self.set_camera_defaults
            camera.get_controls = self.get_controls
            camera.get_control_values = self.get_control_values
            camera.set_image_type = self.set_image_type
            camera.set_roi = self.set_roi
            camera.set_control_value = self.set_control_value
        self.filename = None
        self.files = len(self.get_images())

    @property
    def file_size(self):
        """Get the size of the current image file in bytes.

        Returns:
            int: Size of the file in bytes, or -1 if the file doesn't exist.
        """
        if os.path.exists(self.filename):
            return os.path.getsize(self.filename)
        else:
            return -1

    def set_control_value(self, control_type, value, auto=False):
        """Dummy implementation of setting a camera control value.

        Args:
            control_type: Type of the control to set.
            value: Value to set for the control.
            auto (bool): Whether to use auto mode for this control. Defaults to False.
        """
        self.log.debug(f'Dummy set_control_value: control_type: {control_type} value: {value} auto: {auto}')

    def set_roi(self, bins: int):
        """Dummy implementation of setting the region of interest.

        Args:
            bins (int): Binning value to set.
        """
        self.log.debug(f'Dummy set_roi: {bins}')

    def set_image_type(self, image_type):
        """Dummy implementation of setting the image type.

        Args:
            image_type: Type of image to set.
        """
        self.log.debug(f'Dummy set_image_type: {image_type}')

    def get_control_values(self):
        """Get dummy control values for camera settings.

        Returns:
            dict: Dictionary of current control values with their settings.
        """
        controls = {'Gain': 120, 'Exposure': 30000, 'Offset': 0, 'BandWidth': 100, 'Flip': 0, 'AutoExpMaxGain': 285,
                    'AutoExpMaxExpMS': 30000, 'AutoExpTargetBrightness': 100, 'HighSpeedMode': 1, 'Temperature': 228,
                    'GPS': 0}
        return controls

    def get_controls(self):
        """Get dummy camera controls configuration.

        Returns:
            dict: Dictionary containing all available camera controls with their metadata,
                including min/max values, defaults, and capabilities.
        """
        controls = {'Gain': {'Name': 'Gain', 'Description': 'Gain', 'MaxValue': 570, 'MinValue': 0, 'DefaultValue': 200,
                             'IsAutoSupported': True, 'IsWritable': True, 'ControlType': 0},
                    'Exposure': {'Name': 'Exposure', 'Description': 'Exposure Time(us)', 'MaxValue': 2000000000,
                                 'MinValue': 32, 'DefaultValue': 10000, 'IsAutoSupported': True, 'IsWritable': True,
                                 'ControlType': 1},
                    'Offset': {'Name': 'Offset', 'Description': 'offset', 'MaxValue': 80, 'MinValue': 0,
                               'DefaultValue': 8, 'IsAutoSupported': False, 'IsWritable': True, 'ControlType': 5},
                    'BandWidth': {'Name': 'BandWidth', 'Description': 'The total data transfer rate percentage',
                                  'MaxValue': 100, 'MinValue': 40, 'DefaultValue': 50, 'IsAutoSupported': True,
                                  'IsWritable': True, 'ControlType': 6},
                    'Flip': {'Name': 'Flip', 'Description': 'Flip: 0->None 1->Horiz 2->Vert 3->Both', 'MaxValue': 3,
                             'MinValue': 0, 'DefaultValue': 0, 'IsAutoSupported': False, 'IsWritable': True,
                             'ControlType': 9},
                    'AutoExpMaxGain': {'Name': 'AutoExpMaxGain', 'Description': 'Auto exposure maximum gain value',
                                       'MaxValue': 570, 'MinValue': 0, 'DefaultValue': 285, 'IsAutoSupported': False,
                                       'IsWritable': True, 'ControlType': 10},
                    'AutoExpMaxExpMS': {'Name': 'AutoExpMaxExpMS',
                                        'Description': 'Auto exposure maximum exposure value(unit ms)',
                                        'MaxValue': 60000, 'MinValue': 1, 'DefaultValue': 100, 'IsAutoSupported': False,
                                        'IsWritable': True, 'ControlType': 11},
                    'AutoExpTargetBrightness': {'Name': 'AutoExpTargetBrightness',
                                                'Description': 'Auto exposure target brightness value', 'MaxValue': 160,
                                                'MinValue': 50, 'DefaultValue': 100, 'IsAutoSupported': False,
                                                'IsWritable': True, 'ControlType': 12},
                    'HighSpeedMode': {'Name': 'HighSpeedMode', 'Description': 'Is high speed mode:0->No 1->Yes',
                                      'MaxValue': 1, 'MinValue': 0, 'DefaultValue': 0, 'IsAutoSupported': False,
                                      'IsWritable': True, 'ControlType': 14},
                    'Temperature': {'Name': 'Temperature', 'Description': 'Sensor temperature(degrees Celsius)',
                                    'MaxValue': 1000, 'MinValue': -500, 'DefaultValue': 20, 'IsAutoSupported': False,
                                    'IsWritable': False, 'ControlType': 8},
                    'GPS': {'Name': 'GPS', 'Description': 'the camera has a GPS or not', 'MaxValue': 1, 'MinValue': 0,
                            'DefaultValue': 0, 'IsAutoSupported': False, 'IsWritable': False, 'ControlType': 22}}

        return controls

    def set_camera_defaults(self):
        """Dummy implementation of setting camera defaults (no operation)."""
        pass

    def get_camera_property(self):
        """Get dummy camera properties.

        Returns:
            dict: Empty dictionary as this is a dummy implementation.
        """
        return {}

    def capture(self, initial_sleep=0.01, poll=0.01, buffer_=None, filename=None):
        """Simulate capturing an image from the dummy camera.

        Args:
            initial_sleep (float): Initial sleep time (simulated). Defaults to 0.01.
            poll (float): Poll interval (simulated). Defaults to 0.01.
            buffer_: Unused buffer parameter (for API compatibility).
            filename (str, optional): Output filename (unused in dummy implementation).

        Returns:
            numpy.ndarray: Grayscale image in 16-bit format (uint16).
        """
        self.log.debug(f'Dummy capture: initial_sleep: {initial_sleep} poll: {poll} buffer_: {buffer_} filename: {filename}')
        self.filename = self.get_next_image()
        # cprint(f'  Dummy images served: {filename}', 'yellow')
        # self.log.warning(f'Dummy Capture from: {filename}')

        img = cv2.imread(self.filename, cv2.IMREAD_UNCHANGED)

        # Convert to grayscale if the image is not already single-channel
        if len(img.shape) == 3:  # Check if the image has multiple channels
            grayscale_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        else:
            grayscale_img = img  # Image is already single-channel

        # Ensure the image is 16-bit
        if grayscale_img.dtype != np.uint16:
            # Scale the image to 16-bit range (0-65535)
            grayscale_img = (grayscale_img / grayscale_img.max() * 65535).astype(np.uint16)

        logit(f'Dummy Image: id: {self.image_idx}/{self.image_cnt} filename: {self.filename}', color='green')
        logit(f"      Shape: {img.shape}, Type: {img.dtype} Python Type: {type(img)}", color='yellow')

        # Handle alpha channel if present
        if img.shape[-1] == 4:  # RGBA image
            overlay_image = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)

        # Make the array contiguous if necessary
        # img = np.ascontiguousarray(image)
        return grayscale_img

    def get_images(self):
        """Get list of available test images.

        Returns:
            list: Paths to all PNG images in the images directory.
        """
        images = []
        # Find .png images
        for filename in os.listdir(self.images_dir):
            filepath = os.path.join(self.images_dir, filename)
            if filename.endswith(".png"):
                images.append(filepath)

        return images

    def get_next_image(self):
        """Get the next image in the sequence, cycling back to the first after the last.

        Returns:
            str: Path to the next image file.
        """
        images = self.get_images()

        image = images[self.image_idx]
        # Get next image, or start from start (circling over images)
        self.image_cnt = len(images)
        self.image_idx += 1
        if self.image_idx >= self.image_cnt:
            self.image_idx = 0

        return image


class PueoStarCamera(Camera):
    """
    Rotation?
        Is it possible to put in a feature that rotates the image by 90°, 270, 180°, and zero?
        There is a camera setting to do this.
        I realize that this might make image solving faster because the images are all right side down right now.
    """
    def __init__(self, cfg):
        self.cfg = cfg
        self.log = logging.getLogger('pueo')
        self.log.info('Initialising')
        self.num_cameras = 0
        self.cameras_found = []
        self.camera_id = 0
        self.camera_info = {}
        self.simulated = True

        # TODO: Implement Camera Retry in case of error.

        self.initialize_camera()
        # Initialize the parent class
        self.id = self.camera_id

        try:
            if not self.cameras_found:
                self.log.warning("No cameras detected. Using mock behavior.")
                self.simulated = True
                # Load Dummy Camera
                self.dummy_camera = DummyCamera(self)

            else:
                self.simulated = False
                super().__init__(self.camera_id or self.cameras_found[0])
        except Exception as e:
            logit(f"Error initializing camera: {e}", color='red')
            self.simulated = True

        # self.camera = asi.Camera(self.camera_id)
        self.camera_info = self.get_camera_property()
        self.set_camera_defaults()
        self.print_camera_control_parameters()
        self.log.info('Camera Initialized')

    def __del__(self):
        if self.simulated:
            self.log.debug("Simulated camera close.")
        else:
            super().close()
            self.log.debug("Camera closed successfully.")

    def initialize_camera(self):
        self.num_cameras = 0
        with suppress(FileNotFoundError, Exception):
            self.log.info(f'ASI SDK Lib: {self.cfg.env_filename}')
            asi.init(self.cfg.env_filename)
            self.num_cameras = asi.get_num_cameras()
            self.cameras_found = asi.list_cameras()  # Models names of the connected cameras

        if self.num_cameras == 1:
            self.camera_id = 0
            self.log.debug('Found one camera: %s' % self.cameras_found[0])
        else:
            self.log.error('No cameras found. Exiting')
            print('Switching to DUMMY Camera, serving test images.')
            # print('No cameras found. Exiting ')
            # TODO: Should exit here
            # sys.exit(0)

    def set_camera_defaults(self):
        # Set some sensible defaults. They will need adjusting depending upon
        # the sensitivity, lens and lighting conditions used.
        logit(f'Setting camera defaults', color='green')
        self.set_control_value(asi.ASI_GAMMA, self.cfg.asi_gama)  # nominally 50 so leaving.
        logit(f'{"ASI_GAMMA":>23s}: {self.cfg.asi_gama}', color='yellow')
        self.set_control_value(asi.ASI_BRIGHTNESS, self.cfg.pixel_bias)
        logit(f'{"ASI_BRIGHTNESS":>23s}: {self.cfg.pixel_bias}', color='yellow')
        self.set_control_value(asi.ASI_FLIP, self.cfg.asi_flip)
        logit(f'{"ASI_FLIP:":>23s} {self.cfg.asi_flip}', color='yellow')
        self.set_control_value(asi.ASI_HIGH_SPEED_MODE, 1)  # enable high speed mode. Not caring about read noise.
        logit(f'{"ASI_HIGH_SPEED_MODE:":>23s} 1', color='yellow')
        self.disable_dark_subtract()  # were not subtracting darks.
        logit(f'{"disable_dark_subtract":>23s} Yes', color='yellow')
        # Use minimum USB bandwidth permitted
        asi_bandwidthoverload_max_value = self.get_controls()['BandWidth']['MaxValue']
        self.set_control_value(asi.ASI_BANDWIDTHOVERLOAD, asi_bandwidthoverload_max_value)
        logit(f'{"disable_dark_subtract:":>23s} {asi_bandwidthoverload_max_value}', color='yellow')
        # set initial gains & exposure value. These were best measured in the lab.
        self.set_control_value(asi.ASI_GAIN, self.cfg.min_gain_setting)
        logit(f'{"ASI_GAIN:":>23s} {self.cfg.min_gain_setting}', color='yellow')
        self.set_control_value(asi.ASI_EXPOSURE, self.cfg.exposure_time)  # units microseconds,
        logit(f'{"ASI_EXPOSURE:":>23s} {self.cfg.exposure_time}', color='yellow')
        self.set_image_type(asi.ASI_IMG_RAW8)
        logit(f'{"ASI_IMG_RAW8:":>23s} ASI_IMG_RAW8', color='yellow')
        # Set to binning by 4.
        self.set_roi(bins=self.cfg.roi_bins)

    def print_camera_control_parameters(self):
        # Get all the camera controls and print them to the screen. This will confirm that the camera is getting set to the right values.
        self.log.info('Camera controls:')
        controls = self.get_controls()
        for cn in sorted(controls.keys()):
            self.log.debug('    %s:' % cn)
            for k in sorted(controls[cn].keys()):
                self.log.debug('        %s: %s' % (k, repr(controls[cn][k])))

    def is_available(self):
        dev = usb.core.find(idVendor=self.cfg.camera_id_vendor, idProduct=self.cfg.camera_id_product)
        if dev is None:
            curr_utc_timestamp = None
            while dev is None:
                dt = datetime.datetime.now(timezone.utc)
                utc_time = dt.replace(tzinfo=timezone.utc)
                curr_utc_timestamp = utc_time.timestamp()
                self.log.info(str(curr_utc_timestamp) + ' Lost camera connection. Waiting for reconnection...')
                dev = usb.core.find(idVendor=self.cfg.camera_id_vendor, idProduct=self.cfg.camera_id_product)
                self.prev_time = time.monotonic()
                sleep(1)

            self.log.info(str(curr_utc_timestamp) + ' Camera came back.')
            # sleep(10)    #wait for camera to boot.
            # asi.init(env_filename)
            # num_cameras = asi.get_num_cameras()
            # cameras_found = asi.list_cameras()  # Models names of the connected cameras
            # if num_cameras == 1:
            #    print('Found one camera: %s' % cameras_found[0])
            # global camera
            # camera = asi.Camera(0)
            # set_camera_defaults()

    def power_cycle(self, power_cycle_wait: float):
        """
        Power cycle for the camera.

        This function pauses execution for a specified duration to simulate a power cycle,
        then reinitialize the camera.

        Args:
            power_cycle_wait (float): The number of seconds to wait before reinitializing the camera.
        """
        self.log.debug('Power Cycle Function')

        time.sleep(power_cycle_wait)

        self.log.debug('Reinitializing Camera')
        self.__init__(self.cfg)

# Example usage
if __name__ == "__main__":
    logit('Running PueoStarCamera STANDALONE.')
    try:
        from lib.config import Config
        cfg =  cfg1 = Config(f'conf/config.ini')

        camera = PueoStarCamera(cfg)  # Replace with your port and baud rate
        controls = camera.get_controls()
        for k, v in controls.items():
            print(f'{k:>25s}: {v}')
        pass
        # ... (your application logic) ...
    except Exception as e:
        logit(f"Error: {e}")
    finally:
        logit(f'Exiting.', color='green')
# Last line, well almost
