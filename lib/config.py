import os
import time
from configparser import ConfigParser
from lib.dynamic import Dynamic

class Config(Dynamic):
    """
    This class holds configuration values for your application.
    """

    _config: ConfigParser = None

    # Vars...

    config_file = 'conf/config.ini'  # Note this file can only be changed HERE!!! It is still config.ini, but just for ref.
    dynamic_file = 'conf/dynamic.ini'

    test = False
    debug = False
    log_path = 'logs'
    log_file_path = 'log/test_log.txt'
    telemetry_log = 'logs/telemetry.log'

    # Example of adding new var
    # new_var = 'Happy'

    # CONFIG - [LENS_FOCUS_CONSTANTS]
    # lens focus constants.
    # trial_focus_pos = 8355  # units of counts. Focuser maps 0 to 16383 for lens focus position.
    trial_focus_pos = 2000  # for very near field.

    ###### CLI For Autofocus #######
    autofocus_start_position = 5000
    autofocus_stop_position = 6000
    autofocus_step_count = 10
    autofocus_method = 'sequence_contrast'

    ###### COURSE focus finding parameters#######
    top_end_span_percentage = 0.60
    bottom_end_span_percentage = 0.80
    coarse_focus_step_count = 10

    ###### FINE focus finding parameters#######
    fine_focus_step_count = 10

    lens_focus_homed = False
    exposure_time = 30 * 1000
    pixel_bias = 0  # pixel value offset in counts. This is a pixel bias. Set this to 100 for default.

    # focus should already be close for this to work. Will use ground-based value.
    # env_filename = '/home/windell/PycharmProjects/pueo_star_tracker/ASI_linux_mac_SDK_V1.28/lib/x64/libASICamera2.so'
    env_filename = '~/ASIStudio/lib/libASICamera2.so'
    # lens_focus_homed = False # Already defined above
    # auto exposure and gain constants.
    autogain_update_interval = 5
    autogain_num_bins = 100
    autoexposure_num_bins = 100
    max_autogain_iterations = 10
    max_autoexposure_iterations = 10
    max_gain_setting = 570
    min_gain_setting = 120

    # microseconds
    # 200000 = 200ms
    #  50000 =  50ms
    max_exposure_setting = 200000
    min_exposure_setting = 50000

    autogain_desired_max_pixel_value = 100  # TODO: Add correct value Windell
    autoexposure_desired_max_pixel_value = 100 # TODO: Add correct value Windell

    lab_best_aperture_position = 0
    lab_best_focus = 8352
    lab_best_gain = 120
    # microseconds
    # 100000 = 109ms
    lab_best_exposure = 100000
    autofocus_max_deviation = 0.1
    focus_tolerance_percentage = 2.5 # Percentage

    # New parameterised variables
    # CONFIG - [PATHS]

    fit_points_init = 100  # Used in fit_best_focus

    # Distortion Calibration Params
    lab_fov = 10.79490900481796
    lab_distortion_coefficient_1 = -0.1
    lab_distortion_coefficient_2 = 0.1

    # GUI +- Buttons Steps
    delta_focus_steps = 10
    delta_aperture = 1
    delta_exposure_time = 5
    delta_gain = 5

    # [CAMERA]
    asi_gama = 50
    asi_exposure = 9000
    asi_gain = 100
    asi_flip = 0    # Flip: {'Name': 'Flip', 'Description': 'Flip: 0->None 1->Horiz 2->Vert 3->Both', 'MaxValue': 3, 'MinValue': 0, 'DefaultValue': 0, 'IsAutoSupported': False, 'IsWritable': True, 'ControlType': 9}
    roi_bins = 2
    pixel_saturated_value_raw8 = 255
    pixel_saturated_value_raw16 = 16383
    stdev_error_value = 9999

    camera_id_vendor = 0x03c3
    camera_id_product = 0x294a
    pixel_well_depth = 14

    sbc_dio_camera_pin = 4
    sbc_dio_focuser_pin = 5
    sbc_dio_default = False
    # Time between cycling
    sbc_dio_cycle_delay = 2000
    # Time to reinitialise after power cycle# Ranging from 2.0 - 5.1 step 0.1
    power_cycle_wait = 3000

    # [SOURCES]
    # img_bkg_threshold = 3.1
    img_bkg_threshold = 8.0
    img_number_sources = 40
    img_min_size = 20
    img_max_size = 600

    # source_finder.py
    local_sigma_cell_size = 36
    sigma_clipped_sigma = 3.0
    sigma_clip_sigma = 3.0
    leveling_filter_downscale_factor = 4

    src_box_size_x = 50
    src_box_size_y = 50
    src_filter_size_x = 3
    src_filter_size_y = 3

    src_kernal_size_x = 3
    src_kernal_size_y = 3
    src_sigma_x = 1
    src_dst = 1

    photutils_gaussian_kernal_fwhm = 3.0
    photutils_kernal_size = 5.0
    dilate_mask_iterations = 1
    dilation_radius = 5
    min_potential_source_distance = 100
    level_filter = 9
    level_filter_type = 'median'
    ring_filter_type = 'mean'

    # [STARTRACKER]
    star_tracker_body_rates_max_distance = 100
    focal_ratio = 22692.68
    x_pixel_count = 2072
    y_pixel_count = 1411

    # [ASTROMETRY]
    ast_t3_database = 'data/default_database.npz'
    ast_is_array = True
    ast_is_trail = False
    ast_use_photoutils = False
    ast_subtract_global_bkg = False
    ast_fast = False
    # ast_bkg_threshold = 3.1
    ast_bkg_threshold = 8.0
    ast_number_sources = 20
    ast_min_size = 4
    ast_max_size = 200

    # For astrometry propagation
    # min_pattern_checking_stars = 15
    min_pattern_checking_stars = 10
    include_angular_velocity = True
    angular_velocity_timeout: float = 10.0  # seconds

    solve_timeout = 5000.0

    # [CEDAR]
    cedar_detect_host = 'localhost:50051'
    sigma = '2.5'
    max_size = 50
    binning = False
    return_binned = False
    use_binned_for_star_candidates = False
    detect_hot_pixels = False

    # Custom Pixel Count and Spatial Filtering
    pixel_count_min = 9
    pixel_count_max = 45
    spatial_distance_px = 75
    gaussian_fwhm = 2.0
    cedar_downscale = 3.0

    # [ASTROMETRY.NET]
    an_scale_units = 'degwidth'  # str
    an_scale_low = 0.1  # float
    an_scale_high = 1.0  # float
    an_downsample = 2  # int
    an_overwrite = True  # bool
    an_no_plots = True  # bool
    an_cpulimit = 30  # int
    an_depth = 20  # int
    an_sigma = 6.0  # float
    an_nsigma = 8.0  # float

    # MUST be False
    an_corr = False # bool
    an_new_fits = False # bool
    an_match = False # bool
    an_solved = False # bool

    # [IMAGES]
    png_compression = 0
    save_raw = True
    min_count = 10
    sigma_error_value = 9999
    return_partial_images = True

    resize_mode = 'downscale'
    scale_factor_x = 16
    scale_factor_y = 16
    scale_factors = (scale_factor_x, scale_factor_y)

    raw_resize_mode = 'downscale'
    raw_scale_factor_x = 8
    raw_scale_factor_y = 8
    raw_scale_factors = (raw_scale_factor_x, raw_scale_factor_y)

    inspection_images_keep = 100
    inspection_quality = 80
    inspection_lower_percentile = 1
    inspection_upper_percentile = 99
    inspection_last_image_symlink_name = 'last_inspection_image.jpg'

    # [GENERAL]
    flight_mode = 'flight'
    solver = 'solver1'
    time_interval = 1000000  # Microseconds
    max_processes = 4
    operation_timeout = 60
    current_timeout: float = 200.0 # Angular velocity timeout
    run_autofocus = True
    enable_autogain_with_autofocus = True
    run_autonomous = False
    run_telemetry = True
    run_chamber = False
    run_test = False

    # [PATHS]
    auto_gain_image_path_tmpl = r'/home/windell/PycharmProjects/version_5/{timestamp_string}_auto_gain_exposure_images/'
    focus_image_path_tmpl = r'/home/windell/PycharmProjects/version_5/{timestamp_string}_coarse_focus_images/'
    ultrafine_focus_image_path_tmpl = r'/home/windell/PycharmProjects/version_5/{timestamp_string}_ultrafine_focus_images/'

    partial_results_path = './partial_results'
    partial_image_name = '0 - Input Image.png'
    astro_path = 'logs/astro.json'

    ssd_path = 'ssd_path/'
    sd_card_path = 'sd_card_path/'

    final_path = 'output/'

    calibration_params_file = 'calibration_params_best.txt'
    gui_images_path = 'images'

    inspection_path = 'inspection_images/'

    web_path = 'web/'

    inspection_settings = {}

    # [DEVICES]
    focuser_port = '/dev/ttyUSB0'
    computer_port = '/dev/ttyUSB1'
    telemetry_port = '/dev/ttyUSB2'
    telemetry_baud_rate = 115200
    baud_rate = 115200

    # [STAR_COMM_BRIDGE]
    pueo_server_ip = '0.0.0.0'  # IP address of the server for communication
    server_ip = '127.0.0.1'     # IP address of the server for GUI/Flight Computer communication
    port = 5555                 # Port number for communication
    socket_timeout = 5          # Socket timeout
    max_retry = 5               # Maximum number of retries for connection attempts
    retry_delay = 2             # Delay in seconds between retries
    fq_max_size = 12            # Max number of solutions available/to keep for flight computer API exchange
    msg_max_size = 128          # Max number of messages in message queue

    # [GUI]
    enable_gui_data_exchange = False  # Controls whether the GUI can receive images through the message queue.
    images_keep = 5  # Number of images to keep
    log_reverse = False  # # Set ti Tru to have the Server Log shown in reverse


    # Member functions
    def __init__(self, config_file="conf/config.ini", dynamic_file="conf/dynamic.ini"):
        """
        Reads configuration values from a specified file.

        Args:
            config_file (str, optional): Path to the configuration file. Defaults to "config.ini".
        """
        # Initialize dynamic configuration first
        super().__init__(dynamic_file)

        # Then set Config-specific attributes
        self.config_file = config_file
        self.dynamic_file = dynamic_file

        # Then load the main configuration
        self.reload()

        # Now mark as initialized - future attribute changes will save to dynamic.ini
        self._is_initialized = True

    def get(self, section, key):
        """
        Retrieves a configuration value from a specific section and key.

        Args:
            section (str): The name of the section in the configuration file.
            key (str): The key for the desired configuration value.

        Returns:
            str: The value associated with the key, or None if not found.
        """
        if self._config.has_section(section):
            return self._config.get(section, key)
        else:
            return None

    def read(self, config_file='conf/config.ini', max_retries=5, delay=1):
        """
        Attempts to read a configuration file with retry logic in case of errors.

        Args:
            config_file (str): The path to the configuration file.
            max_retries (int, optional): The maximum number of retries if reading fails. Defaults to 5.
            delay (int, optional): The delay (in seconds) between each retry attempt. Defaults to 1 second.

        Returns:
            ConfigParser: A ConfigParser object containing the parsed configuration data.

        Raises:
            FileNotFoundError: If the specified config file is not found.
            PermissionError: If the script lacks permission to access the config file.
            (Other potential exceptions depending on specific scenarios)

        This function attempts to read the configuration file with the provided path. If an error occurs during reading, it will retry up to `max_retry` times with a delay of `delay` seconds between attempts. If the maximum number of retries is exhausted, the last encountered error will be raised.
        """
        self.config_file = config_file
        print(f'Reading config file: {self.config_file}')
        config = ConfigParser()

        attempt = 0
        while attempt < max_retries:
            try:
                # Attempt to read the config file
                read_list = config.read(config_file)
                break  # Success, exit the loop
            except Exception as e:
                attempt += 1
                print(f'Attempt {attempt} failed: {e}')
                if attempt < max_retries:
                    time.sleep(delay)  # Wait before retrying
                else:
                    raise  # If max retries reached, raise the error
        return config

    def reload(self):
        """Read config and then load and update all vars"""
        self._config = self.read(self.config_file)
        self.load()  # This loads static config values into attributes

        # Now update from dynamic (will create dynamic.ini if first time)
        super().update_from_dynamic()

    def load(self):
        """
        Attempts to extract vars from config.
        """

        # Loading section: [LENS_FOCUS_CONSTANTS]
        # config = self._config.getint('GLOBAL', 'config', fallback=self.config)

        self.test = self._config.getboolean('GLOBAL', 'test', fallback=self.test)
        self.debug = self._config.getboolean('GLOBAL', 'debug', fallback=self.debug)
        self.log_file_path = self._config.get('GLOBAL', 'log_file_path', fallback=self.log_file_path)
        self.log_path = self._config.get('LOG', 'path', fallback=self.log_path)
        self.telemetry_log = self._config.get('LOG', 'telemetry_log', fallback=self.telemetry_log)
        # Example of adding new var
        # self.new_var = self._config.get('GLOBAL', 'new_var', fallback=self.new_var)

        self.trial_focus_pos = self._config.getint('LENS_FOCUS_CONSTANTS', 'trial_focus_pos',
                                                  fallback=self.trial_focus_pos)

        self.autofocus_start_position = self._config.getint('LENS_FOCUS_CONSTANTS', 'autofocus_start_position', fallback=self.autofocus_start_position)
        self.autofocus_stop_position = self._config.getint('LENS_FOCUS_CONSTANTS', 'autofocus_stop_position', fallback=self.autofocus_stop_position)
        self.autofocus_step_count = self._config.getint('LENS_FOCUS_CONSTANTS', 'autofocus_step_count', fallback=self.autofocus_step_count)
        self.autofocus_method = self._config.get('LENS_FOCUS_CONSTANTS', 'autofocus_method', fallback=self.autofocus_method)

        self.top_end_span_percentage = self._config.getfloat('LENS_FOCUS_CONSTANTS', 'top_end_span_percentage',
                                                            fallback=self.top_end_span_percentage)
        self.bottom_end_span_percentage = self._config.getfloat('LENS_FOCUS_CONSTANTS', 'bottom_end_span_percentage',
                                                               fallback=self.bottom_end_span_percentage)
        self.coarse_focus_step_count = self._config.getint('LENS_FOCUS_CONSTANTS', 'coarse_focus_step_count',
                                                          fallback=self.coarse_focus_step_count)

        self.fine_focus_step_count = self._config.getint('LENS_FOCUS_CONSTANTS', 'fine_focus_step_count',
                                                        fallback=self.fine_focus_step_count)
        self.lens_focus_homed = self._config.getboolean('LENS_FOCUS_CONSTANTS', 'lens_focus_homed',
                                                       fallback=self.lens_focus_homed)
        self.exposure_time = self._config.getint('LENS_FOCUS_CONSTANTS', 'exposure_time', fallback=self.exposure_time)
        self.pixel_bias = self._config.getint('LENS_FOCUS_CONSTANTS', 'pixel_bias', fallback=self.pixel_bias)

        env_filename = self._config.get('LENS_FOCUS_CONSTANTS', 'env_filename', fallback=self.env_filename)
        self.env_filename = os.path.abspath(os.path.expanduser(env_filename))

        self.autogain_update_interval = self._config.getint('LENS_FOCUS_CONSTANTS', 'autogain_update_interval',
                                                           fallback=self.autogain_update_interval)
        self.autogain_num_bins = self._config.getint('LENS_FOCUS_CONSTANTS', 'autogain_num_bins',
                                                    fallback=self.autogain_num_bins)
        self.autoexposure_num_bins = self._config.getint('LENS_FOCUS_CONSTANTS', 'autoexposure_num_bins',
                                                    fallback=self.autoexposure_num_bins)
        self.max_autogain_iterations = self._config.getint('LENS_FOCUS_CONSTANTS',
                                                          'max_autogain_iterations',
                                                          fallback=self.max_autogain_iterations)
        self.max_autoexposure_iterations = self._config.getint('LENS_FOCUS_CONSTANTS',
                                                              'max_autoexposure_iterations',
                                                              fallback=self.max_autoexposure_iterations)

        self.max_gain_setting = self._config.getint('LENS_FOCUS_CONSTANTS', 'max_gain_setting',
                                                   fallback=self.max_gain_setting)
        self.min_gain_setting = self._config.getint('LENS_FOCUS_CONSTANTS', 'min_gain_setting',
                                                   fallback=self.min_gain_setting)

        self.max_exposure_setting = self._config.getint('LENS_FOCUS_CONSTANTS', 'max_exposure_setting',
                                                   fallback=self.max_exposure_setting)
        self.min_exposure_setting = self._config.getint('LENS_FOCUS_CONSTANTS', 'min_exposure_setting',
                                                   fallback=self.min_exposure_setting)

        self.autogain_desired_max_pixel_value = self._config.getint('LENS_FOCUS_CONSTANTS', 'autogain_desired_max_pixel_value',
                                                                   fallback=self.autogain_desired_max_pixel_value)
        self.autoexposure_desired_max_pixel_value = self._config.getint('LENS_FOCUS_CONSTANTS', 'autoexposure_desired_max_pixel_value',
                                                                       fallback=self.autoexposure_desired_max_pixel_value)
        self.lab_best_aperture_position = self._config.getint('LENS_FOCUS_CONSTANTS', 'lab_best_aperture_position',
                                                     fallback=self.lab_best_aperture_position)
        self.lab_best_focus = self._config.getint('LENS_FOCUS_CONSTANTS', 'lab_best_focus', fallback=self.lab_best_focus)
        self.lab_best_gain = self._config.getint('LENS_FOCUS_CONSTANTS', 'lab_best_gain', fallback=self.lab_best_gain)
        self.lab_best_exposure = self._config.getint('LENS_FOCUS_CONSTANTS', 'lab_best_exposure',
                                                    fallback=self.lab_best_exposure)

        self.autofocus_max_deviation = self._config.getfloat('LENS_FOCUS_CONSTANTS', 'autofocus_max_deviation',
                                                            fallback=self.autofocus_max_deviation)

        focus_tolerance_percentage = self._config.getfloat('LENS_FOCUS_CONSTANTS', 'focus_tolerance_percentage',
                                                            fallback=self.focus_tolerance_percentage)

        self.fit_points_init = self._config.getint('LENS_FOCUS_CONSTANTS', 'fit_points_init',
                                                  fallback=self.fit_points_init)
        self.lab_fov = self._config.getfloat('LENS_FOCUS_CONSTANTS', 'lab_fov', fallback=self.lab_fov)
        self.lab_distortion_coefficient_1 = self._config.getfloat('LENS_FOCUS_CONSTANTS', 'lab_distortion_coefficient_1',
                                                                 fallback=self.lab_distortion_coefficient_1)
        self.lab_distortion_coefficient_2 = self._config.getfloat('LENS_FOCUS_CONSTANTS', 'lab_distortion_coefficient_2',
                                                                 fallback=self.lab_distortion_coefficient_2)

        # Buttons GUI increments
        self.delta_focus_steps = self._config.getint('LENS_FOCUS_CONSTANTS', 'delta_focus_steps',
                                                    fallback=self.delta_focus_steps)
        self.delta_aperture = self._config.getint('LENS_FOCUS_CONSTANTS', 'delta_aperture', fallback=self.delta_aperture)
        self.delta_exposure_time = self._config.getint('LENS_FOCUS_CONSTANTS', 'delta_exposure_time',
                                                      fallback=self.delta_exposure_time)
        self.delta_gain = self._config.getint('LENS_FOCUS_CONSTANTS', 'delta_gain', fallback=self.delta_gain)

        # [CAMERA] section
        self.asi_gama = self._config.getint('CAMERA', 'asi_gama', fallback=self.asi_gama)
        self.asi_exposure = self._config.getint('CAMERA', 'asi_exposure', fallback=self.asi_exposure)
        self.asi_gain = self._config.getint('CAMERA', 'asi_gain', fallback=self.asi_gain)
        self.asi_flip = self._config.getint('CAMERA', 'asi_flip', fallback=self.asi_flip)
        self.roi_bins = self._config.getint('CAMERA', 'roi_bins', fallback=self.roi_bins)
        self.pixel_saturated_value_raw8 = self._config.getint('CAMERA', 'pixel_saturated_value_raw8',
                                                             fallback=self.pixel_saturated_value_raw8)
        self.pixel_saturated_value_raw16 = self._config.getint('CAMERA', 'pixel_saturated_value_raw16',
                                                              fallback=self.pixel_saturated_value_raw16)
        self.stdev_error_value = self._config.getint('CAMERA', 'stdev_error_value', fallback=self.stdev_error_value)

        camera_id_vendor_hex = self._config.get('CAMERA', 'camera_id_vendor', fallback=hex(self.camera_id_vendor))
        self.camera_id_vendor = int(camera_id_vendor_hex, 16)
        camera_id_product_hex = self._config.get('CAMERA', 'camera_id_product', fallback=hex(self.camera_id_product))
        self.camera_id_product = int(camera_id_product_hex, 16)
        self.pixel_well_depth = self._config.getint('CAMERA', 'pixel_well_depth', fallback=self.pixel_well_depth)

        self.sbc_dio_camera_pin = self._config.getint('CAMERA', 'sbc_dio_camera_pin', fallback=self.sbc_dio_camera_pin)
        self.sbc_dio_focuser_pin = self._config.getint('CAMERA', 'sbc_dio_focuser_pin',
                                                      fallback=self.sbc_dio_focuser_pin)
        self.sbc_dio_default = self._config.getboolean('CAMERA', 'sbc_dio_default', fallback=self.sbc_dio_default)
        self.sbc_dio_cycle_delay = self._config.getint('CAMERA', 'sbc_dio_cycle_delay',
                                                      fallback=self.sbc_dio_cycle_delay)

        self.power_cycle_wait = self._config.getint('CAMERA', 'power_cycle_wait', fallback=self.power_cycle_wait)

        # [SOURCES]
        self.img_bkg_threshold = self._config.getfloat('SOURCES', 'img_bkg_threshold', fallback=self.img_bkg_threshold)
        self.img_number_sources = self._config.getint('SOURCES', 'img_number_sources', fallback=self.img_number_sources)
        self.img_min_size = self._config.getint('SOURCES', 'img_min_size', fallback=self.img_min_size)
        self.img_max_size = self._config.getint('SOURCES', 'img_max_size', fallback=self.img_max_size)

        # source_finder.py
        self.local_sigma_cell_size = self._config.getint('SOURCES', 'local_sigma_cell_size',
                                                        fallback=self.local_sigma_cell_size)
        self.sigma_clipped_sigma = self._config.getfloat('SOURCES', 'sigma_clipped_sigma',
                                                        fallback=self.sigma_clipped_sigma)
        self.sigma_clip_sigma = self._config.getfloat('SOURCES', 'sigma_clip_sigma', fallback=self.sigma_clip_sigma)
        self.leveling_filter_downscale_factor = self._config.getint('SOURCES', 'leveling_filter_downscale_factor',
                                                                   fallback=self.leveling_filter_downscale_factor)

        self.src_box_size_x = self._config.getint('SOURCES', 'src_box_size_x', fallback=self.src_box_size_x)
        self.src_box_size_y = self._config.getint('SOURCES', 'src_box_size_y', fallback=self.src_box_size_y)
        self.src_filter_size_x = self._config.getint('SOURCES', 'src_filter_size_x', fallback=self.src_filter_size_x)
        self.src_filter_size_y = self._config.getint('SOURCES', 'src_filter_size_y', fallback=self.src_filter_size_y)

        self.src_kernal_size_x = self._config.getint('SOURCES', 'src_kernal_size_x', fallback=self.src_kernal_size_x)
        self.src_kernal_size_y = self._config.getint('SOURCES', 'src_kernal_size_y', fallback=self.src_kernal_size_y)
        self.src_sigma_x = self._config.getint('SOURCES', 'src_sigma_x', fallback=self.src_sigma_x)
        self.src_dst = self._config.getint('SOURCES', 'src_dst', fallback=self.src_dst)

        self.photutils_gaussian_kernal_fwhm = self._config.getfloat('SOURCES', 'photutils_gaussian_kernal_fwhm',
                                                                   fallback=self.photutils_gaussian_kernal_fwhm)
        self.photutils_kernal_size = self._config.getfloat('SOURCES', 'photutils_kernal_size',
                                                          fallback=self.photutils_kernal_size)
        self.dilate_mask_iterations = self._config.getint('SOURCES', 'dilate_mask_iterations',
                                                         fallback=self.dilate_mask_iterations)

        self.dilation_radius = self._config.getint('SOURCES', 'dilation_radius', fallback=self.dilation_radius)
        self.min_potential_source_distance = self._config.getint('SOURCES', 'min_potential_source_distance',
                                                                fallback=self.min_potential_source_distance)

        self.level_filter = self._config.getint('SOURCES', 'level_filter', fallback=self.level_filter)
        self.level_filter_type = self._config.get('SOURCES', 'level_filter_type', fallback=self.level_filter_type)
        self.ring_filter_type = self._config.get('SOURCES', 'ring_filter_type', fallback=self.ring_filter_type)

        # [STARTRACKER]
        self.star_tracker_body_rates_max_distance = self._config.getint('STARTRACKER',
                                                                       'star_tracker_body_rates_max_distance',
                                                                       fallback=self.star_tracker_body_rates_max_distance)
        self.focal_ratio = self._config.getfloat('STARTRACKER', 'focal_ratio', fallback=self.focal_ratio)
        self.x_pixel_count = self._config.getint('STARTRACKER', 'x_pixel_count', fallback=self.x_pixel_count)
        self.y_pixel_count = self._config.getint('STARTRACKER', 'y_pixel_count', fallback=self.y_pixel_count)

        # [ASTROMETRY]
        self.ast_t3_database = self._config.get('ASTROMETRY', 'ast_t3_database', fallback=self.ast_t3_database)
        self.ast_is_array = self._config.getboolean('ASTROMETRY', 'ast_is_array', fallback=self.ast_is_array)
        self.ast_is_trail = self._config.getboolean('ASTROMETRY', 'ast_is_trail', fallback=self.ast_is_trail)
        self.ast_use_photoutils = self._config.getboolean('ASTROMETRY', 'ast_use_photoutils',
                                                         fallback=self.ast_use_photoutils)
        self.ast_subtract_global_bkg = self._config.getboolean('ASTROMETRY', 'ast_subtract_global_bkg',
                                                               fallback=self.ast_subtract_global_bkg)
        self.ast_fast = self._config.getboolean('ASTROMETRY', 'ast_fast', fallback=self.ast_fast)
        self.ast_bkg_threshold = self._config.getfloat('ASTROMETRY', 'ast_bkg_threshold',
                                                      fallback=self.ast_bkg_threshold)
        self.ast_number_sources = self._config.getint('ASTROMETRY', 'ast_number_sources',
                                                     fallback=self.ast_number_sources)
        self.ast_min_size = self._config.getint('ASTROMETRY', 'ast_min_size', fallback=self.ast_min_size)
        self.ast_max_size = self._config.getint('ASTROMETRY', 'ast_max_size', fallback=self.ast_max_size)

        self.min_pattern_checking_stars = self._config.getint('ASTROMETRY', 'min_pattern_checking_stars',
                                                             fallback=self.min_pattern_checking_stars)
        self.include_angular_velocity = self._config.getboolean('ASTROMETRY', 'include_angular_velocity',
                                                               fallback=self.include_angular_velocity)

        self.angular_velocity_timeout = self._config.getfloat('ASTROMETRY', 'angular_velocity_timeout',
                                                             fallback=self.angular_velocity_timeout)

        self.solve_timeout = self._config.getfloat('ASTROMETRY', 'solve_timeout', fallback=self.solve_timeout)

        # [CEDAR}
        self.cedar_detect_host = self._config.get('CEDAR', 'host', fallback=self.cedar_detect_host)
        self.sigma = self._config.get('CEDAR', 'sigma', fallback=self.sigma)
        self.max_size = self._config.getint('CEDAR', 'max_size', fallback=self.max_size)
        self.binning = self._config.getint('CEDAR', 'binning', fallback=self.binning)
        self.binning = self.binning or self.binning != 0
        self.return_binned = self._config.getboolean('CEDAR', 'return_binned', fallback=self.return_binned)
        self.use_binned_for_star_candidates = self._config.getboolean('CEDAR', 'use_binned_for_star_candidates',
                                                                     fallback=self.use_binned_for_star_candidates)
        self.detect_hot_pixels = self._config.getboolean('CEDAR', 'detect_hot_pixels',
                                                        fallback=self.detect_hot_pixels)

        # Custom Pixel Count and Spatial Filtering
        self.pixel_count_min = self._config.getint('CEDAR', 'pixel_count_min', fallback=self.pixel_count_min)
        self.pixel_count_max = self._config.getint('CEDAR', 'pixel_count_max', fallback=self.pixel_count_max)
        self.spatial_distance_px = self._config.getint('CEDAR', 'spatial_distance_px', fallback=self.spatial_distance_px)
        self.gaussian_fwhm = self._config.getfloat('CEDAR', 'gaussian_fwhm', fallback=self.gaussian_fwhm)
        self.cedar_downscale = self._config.getfloat('CEDAR', 'cedar_downscale', fallback=self.cedar_downscale)

        # [ASTROMETRY.NET]
        self.an_scale_units = self._config.get('ASTROMETRY.NET', 'scale_units', fallback=self.an_scale_units)
        self.an_scale_low = self._config.getfloat('ASTROMETRY.NET', 'scale_low', fallback=self.an_scale_low)
        self.an_scale_high = self._config.getfloat('ASTROMETRY.NET', 'scale_high', fallback=self.an_scale_high)
        self.an_downsample = self._config.getint('ASTROMETRY.NET', 'downsample', fallback=self.an_downsample)
        self.an_overwrite  = self._config.getboolean('ASTROMETRY.NET', 'overwrite ', fallback=self.an_overwrite)
        self.an_no_plots  = self._config.getboolean('ASTROMETRY.NET', 'no_plots ', fallback=self.an_no_plots )
        self.an_cpulimit = self._config.getint('ASTROMETRY.NET', 'cpulimit', fallback=self.an_cpulimit)
        self.an_depth = self._config.getint('ASTROMETRY.NET', 'depth', fallback=self.an_depth)

        self.an_sigma = self._config.getfloat('ASTROMETRY.NET', 'sigma', fallback=self.an_sigma)
        self.an_nsigma = self._config.getfloat('ASTROMETRY.NET', 'nsigma', fallback=self.an_nsigma)

        self.an_corr = self._config.getboolean('ASTROMETRY.NET', 'corr', fallback=self.an_depth)
        self.an_new_fits = self._config.getboolean('ASTROMETRY.NET', 'new_fits', fallback=self.an_new_fits)
        self.an_match = self._config.getboolean('ASTROMETRY.NET', 'match', fallback=self.an_match)
        self.an_solved = self._config.getboolean('ASTROMETRY.NET', 'solved', fallback=self.an_solved)

        # [IMAGES]
        self.png_compression = self._config.getint('IMAGES', 'png_compression', fallback=self.png_compression)
        self.save_raw = self._config.getboolean('IMAGES', 'save_raw', fallback=self.save_raw)
        self.min_count = self._config.getint('IMAGES', 'min_count', fallback=self.min_count)
        self.sigma_error_value = self._config.getint('IMAGES', 'sigma_error_value', fallback=self.sigma_error_value)
        self.return_partial_images = self._config.getboolean('IMAGES', 'return_partial_images',
                                                            fallback=self.return_partial_images)
        # Downscale - SD images
        self.resize_mode = self._config.get('IMAGES', 'resize_mode', fallback=self.resize_mode)
        self.scale_factor_x = self._config.getfloat('IMAGES', 'scale_factor_x', fallback=self.scale_factor_x)
        self.scale_factor_y = self._config.getfloat('IMAGES', 'scale_factor_y', fallback=self.scale_factor_y)
        self.scale_factors = (self.scale_factor_x, self.scale_factor_y)

        # Downscale - RAW images
        self.raw_resize_mode = self._config.get('IMAGES', 'raw_resize_mode', fallback=self.raw_resize_mode)
        self.raw_scale_factor_x = self._config.getfloat('IMAGES', 'raw_scale_factor_x', fallback=self.raw_scale_factor_x)
        self.raw_scale_factor_y = self._config.getfloat('IMAGES', 'raw_scale_factor_y', fallback=self.raw_scale_factor_y)
        self.raw_scale_factors = (self.raw_scale_factor_x, self.raw_scale_factor_y)

        self.inspection_images_keep = self._config.getint('IMAGES', 'inspection_images_keep', fallback=self.inspection_images_keep)
        self.inspection_quality = self._config.getint('IMAGES', 'inspection_quality', fallback=self.inspection_quality)
        self.inspection_lower_percentile = self._config.getint('IMAGES', 'inspection_lower_percentile', fallback=self.inspection_lower_percentile)
        self.inspection_upper_percentile = self._config.getint('IMAGES', 'inspection_upper_percentile', fallback=self.inspection_upper_percentile)
        self.inspection_last_image_symlink_name = self._config.get('IMAGES', 'inspection_last_image_symlink_name', fallback=self.inspection_last_image_symlink_name)

        # [GENERAL]
        self.flight_mode = self._config.get('GENERAL', 'flight_mode', fallback=self.flight_mode)
        self.solver = self._config.get('GENERAL', 'solver', fallback=self.solver)
        self.time_interval = self._config.getint('GENERAL', 'time_interval', fallback=self.time_interval)
        self.max_processes = self._config.getint('GENERAL', 'max_processes', fallback=self.max_processes)
        self.operation_timeout = self._config.getint('GENERAL', 'operation_timeout', fallback=self.operation_timeout)
        self.current_timeout = self._config.getfloat('GENERAL', 'current_timeout', fallback=self.current_timeout)
        self.run_autofocus = self._config.getboolean('GENERAL', 'run_autofocus', fallback=self.run_autofocus)
        self.enable_autogain_with_autofocus = self._config.getboolean('GENERAL', 'enable_autogain_with_autofocus', fallback=self.enable_autogain_with_autofocus)

        self.run_autonomous = self._config.getboolean('GENERAL', 'run_autonomous', fallback=self.run_autonomous)
        self.run_telemetry = self._config.getboolean('GENERAL', 'run_telemetry', fallback=self.run_telemetry)
        self.run_chamber = self._config.getboolean('GENERAL', 'run_chamber', fallback=self.run_chamber)
        self.run_test = self._config.getboolean('GENERAL', 'run_test', fallback=self.run_test)

        # [PATHS] section
        # auto_gain_image_path
        self.auto_gain_image_path_tmpl = self._config.get('PATHS', 'auto_gain_image_path_tmpl',
                                                         fallback=self.auto_gain_image_path_tmpl)
        self.focus_image_path_tmpl = self._config.get('PATHS', 'focus_image_path_tmpl',
                                                     fallback=self.focus_image_path_tmpl)
        self.ultrafine_focus_image_path_tmpl = self._config.get('PATHS', 'ultrafine_focus_image_path_tmpl',
                                                               fallback=self.ultrafine_focus_image_path_tmpl)

        self.partial_results_path = self._config.get('PATHS', 'partial_results_path', fallback=self.partial_results_path)
        self.partial_image_name = self._config.get('PATHS', 'partial_image_name', fallback=self.partial_image_name)
        self.astro_path = self._config.get('PATHS', 'astro_path', fallback=self.astro_path)

        self.ssd_path = self._config.get('PATHS', 'ssd_path', fallback=self.ssd_path)
        self.sd_card_path = self._config.get('PATHS', 'sd_card_path', fallback=self.sd_card_path)
        self.final_path = self._config.get('PATHS', 'final_path', fallback=self.final_path)

        self.calibration_params_file = self._config.get('PATHS', 'calibration_params_file',
                                                       fallback=self.calibration_params_file)
        self.gui_images_path = self._config.get('PATHS', 'gui_images_path', fallback=self.gui_images_path)

        self.inspection_path = self._config.get('PATHS', 'inspection_path', fallback=self.inspection_path)

        self.web_path = self._config.get('PATHS', 'web_path', fallback=self.web_path)

        # Create inspection_settings dict
        self.inspection_settings = {
            'images_keep': self.inspection_images_keep,
            'quality': self.inspection_quality,
            'lower_percentile': self.inspection_lower_percentile,
            'upper_percentile': self.inspection_upper_percentile,
            'path': self.inspection_path,
            'last_image_symlink_name': self.inspection_last_image_symlink_name,
            'web_path': self.web_path
        }

        # [DEVICES]
        self.focuser_port = self._config.get('DEVICES', 'focuser_port', fallback=self.focuser_port)
        self.computer_port = self._config.get('DEVICES', 'computer_port', fallback=self.computer_port)
        self.baud_rate = self._config.getint('DEVICES', 'baud_rate', fallback=self.baud_rate)
        self.telemetry_port = self._config.get('DEVICES', 'telemetry_port', fallback=self.telemetry_port)
        self.telemetry_baud_rate = self._config.getint('DEVICES', 'telemetry_baud_rate',
                                                      fallback=self.telemetry_baud_rate)

        # [STAR_COMM_BRIDGE]
        self.pueo_server_ip = self._config.get('STAR_COMM_BRIDGE', 'pueo_server_ip', fallback=self.pueo_server_ip)
        self.server_ip = self._config.get('STAR_COMM_BRIDGE', 'server_ip', fallback=self.server_ip)
        self.port = self._config.getint('STAR_COMM_BRIDGE', 'port', fallback=self.port)
        self.socket_timeout = self._config.getint('STAR_COMM_BRIDGE', 'socket_timeout', fallback=self.socket_timeout)
        self.max_retry = self._config.getint('STAR_COMM_BRIDGE', 'max_retry', fallback=self.max_retry)
        self.retry_delay = self._config.getint('STAR_COMM_BRIDGE', 'retry_delay', fallback=self.retry_delay)
        self.fq_max_size = self._config.getint('STAR_COMM_BRIDGE', 'fq_max_size', fallback=self.fq_max_size)
        self.msg_max_size = self._config.getint('STAR_COMM_BRIDGE', 'msg_max_size', fallback=self.msg_max_size)

        # [GUI]
        self.enable_gui_data_exchange = self._config.getboolean('GUI', 'enable_gui_data_exchange', fallback=self.enable_gui_data_exchange)
        self.images_keep = self._config.getint('GUI', 'images_keep', fallback=self.images_keep)
        self.log_reverse = self._config.getboolean('GUI', 'log_reverse', fallback=self.log_reverse)

    def save(self, config_file='conf/config.ini'):
        """Load config and update only the ones applicable"""

        config = self.read(config_file)

        config['GLOBAL'] = {
            'test': self.test,
            'debug': self.debug,
            # 'telemetry_log': self.telemetry_log
            # 'new_var': self.new_var
        }

        config['LENS_FOCUS_CONSTANTS'] = {
            'trial_focus_pos': self.trial_focus_pos,

            'autofocus_start_position': self.autofocus_start_position,
            'autofocus_stop_position': self.autofocus_stop_position,
            'autofocus_step_count': self.autofocus_step_count,
            'autofocus_method': self.autofocus_method,

            'top_end_span_percentage': self.top_end_span_percentage,
            'bottom_end_span_percentage': self.bottom_end_span_percentage,
            'coarse_focus_step_count': self.coarse_focus_step_count,
            'fine_focus_step_count': self.fine_focus_step_count,
            'lens_focus_homed': self.lens_focus_homed,
            'exposure_time': self.exposure_time,
            'pixel_bias': self.pixel_bias,
            'env_filename': self.env_filename,
            'autogain_update_interval': self.autogain_update_interval,
            'autogain_num_bins': self.autogain_num_bins,
            'autoexposure_num_bins': self.autoexposure_num_bins,
            'max_autogain_iterations': self.max_autogain_iterations,
            'max_autoexposure_iterations': self.max_autoexposure_iterations,
            'max_gain_setting': self.max_gain_setting,
            'min_gain_setting': self.min_gain_setting,
            'autogain_desired_max_pixel_value': self.autogain_desired_max_pixel_value,
            'autoexposure_desired_max_pixel_value': self.autoexposure_desired_max_pixel_value,
            'lab_best_aperture_position': self.lab_best_aperture_position,
            'lab_best_focus': self.lab_best_focus,
            'lab_best_gain': self.lab_best_gain,
            'lab_best_exposure': self.lab_best_exposure,
            'autofocus_max_deviation': self.autofocus_max_deviation,
            'focus_tolerance_percentage': self.focus_tolerance_percentage,
            'fit_points_init': self.fit_points_init,
            'lab_fov': self.lab_fov,
            'lab_distortion_coefficient_1': self.lab_distortion_coefficient_1,
            'lab_distortion_coefficient_2': self.lab_distortion_coefficient_2,
            'delta_focus_steps': self.delta_focus_steps,
            'delta_aperture': self.delta_aperture,
            'delta_exposure_time': self.delta_exposure_time,
            'delta_gain': self.delta_gain
        }

        config['CAMERA'] = {
            'asi_gama': self.asi_gama,
            'asi_exposure': self.asi_exposure,
            'asi_gain': self.asi_gain,
            'asi_flip': self.asi_flip,
            'roi_bins': self.roi_bins,
            'pixel_saturated_value_raw8': self.pixel_saturated_value_raw8,
            'pixel_saturated_value_raw16': self.pixel_saturated_value_raw16,
            'stdev_error_value': self.stdev_error_value,
            'camera_id_vendor': hex(self.camera_id_vendor),
            'camera_id_product': hex(self.camera_id_product),
            'pixel_well_depth': self.pixel_well_depth,
            'sbc_dio_camera_pin': self.sbc_dio_camera_pin,
            'sbc_dio_focuser_pin': self.sbc_dio_focuser_pin,
            'sbc_dio_default': self.sbc_dio_default,
            'sbc_dio_cycle_delay': self.sbc_dio_cycle_delay,
            'power_cycle_wait': self.power_cycle_wait
        }

        config['SOURCES'] = {
            'img_bkg_threshold': self.img_bkg_threshold,
            'img_number_sources': self.img_number_sources,
            'img_min_size': self.img_min_size,
            'img_max_size': self.img_max_size,
            'local_sigma_cell_size': self.local_sigma_cell_size,
            'sigma_clipped_sigma': self.sigma_clipped_sigma,
            'sigma_clip_sigma': self.sigma_clip_sigma,
            'leveling_filter_downscale_factor': self.leveling_filter_downscale_factor,
            'src_box_size_x': self.src_box_size_x,
            'src_box_size_y': self.src_box_size_y,
            'src_filter_size_x': self.src_filter_size_x,
            'src_filter_size_y': self.src_filter_size_y,
            'src_kernal_size_x': self.src_kernal_size_x,
            'src_kernal_size_y': self.src_kernal_size_y,
            'src_sigma_x': self.src_sigma_x,
            'src_dst': self.src_dst,
            'photutils_gaussian_kernal_fwhm': self.photutils_gaussian_kernal_fwhm,
            'photutils_kernal_size': self.photutils_kernal_size,
            'dilate_mask_iterations': self.dilate_mask_iterations,
            'dilation_radius': self.dilation_radius,
            'min_potential_source_distance': self.min_potential_source_distance,
            'level_filter': self.level_filter,
            'level_filter_type': self.level_filter_type,
            'ring_filter_type': self.ring_filter_type
        }

        config['STARTRACKER'] = {
            'star_tracker_body_rates_max_distance': self.star_tracker_body_rates_max_distance,
            'focal_ratio': self.focal_ratio,
            'x_pixel_count': self.x_pixel_count,
            'y_pixel_count': self.y_pixel_count
        }

        config['ASTROMETRY'] = {
            'ast_t3_database': self.ast_t3_database,
            'ast_is_array': self.ast_is_array,
            'ast_is_trail': self.ast_is_trail,
            'ast_use_photoutils': self.ast_use_photoutils,
            'ast_subtract_global_bkg': self.ast_subtract_global_bkg,
            'ast_fast': self.ast_fast,
            'ast_bkg_threshold': self.img_bkg_threshold,
            'ast_number_sources': self.img_number_sources,
            'ast_min_size': self.img_min_size,
            'ast_max_size': self.img_max_size,
            'min_pattern_checking_stars': self.min_pattern_checking_stars,
            'include_angular_velocity': self.include_angular_velocity,
            'angular_velocity_timeout': self.angular_velocity_timeout,
            'solve_timeout': self.solve_timeout
        }

        config['CEDAR'] = {
            'host': self.cedar_detect_host,
            'sigma': self.sigma,
            'max_size': self.max_size,
            'binning': self.binning,
            'return_binned': self.return_binned,
            'use_binned_for_star_candidates': self.use_binned_for_star_candidates,
            'detect_hot_pixels': self.detect_hot_pixels,
            'pixel_count_min':  self.pixel_count_min,
            'pixel_count_max':  self.pixel_count_max,
            'spatial_distance_px': self.spatial_distance_px,
            'gaussian_fwhm': self.gaussian_fwhm,
            'cedar_downscale': self.cedar_downscale
        }

        config['IMAGES'] = {
            'png_compression': self.png_compression,
            'save_raw': self.save_raw,
            'min_count': self.min_count,
            'sigma_error_value': self.sigma_error_value,
            'return_partial_images': self.return_partial_images,

            'resize_mode': self.resize_mode,
            'scale_factor_x': self.scale_factor_x,
            'scale_factor_y': self.scale_factor_y,

            'raw_resize_mode': self.raw_resize_mode,
            'raw_scale_factor_x': self.raw_scale_factor_x,
            'raw_scale_factor_y': self.raw_scale_factor_y,

            'inspection_images_keep': self.inspection_images_keep,
            'inspection_quality': self.inspection_quality,
            'inspection_lower_percentile': self.inspection_lower_percentile,
            'inspection_upper_percentile': self.inspection_upper_percentile,
            'inspection_last_image_symlink_name': self.inspection_last_image_symlink_name
        }

        config['PATHS'] = {
            'auto_gain_image_path_tmpl': self.auto_gain_image_path_tmpl,
            'focus_image_path_tmpl': self.focus_image_path_tmpl,
            'ultrafine_focus_image_path_tmpl': self.ultrafine_focus_image_path_tmpl,
            'partial_results_path': self.partial_results_path,
            'partial_image_name': self.partial_image_name,
            'astro_path': self.astro_path,
            'ssd_path': self.ssd_path,
            'final_path': self.final_path,
            'calibration_params_file': self.calibration_params_file,
            'gui_images_path': self.gui_images_path,
            'inspection_path': self.inspection_path,
            'web_path': self.web_path
        }

        config['GENERAL'] = {
            'flight_mode': self.flight_mode,
            'solver': self.solver,
            'time_interval': self.time_interval,
            'max_processes': self.max_processes,
            'operation_timeout': self.operation_timeout,
            'current_timeout': self.current_timeout,
            'run_autofocus': self.run_autofocus,
            'enable_autogain_with_autofocus': self.enable_autogain_with_autofocus,
            'run_autonomous': self.run_autonomous,
            'run_telemetry': self.run_telemetry,
            'run_chamber': self.run_chamber,
            'run_test': self.run_test
        }

        config['DEVICES'] = {
            'focuser_port': self.focuser_port,
            'computer_port': self.computer_port,
            'baud_rate': self.baud_rate,
            'telemetry_port': self.telemetry_port,
            'telemetry_baud_rate': self.telemetry_baud_rate
        }

        config['STAR_COMM_BRIDGE'] = {
            'pueo_server_ip': self.pueo_server_ip,
            'server_ip': self.server_ip,
            'port': self.port,
            'socket_timeout': self.socket_timeout,
            'max_retry': self.max_retry,
            'retry_delay': self.retry_delay,
            'fq_max_size': self.fq_max_size,
            'msg_max_size': self.msg_max_size
        }

        config['GUI'] = {
            'enable_gui_data_exchange': self.enable_gui_data_exchange,
            'images_keep': self.images_keep,
            'log_reverse': self.log_reverse
        }

        save_config_file = 'conf/config_save_test.ini'
        print(f'Saving config.ini: {save_config_file}')
        with open(save_config_file, 'w') as file:
            config.write(file)


# Example usage (optional)
if __name__ == "__main__":
    cfg = Config(config_file="../conf/config.ini", dynamic_file="../conf/dynamic.ini")

    # Access configuration values
    # my_value = cfg.get("database", "host")
    print(f"TEST: {cfg.test}")
    # print(f"NEW VAR: {cfg.new_var}")

    # Example of adding new var
    # cfg.new_var = 'Super happy'
    # cfg.save()
    def log():
        print(f'lab_best_focus: {cfg.lab_best_focus}')
        print(f'  solver: {cfg.solver}')
        print(f'  run_autonomous: {cfg.run_autonomous}')

    print(f'Orig')
    log()

    print(f'Updated')
    cfg.set_dynamic(solver='solver1', run_autonomous=True, lab_best_focus=8355)
    log()
