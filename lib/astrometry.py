# Standard imports
from contextlib import suppress
import logging
import sys
import time
from time import perf_counter as precision_timestamp
import os
from pathlib import Path
# External imports
import matplotlib.pyplot as plt
import cv2
import numpy as np

from tqdm import tqdm

# Custom imports
from lib.config import Config
from lib.common import current_timestamp, cprint, get_dt
from lib.utils import read_image_grayscale, read_image_BGR, display_overlay_info, timed_function, print_img_info
from lib.source_finder import global_background_estimation, local_levels_background_estimation, \
    median_background_estimation, sextractor_background_estimation, find_sources, find_sources_photutils, \
    select_top_sources, select_top_sources_photutils, source_finder
from lib.compute_star_centroids import compute_centroids_from_still, compute_centroids_from_trail, \
    compute_centroids_photutils
from lib.utils import get_files, split_path
from lib.tetra3 import Tetra3
from lib.cedar_solve import Tetra3 as Tetra3Cedar  # , cedar_detect_client
from lib.cedar import Cedar

# from lib.tetra3_cedar_old import Tetra3 as Tetra3Cedar # , cedar_detect_client


class Astrometry:
    """
    Note:
        solver: 'solver1' ~ genuine tetra3 | 'solver2' ~ cedar tetra3
    Params:
        database_name

    """
    _t3 = None
    test_data = {}

    def __init__(self, database_name=None, cfg=None):
        self.log = logging.getLogger('pueo')
        self.log.info('Initializing Astrometry Object')
        self.test = False
        self.database_name = database_name
        self.cfg = cfg
        self.solver = 'solver2'  # Default solver

        # apply distortion calibration
        if database_name is None:
            db_file = '../data/default_database.npz'
            self.database_name = db_file if os.path.exists(db_file) else 'data/default_database.npz'
        self.database_path = Path(self.database_name).resolve()

        # Create instance
        # if self.solver == 'solver1':
        self.t3_genuine = Tetra3(load_database=self.database_path)
        # Load a database
        self.log.info(f'Database for Genuine Tetra3: {self.database_path}')
        # self.t3_genuine.load_database(path=self.database_path)
        # else
        self.t3_cedar = Tetra3Cedar(load_database=self.database_path)
        # TODO: Remove cedar_detect if not used
        # with suppress(AttributeError, Exception):
        #     self.cedar_detect = cedar_detect_client.CedarDetectClient()
        self.log.info(f'Database for Cedar Tetra3: {self.database_path}')
        # self.t3_cedar.load_database(path=self.database_path)

        self.cedar = Cedar(database_name, cfg)
        self.cedar.test = self.test

    @property
    def solver_name(self):
        return 'Cedar Tetra3' if self.solver == 'solver2' else 'ESA Tetra3'

    @property
    def t3(self):
        self.log.debug(f'Serving astro solver: {self.solver}')
        cprint(f'Serving astro solver: {self.solver}', 'cyan')
        return self.t3_genuine if self.solver == 'solver1' else self.t3_cedar

    def tetra3_generate_database(
            self,
            star_catalog: str,
            max_fov: float,
            star_max_magnitude: int,
            output_name: str) -> None:
        """Generate a new star database.

        Args:
            star_catalog (str): Abbreviated name of the star catalog to use. Must be one of 'bsc5', 'hip_main', or 'tyc_main'.
            max_fov (float): Maximum angle (in degrees) allowed between stars in the same pattern. This determines
                             the angular separation for star grouping.
            star_max_magnitude (int): Dimmest apparent magnitude of stars to include in the database. Stars
                                       fainter than this magnitude will be excluded.
            output_name (str or pathlib.Path): The file path or name where the generated star catalog will be saved.
        """
        # Generate and save database
        self.t3.generate_database(
            star_catalog=star_catalog, max_fov=max_fov, star_max_magnitude=star_max_magnitude, save_as=output_name
        )

    def tetra3_solver(
            self,
            sources_mask: np.ndarray,
            precomputed_star_centroids,
            FOV=None,
            distortion=0,
            min_pattern_checking_stars=15,
            resize_factor=1.0
    ) -> dict:
        """Find the direction in the sky of an image by calculating a "fingerprint" of the star centroids detected in the image and looking for matching fingerprints in
            a pattern catalog loaded into memory.

        Args:
            sources_mask (np.ndarray): A binary mask of detected point sources in the image, indicating the positions of stars.
            precomputed_star_centroids (np.ndarray): An array containing the precomputed centroids of the detected stars,
                                                     which are used for pattern recognition.
            FOV (float, optional): The estimated horizontal field of view of the image in degrees. Default is None.
            distortion (float, optional): The radial distortion factor to correct for lens distortion. Default is 0.
            min_pattern_checking_stars (int, optional): Number of stars used to create possible patterns to look up in database. Default is 15.

        Returns:
            dict: A dictionary containing the results of the sky direction solution, with the following keys:
                - 'RA' (float): Right Ascension in degrees, representing the celestial longitude of the solved image.
                - 'Dec' (float): Declination in degrees, representing the celestial latitude of the solved image.
                - 'Roll' (float): Roll angle in degrees, representing the rotation angle of the camera with respect to a reference direction.
                - 'FOV' (float): Horizontal Field of View in degrees, representing the angular extent of the image in the horizontal direction.
                - 'distortion' (float): Calculated distortion of the provided image
                - 'Cross-Boresight RMSE' (float): Root Mean Square Error of the cross-boresight alignment in arcseconds, indicating the accuracy of alignment.
                - 'Roll RMSE' (float): Root Mean Square Error of the roll angle estimation in arcseconds, indicating the accuracy of roll angle estimation.
                - 'Matches' (int): Number of star matches used for the solution.
                - 'Prob' (float): Probability of the solution, indicating the confidence level of the match.
                - 'T_solve' (float): Time taken to solve the image in seconds.
                - 'matched_centroids' : An Mx2 list with the (y, x) pixel coordinates in the image corresponding to each matched star
                - 'visual' : A PIL image with spots for the given centroids in white, the coarse FOV and distortion estimates in orange,
                the final FOV and distortion estimates in green. Also has circles for the catalogue stars in green or red for successful/unsuccessful match
        """
        height, width = sources_mask.shape
        # number of pattern checking stars used
        # TODO: Done 15 min_pattern_checking_stars; Implemented min_pattern_checking_stars as input param on functions: tetra3_solver, optimize_calibration_parameters, do_astrometry, config.ini
        pattern_checking_stars = min(min_pattern_checking_stars, len(precomputed_star_centroids))
        cprint(f"Solving: pattern_checking_stars: {min_pattern_checking_stars}, precomputed_star_centroids: {len(precomputed_star_centroids)}", color='cyan')
        # Solving using centroids
        result = self.t3.solve_from_centroids(
            star_centroids=precomputed_star_centroids,
            size=(int(height/resize_factor), int(width/resize_factor)),
            fov_estimate=FOV,
            # fov_max_error=FOV * 0.10,  # Default None
            # pattern_checking_stars=pattern_checking_stars,
            # match_radius=0.01,  # Default 0.01
            # match_threshold=1e-3,
            solve_timeout=5000.0, # Default None milliseconds
            # distortion=distortion,
            return_matches=True,
            # return_visual=False,
        )
        #         star_centroids,
        #         size,
        #         fov_estimate=None,
        #         fov_max_error=None,
        #         pattern_checking_stars=8,
        #         match_radius=0.01,
        #         match_threshold=1e-3,
        #         solve_timeout=None,
        #         target_pixel=None,
        #         distortion=0,
        #         return_matches=False,
        #         return_visual=False,

        print(f"solve_from_centroids : done")
        # plt.imshow(result['visual'])
        # plt.show(block=True)
        return result

    def optimize_calibration_parameters(
            self,
            is_trail,
            number_sources,
            bkg_threshold,
            min_size,
            max_size,
            calibration_images_dir,
            log_file_path="log/calibration_log.txt",
            calibration_params_path="calibration_params.txt",
            update_calibration=False,
            min_pattern_checking_stars=15,
            local_sigma_cell_size=36,
            sigma_clipped_sigma=3.0,
            leveling_filter_downscale_factor=4,
            src_kernal_size_x=3,
            src_kernal_size_y=3,
            src_sigma_x=1,
            src_dst=1,
            dilate_mask_iterations=1,
            scale_factors=(8, 8),
            resize_mode='downscale'
    ):
        """Optimize camera calibration parameters based on astrometry analysis of
        calibration images.

        This function computes and updates the optimal calibration parameters for a camera system
        using a set of calibration images. It performs astrometry on the images to derive
        parameters like the Field of View (FOV) and distortion coefficients, and assesses
        the resulting parameters against previously stored best parameters.

        Args:
            is_trail (bool): Indicates if the sources in the image are trails (True) or points (False).
            number_sources (int): The maximum number of sources to detect in the images.
            bkg_threshold (float): The threshold value used to identify sources. Pixels whose values exceed
                `local background + threshold * local noise` will be marked as source pixels.
            min_size (int): Minimum size of detected sources (in pixels) to consider for calibration.
            max_size (int): Maximum size of detected sources (in pixels) to consider for calibration.
            calibration_images_dir (str): Directory containing calibration images for processing.
            log_file_path (str, optional): Path to the log file for recording calibration logs. Defaults to "log/calibration_log.txt".
            calibration_params_path (str, optional): Path to the file for saving calibration parameters. Defaults to "calibration_params.txt".
            update_calibration (bool, optional): If True, updates the existing calibration parameters with new images. Defaults to False.
            min_pattern_checking_stars (int, optional): Number of stars used to create possible patterns to look up in database. Default is 15.
            local_sigma_cell_size (int, optional): The size of the cells (in pixels) used for estimating local noise
                levels. Defaults to 36.
            sigma_clipped_sigma (float, optional): The sigma value to use for sigma-clipping when calculating the
                background statistics (mean, median, standard deviation). It controls how aggressive the clipping is.
                Higher values mean less clipping (more tolerance for outliers). Defaults to 3.0.
            leveling_filter_downscale_factor (int, optional): The downscaling factor to apply when creating the
                downsampled image used for local level estimation. Defaults to 4.
            src_kernal_size_x (int, optional): The width of the Gaussian kernel used for smoothing the image. Defaults to 3.
            src_kernal_size_y (int, optional): The height of the Gaussian kernel used for smoothing the image. Defaults to 3.
            src_sigma_x (float, optional): The standard deviation in the X direction for the Gaussian kernel. Defaults to 1.
            src_dst (int, optional): The depth of the output image. Defaults to 1.
            dilate_mask_iterations (int, optional): The number of iterations for dilating the mask to merge nearby
                sources. A higher value merges more pixels. Defaults to 1.
            scale_factors=(float, float optional): Downscaling factors
            resize_mode=(str, optional): resize mode

        Returns:
            dict: A dictionary containing the optimized calibration parameters, including:
                - 'FOV': The average field of view determined from the calibration images.
                - 'distortion': A list containing distortion coefficients calculated from the images.
                - 'RMSE': The root mean square error associated with the new calibration parameters.
        """
        # list of the default best params
        default_best_params = {"FOV": None, "distortion": [-0.1, 0.1]}

        curr_params = {"FOV": [], "distortion": []}
        prev_params = {"FOV": [], "distortion": []}
        curr_best_params = {}
        prev_best_params = {}

        # read previous list of params from file
        if update_calibration:
            try:
                with open(calibration_params_path, "r") as file:
                    lines = file.readlines()
                    for line in lines:
                        line = line.strip()
                        if line:
                            key, value = line.split(":")
                            prev_params[key.strip()] = [float(v.strip()) for v in value.split(",")]
            except FileNotFoundError:
                print("Error: Distortion calibration file not found.")

            # update current calibration with new images
            curr_params = prev_params

        # read best distortions calibration parameters
        try:
            with open("calibration_params_best.txt", "r") as file:
                lines = file.readlines()
                for line in lines:
                    line = line.strip()
                    if line:
                        key, value = line.split(":")
                        prev_best_params[key.strip()] = float(value.strip())
                        default_best_params = prev_best_params
        except FileNotFoundError:
            print("Error: Distortion calibration file not found.")

        # Check if the directory is empty
        if not os.listdir(calibration_images_dir):
            print(f"The distortion calibration images directory {calibration_images_dir} is empty.")
            if prev_best_params != {}:
                print(f"returning previous calibration params.")
                return prev_best_params
            else:
                print(f"returning default calibration params.")
                return default_best_params
        else:
            # get calibration params from the new set of images
            for filename in os.listdir(calibration_images_dir):
                # calibration image path
                image_path = os.path.join(calibration_images_dir, filename)
                # do astrometry
                astrometry, _, contours_img = self.do_astrometry(
                    img=image_path,
                    is_array=False,
                    is_trail=is_trail,
                    use_photutils=False,  # TODO: NEW
                    subtract_global_bkg=False,  # TODO: NEW
                    fast=False,  # TODO: NEW
                    number_sources=number_sources,
                    bkg_threshold=bkg_threshold,
                    min_size=min_size,
                    max_size=max_size,
                    distortion_calibration_params=default_best_params,
                    log_file_path=log_file_path,
                    min_pattern_checking_stars=min_pattern_checking_stars,
                    local_sigma_cell_size=local_sigma_cell_size,
                    sigma_clipped_sigma=sigma_clipped_sigma,
                    leveling_filter_downscale_factor=leveling_filter_downscale_factor,
                    src_kernal_size_x=src_kernal_size_x,
                    src_kernal_size_y=src_kernal_size_y,
                    src_sigma_x=src_sigma_x,
                    src_dst=src_dst,
                    return_partial_images=False,
                    dilate_mask_iterations=dilate_mask_iterations,
                )
                print(f"Astrometry : {astrometry}")
                # display overlay image
                # TODO: Done timestamp_string should be the current timestamp; Milan: Implemented current_timestamp() added to utils.py and updated code there
                # timestamp_string = "2023-09-06 10:00:00"
                timestamp_string = current_timestamp()
                omega = (0.0, 0.0, 0.0)
                display_overlay_info(contours_img, timestamp_string, astrometry, omega, scale_factors=scale_factors,
                                     resize_mode=resize_mode)

                for key in curr_params.keys():
                    curr_params[key].append(astrometry[key])

        # Compute mean of current calibration params
        for key, values in curr_params.items():
            if values:
                mean = sum(values) / len(values)
                curr_best_params[key] = mean

        # compute mean RMSE using new params
        set_RMSE = []
        for filename in os.listdir(calibration_images_dir):
            # calibration image path
            image_path = os.path.join(calibration_images_dir, filename)
            # do astrometry
            # TODO: The new version of do_astrometry from DEVELOPER has several new params that were not accounted for here
            astrometry, _, _ = self.do_astrometry(
                img=image_path,
                is_array=False,
                is_trail=is_trail,
                use_photutils=False,  # TODO: NEW
                subtract_global_bkg=False,  # TODO: NEW
                fast=False,  # TODO: NEW
                number_sources=number_sources,
                bkg_threshold=bkg_threshold,
                min_size=min_size,
                max_size=max_size,
                distortion_calibration_params=curr_best_params,
                log_file_path=log_file_path,
                min_pattern_checking_stars=min_pattern_checking_stars,
                local_sigma_cell_size=local_sigma_cell_size,
                sigma_clipped_sigma=sigma_clipped_sigma,
                leveling_filter_downscale_factor=leveling_filter_downscale_factor,
                src_kernal_size_x=src_kernal_size_x,
                src_kernal_size_y=src_kernal_size_y,
                src_sigma_x=src_sigma_x,
                src_dst=src_dst,
                return_partial_images=False,
                dilate_mask_iterations=dilate_mask_iterations,
            )
            set_RMSE.append(astrometry["RMSE"])
        curr_best_params["RMSE"] = sum(set_RMSE) / len(set_RMSE)

        # Assess the RMSE for the new calibration parameters in comparison to the previous ones
        if not update_calibration and prev_best_params != {}:
            if curr_best_params['RMSE'] > prev_best_params['RMSE']:
                return prev_best_params

        # write new calibration params to file
        with open(calibration_params_path, "w") as file:
            # Iterate over the dictionary and write each key-value pair
            for key, values in curr_params.items():
                # Convert the list of values to a string
                values_str = ', '.join(str(value) for value in values)
                # Write the key-value pair to the file
                file.write(f"{key}: {values_str}\n")

        # write best calibration params to file
        with open(calibration_params_path.replace(".txt", "_best.txt"), "w") as file:
            # Iterate over the dictionary and write each key-value pair
            for key, value in curr_best_params.items():
                # Write the key-value pair to the file
                file.write(f"{key}: {value}\n")

        return curr_best_params

    # TODO: Windell: The variables with hard coded values here should be put in the config file.
    # TODO:   Milan: All params when using do_astrometry from pueo_star_camera_operation_code.py pass all vars ...
    def do_astrometry(
            self,
            img,
            is_array=True,
            is_trail=True,
            use_photutils=False,
            subtract_global_bkg=False,
            fast=False,
            number_sources=20,
            bkg_threshold=3.1,
            min_size=4,
            max_size=200,
            distortion_calibration_params=None,
            log_file_path="log/test_log.txt",
            min_pattern_checking_stars=15,
            local_sigma_cell_size=36,
            sigma_clipped_sigma=3.0,
            leveling_filter_downscale_factor=4,
            src_kernal_size_x=3,
            src_kernal_size_y=3,
            src_sigma_x=1,
            src_dst=1,
            dilate_mask_iterations=1,
            return_partial_images=False,
            partial_results_path="./partial_results",
            solver='solver1'
    ):
        """Perform astrometry on an input image to determine celestial coordinates.

        This function processes an image to identify celestial sources and computes their centroids,
        performing astrometric analysis to find the direction in the sky. It utilizes background estimation
        techniques, source detection, and centroid calculation. If distortion calibration parameters are provided,
        they will be applied during the solving process.

        Args:
            img (np.ndarray or str): The input image as a 2D array (if `is_array` is True) or the
                file path to an image (if `is_array` is False).
            is_array (bool, optional): Flag indicating whether the image is provided as a 2D array (True)
                or as a file path (False). Default is True.
            is_trail (bool, optional): Flag indicating whether the image represents a star trail (True)
                or point sources (False). Default is True.
            use_photutils (bool, optional): Flag indicating whether to use photutils functions. Default is False.
            number_sources (int, optional): Maximum number of sources to detect in the image. Default is 20.
            bkg_threshold (float, optional): Background threshold for source detection. Default is 3.1.
            min_size (int, optional): Minimum size of detected sources (in pixels) to consider for calibration. Default is 4.
            max_size (int, optional): Maximum size of detected sources (in pixels) to consider for calibration. Default is 200.
            distortion_calibration_params (dict, optional): Calibration parameters for distortion, containing
                'FOV' and 'distortion' coefficients. Default is an empty dictionary.
            log_file_path (str, optional): Path to the log file for recording calibration logs.
                Defaults to "log/test_log.txt".
            min_pattern_checking_stars (int, optional): Number of stars used to create possible patterns to look up in database. Default is 15.
            local_sigma_cell_size (int, optional): The size of the cells (in pixels) used for estimating local noise
                levels. Defaults to 36.
            sigma_clipped_sigma (float, optional): The sigma value to use for sigma-clipping when calculating the
                background statistics (mean, median, standard deviation). It controls how aggressive the clipping is.
                Higher values mean less clipping (more tolerance for outliers). Defaults to 3.0.
            leveling_filter_downscale_factor (int, optional): The downscaling factor to apply when creating the
                downsampled image used for local level estimation. Defaults to 4.
            src_kernal_size_x (int, optional): The width of the Gaussian kernel used for smoothing the image. Defaults to 3.
            src_kernal_size_y (int, optional): The height of the Gaussian kernel used for smoothing the image. Defaults to 3.
            src_sigma_x (float, optional): The standard deviation in the X direction for the Gaussian kernel. Defaults to 1.
            src_dst (int, optional): The depth of the output image. Defaults to 1.
            dilate_mask_iterations (int, optional): The number of iterations for dilating the mask to merge nearby
                sources. A higher value merges more pixels. Defaults to 1.
            return_partial_images (bool, optional): Whether to return intermediate partial images for debugging.
                Defaults to False.
            partial_results_path (str, optional): The directory path where intermediate results will be saved
                if `return_partial_images` is True. Defaults to "./partial_results/".
            solver (str, optional): solver1|solver2: genuine or cedar solver.
        Returns:
            tuple: A tuple containing:
                - astrometry (dict): A dictionary with astrometric solutions including matched centroids and
                  additional details such as RA, Dec, Roll, etc.
                - precomputed_star_centroids (np.ndarray): An array of computed centroids for detected sources.
                - contours_img (np.ndarray): An image with drawn contours for detected sources.
        """

        def get_params():
            return {
                'is_array': is_array,
                'is_trail': is_trail,
                'use_photutils': use_photutils,
                'subtract_global_bkg': subtract_global_bkg,
                'fast': fast,
                'number_sources': number_sources,
                'bkg_threshold': bkg_threshold,
                'min_size': min_size,
                'max_size': max_size,
                'distortion_calibration_params': distortion_calibration_params,
                'log_file_path': log_file_path,
                'min_pattern_checking_stars': min_pattern_checking_stars,
                'local_sigma_cell_size': local_sigma_cell_size,
                'sigma_clipped_sigma': sigma_clipped_sigma,
                'leveling_filter_downscale_factor': leveling_filter_downscale_factor,
                'src_kernal_size_x': src_kernal_size_x,
                'src_kernal_size_y': src_kernal_size_y,
                'src_sigma_x': src_sigma_x,
                'src_dst': src_dst,
                'dilate_mask_iterations': dilate_mask_iterations,
                'return_partial_images': return_partial_images,
                'partial_results_path': partial_results_path,
                'cedar': cedar_data,
                'solver': solver
            }

        t0 = time.monotonic()
        # Capture local params, but remove image
        cedar_data = {}

        resize_factor = 1.0

        self.solver = solver
        print(f'solver: {self.solver}')
        if distortion_calibration_params is None:
            distortion_calibration_params = {}
        # read image in array format. From camera
        if is_array:
            # Scale the 16-bit values to 8-bit (0-255) range
            # The scale_factor = 2**14 - 1 = 16383.0
            scale_factor = float(2 ** self.cfg.pixel_well_depth) - 1
            scaled_data = ((img / scale_factor) * 255).astype(np.uint8)
            # scaled_data = img
            # img = scaled_data
            # Create a BGR image
            img_bgr = cv2.merge([scaled_data, scaled_data, scaled_data])
            # img_bgr = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # read image from file. For debugging using png/tif
        if not is_array:
            print(f'Reading Image: {img}')
            # img_bgr = read_image_BGR(img)
            img = read_image_grayscale(img)

            img_bgr = cv2.merge([img, img, img])

        print_img_info(img)
        print_img_info(img_bgr, 'bgr')

        # save image. For debugging
        if return_partial_images:
            # create partial results folder. For debugging
            if not os.path.exists(partial_results_path):
                os.makedirs(partial_results_path)
            # save input image
            cv2.imwrite(os.path.join(partial_results_path, "0 - Input Image-bgr.png"), img_bgr)
            cv2.imwrite(os.path.join(partial_results_path, "0 - Input Image-img.png"), img)

        # Source finder
        print(f"--------Subtract background--------")
        # global background estimation
        total_exec_time = 0.0
        if subtract_global_bkg:
            global_cleaned_img, global_exec_time = timed_function(
                global_background_estimation, img, sigma_clipped_sigma, return_partial_images, partial_results_path
            )
            img = global_cleaned_img
            total_exec_time += global_exec_time

        # This is False in config.ini as ast_use_photoutils param
        # TODO: This is forced for now until cedar detect is implemented
        if self.solver == 'solver1': #  or True:
            if use_photutils:
                # Local background estimation
                (cleaned_img, background_img), local_exec_time = timed_function(
                    # median_background_estimation,
                    sextractor_background_estimation,
                    img,
                    return_partial_images=return_partial_images,
                    partial_results_path=partial_results_path,
                )
                # Find sources
                (masked_image, segment_map), find_sources_exec_time = timed_function(
                    find_sources_photutils, img, background_img
                )
                if segment_map is None:
                    return None, None, img_bgr  # astrometry, precomputed_star_centroids, contours_img
                # Select top sources
                (masked_image, sources_mask, sources_contours), top_sources_exec_time = timed_function(
                    select_top_sources_photutils,
                    img,
                    masked_image,
                    segment_map,
                    number_sources=number_sources,
                    return_partial_images=return_partial_images,
                    partial_results_path=partial_results_path,
                )
                # Compute centroids
                (precomputed_star_centroids, contours_img), centroids_exec_time = timed_function(
                    compute_centroids_photutils, cleaned_img, img_bgr, segment_map, number_sources
                )
                total_exec_time += (
                        float(local_exec_time)
                        + float(find_sources_exec_time)
                        + float(top_sources_exec_time)
                        + float(centroids_exec_time)
                )
            else:
                ##### Uncomment the following lines to use the source-finding functions independently
                # (cleaned_img, background_img), local_exec_time = timed_function(local_levels_background_estimation, img, log_file_path, leveling_filter_downscale_factor, return_partial_images, partial_results_path)    # cleaned_img, background_img = sextractor_background_estimation(img, return_partial_images)
                # (masked_image, estimated_noise), find_sources_exec_time = timed_function(find_sources, img, background_img,fast, bkg_threshold, local_sigma_cell_size,
                #                            src_kernal_size_x, src_kernal_size_y, src_sigma_x, src_dst, return_partial_images, partial_results_path)
                # (masked_image, sources_mask, sources_contours), top_sources_exec_time = timed_function(select_top_sources, img, masked_image, estimated_noise, fast, number_sources=number_sources,
                #                                                             min_size=min_size, max_size=max_size,
                #                                                             dilate_mask_iterations=dilate_mask_iterations,
                #                                                             return_partial_images=return_partial_images, partial_results_path=partial_results_path)
                # source_finder_exec_time = int(local_exec_time) + int(find_sources_exec_time) + int(top_sources_exec_time)
                #####

                # source finder pipeline
                (masked_image, sources_mask, sources_contours), source_finder_exec_time = timed_function(
                    source_finder,
                    img,
                    log_file_path,
                    leveling_filter_downscale_factor,
                    fast,
                    bkg_threshold,
                    local_sigma_cell_size,
                    src_kernal_size_x,
                    src_kernal_size_y,
                    src_sigma_x,
                    src_dst,
                    number_sources,
                    min_size,
                    max_size,
                    dilate_mask_iterations,
                    is_trail,
                    return_partial_images,
                    partial_results_path,
                )

                # Compute source centroids
                # ast_is_trail = False in config.ini
                if is_trail:
                    # Highly experimental
                    (precomputed_star_centroids, contours_img), centroids_exec_time = timed_function(
                        compute_centroids_from_trail,
                        masked_image,
                        sources_mask,
                        img=img_bgr,
                        log_file_path=log_file_path,
                        return_partial_images=return_partial_images,
                        partial_results_path=partial_results_path,
                    )
                else:
                    # Will be using this
                    (precomputed_star_centroids, contours_img), centroids_exec_time = timed_function(
                        compute_centroids_from_still,
                        masked_image,
                        sources_contours,
                        img=img_bgr,
                        log_file_path=log_file_path,
                        return_partial_images=return_partial_images,
                        partial_results_path=partial_results_path,
                    )
                total_exec_time += (
                        float(source_finder_exec_time)
                        + float(centroids_exec_time)
                )
        elif self.solver == 'solver2':
            solver2_t0 = time.monotonic()
            contours_img = img_bgr
            self.cedar.tetra3_solver = self.tetra3_solver
            self.cedar.solver_name = self.solver_name
            self.cedar.test = self.test
            self.cedar.test_data = self.test_data
            self.cedar.return_partial_images = return_partial_images
            self.cedar.partial_results_path = partial_results_path
            self.cedar.distortion_calibration_params = distortion_calibration_params
            self.cedar.min_pattern_checking_stars = min_pattern_checking_stars
            precomputed_star_centroids, cedar_star_centroids, resize_factor, astrometry, tetra3_exec_time = self.cedar.get_centroids(img_bgr, img)
            total_exec_time += (time.monotonic() - solver2_t0)
        # Solve image using precomputed centroids
        # TODO: ERROR> precomputed_star_centroids != []:
        # Comparing variable of type np.ndarray to an empty list does not really work!!!
        # To check if list is not empty you can do:
        #   1. if array.size:
        #   2. if array.any():

        # if precomputed_star_centroids != []:
        # Milan: Check if the array is not empty effectively
        if isinstance(precomputed_star_centroids, np.ndarray) and precomputed_star_centroids.shape[0]:
            print("--------Get direction in the sky using tetra3--------")
            print(
                f'Image: {img.shape[0]}x{img.shape[1]} {img.dtype} {img.size} distortion: {distortion_calibration_params}')

            max_c = 0 or precomputed_star_centroids.shape[0]
            if distortion_calibration_params and solver == 'solver1':
                astrometry, tetra3_exec_time = timed_function(
                    self.tetra3_solver,
                    img,
                    precomputed_star_centroids[:max_c, :2],
                    FOV=distortion_calibration_params["FOV"],
                    distortion=distortion_calibration_params["distortion"],
                    min_pattern_checking_stars=min_pattern_checking_stars,
                    resize_factor=resize_factor
                )
            else:
                # Astrometry already created as part of cedar_detect (self.cedar.get_centroids)
                pass

            # Add execution time
            astrometry["precomputed_star_centroids"] = precomputed_star_centroids.shape[0]
            astrometry['cedar_detect'] = self.cedar.cd_solutions.copy()
            astrometry["params"] = get_params()
            astrometry["tetra3_exec_time"] = tetra3_exec_time
            print("Astrometry: " + str(astrometry))

            # Draw valid matched star contours (Green)
            color_green = (0, 255, 0)  # Green
            color_red = (0, 0, 255)  # Red
            color_blue = (255, 0, 0)  # Blue
            color_yellow = (0, 255, 255)  # BGR Yellow (Blue=0, Green=255, Red=255)

            def draw_centroids(centroids, color, radius_div=80):
                """
                radius_div:
                    160 - detected stars
                    120 - candidates stars
                    80 - confirmed stars - matched centroids
                """
                sources_radius = img.shape[0] / radius_div
                # Centroids from astrometry solution are a list, candidates are an np.ndarray
                centroids = centroids[:, :2] if isinstance(centroids, np.ndarray) else centroids
                for idx, (y, x) in enumerate(centroids):
                    # draw contour
                    cv2.circle(contours_img, (int(x * resize_factor), int(y * resize_factor)), int(sources_radius),
                               color, 4)
                    if idx < 5 and radius_div == 160:
                        cv2.circle(contours_img, (int(x * resize_factor), int(y * resize_factor)), int(sources_radius),
                                   color_yellow, 4)

                    # cv2.circle(contours_img, (int(x), int(y)), 1, color_green, -1)

            if self.solver == 'solver2':
                # Draw cedar-detected centroids
                if cedar_star_centroids.shape[0]:
                    draw_centroids(cedar_star_centroids, color_red, 160)

                # Draw cedar-detected centroids filtered
                if precomputed_star_centroids.shape[0]:
                    draw_centroids(precomputed_star_centroids, color_blue, 120)

            # Draw cedar-detected centroids
            if astrometry.get('FOV') is not None:
                draw_centroids(astrometry['matched_centroids'], color_green, 80)

        else:
            astrometry = {}

        total_exec_time = time.monotonic() - t0
        astrometry["total_exec_time"] = total_exec_time if astrometry else None
        print(f'  do_astrometry completed in {total_exec_time:.2f}s.')
        astrometry['solver'] = self.solver_name if astrometry else None
        return astrometry, precomputed_star_centroids, contours_img

    def test_astrometry(self, img_path, display=False):
        """Do astrometry on existing files.

        Set display=False to only save output images, and no display.
        """
        # params
        cfg = Config()
        cfg.ast_is_array = False

        use_photutils = False
        default_best_params = {"FOV": cfg.lab_fov,
                               "distortion": [cfg.lab_distortion_coefficient_1, cfg.lab_distortion_coefficient_2]}
        fast = False
        subtract_global_bkg = False
        log_file_path = "log/"
        partial_results_path = "./partial_results/"

        # image specific paths
        img = img_path
        if "/" in img:
            img_name = img_path.split("/")[-1].split(".")[0]
        else:
            img_name = img_path.split(".")[0]
        img_log_file_path = f"{os.path.join(log_file_path, 'log_' + img_name)}.txt"
        img_partial_results_path = os.path.join(partial_results_path, img_name)
        # create partial results dir
        if not os.path.exists(img_partial_results_path):
            os.makedirs(img_partial_results_path)

        # perform astrometry
        astrometry, curr_star_centroids, contours_img = self.do_astrometry(
            img,
            cfg.ast_is_array,
            cfg.ast_is_trail,
            use_photutils,
            subtract_global_bkg,
            fast,
            cfg.img_number_sources,
            cfg.img_bkg_threshold,
            cfg.img_min_size,
            cfg.img_max_size,
            default_best_params,
            img_log_file_path,
            cfg.min_pattern_checking_stars,
            cfg.local_sigma_cell_size,
            cfg.sigma_clipped_sigma,
            cfg.leveling_filter_downscale_factor,
            cfg.src_kernal_size_x,
            cfg.src_kernal_size_y,
            cfg.src_sigma_x,
            return_partial_images=cfg.return_partial_images,
            partial_results_path=img_partial_results_path,
            solver='solver2'
        )
        # display overlay image
        timestamp_string = current_timestamp("%d-%m-%Y-%H-%M-%S")
        omega = (0.0, 0.0, 0.0)
        # display_overlay_info(img, timestamp_string, astrometry, omega, display=True, image_filename=None, downscale_factors=(8, 8)):
        display_overlay_info(
            contours_img,
            timestamp_string,
            astrometry,
            omega,
            display=display,
            image_filename=img_name,
            partial_results_path=img_partial_results_path,
            scale_factors=cfg.scale_factors,
            resize_mode=cfg.resize_mode
        )

    def run_tests(self, dir_path="./test_images", img_path=None, display=False):
        """Run astrometry on a single image or all images in a specified directory."""
        # if no image path is given run tests on all images in a folder
        if img_path is None:
            for file in os.listdir(dir_path):
                if file.lower().endswith((".jpg", ".jpeg", ".png", ".tif", ".tiff")):
                    # image specific paths
                    img_path = os.path.join(dir_path, file)
                    print("#####################################################")
                    print(f"Processing : {img_path}")
                    print("#####################################################")
                    self.test_astrometry(img_path, display)
        else:
            if img_path.lower().endswith((".jpg", ".jpeg", ".png", ".tif", ".tiff")):
                # image specific paths
                self.test_astrometry(img_path, display)


# RUN LOCAL TESTS
if __name__ == "__main__":
    astrometry = Astrometry(database_name=None)
    # run tests
    # run_tests(img_path="./test_images/cloudy_20240422.png", display=True)
    if False:
        astrometry.run_tests(dir_path="../test_images", display=False)

    # Generate optimal calibration params from a set of images
    # optimize_calibration_parameters(is_trail,number_sources, min_size, max_size, calibration_images_dir="../data/calibration_images",log_file_path="log/calibration_log.txt", calibration_params_path="calibration_params.txt", update_calibration=True)

    # Generate database
    # tetra3_generate_database(star_catalog="bsc5", max_fov=14, star_max_magnitude=7, output_name="fov_13_5_bsc5_database.npz")

    # The ‘BSC5’ data is available from <http://tdc-www.harvard.edu/catalogs/bsc5.html> (use byte format file) and
    # ‘hip_main’ and ‘tyc_main’ are available from <https://cdsarc.u-strasbg.fr/ftp/cats/I/239/> (save the appropriate .dat file).
    # The downloaded catalogue must be placed in the tetra3 directory.
    # Windel database
    print('Creating tetra3 Database: ')
    t0 = time.monotonic()
    astrometry.t3_cedar.generate_database(
        max_fov=10.79, ## 2.58,
        # min_fov=2.1,
        save_as="fov_10.79_mag9.0_tyc_v11_cedar_database.npz",
        star_catalog="tyc_main",  # tyc_main
        # pattern_stars_per_fov=10,  # Removed for test, Default: 150
        # verification_stars_per_fov=20, # was 20
        star_max_magnitude=9,
        pattern_max_error=0.005,
        # star_min_separation=0.01,  # (this is in degrees)
        # pattern_max_error=0.005,
        # simplify_pattern=False,
        # range_ra=None,
        # range_dec=None,
        # presort_patterns=True,
        # save_largest_edge=False,
        multiscale_step=1.5
    )
    from common import get_dt

    print(f'Completed in {get_dt(t0)}.')
