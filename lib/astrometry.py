# Standard imports
from contextlib import suppress
import logging
import traceback
import sys
import time
from time import perf_counter as precision_timestamp
import os
from pathlib import Path
# External imports
import matplotlib.pyplot as plt
import cv2
import numpy as np
from scipy.optimize import linear_sum_assignment
from skimage.color.rgb_colors import silver

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
from lib.astrometry_net import  AstrometryNet

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
        if self.solver == 'solver1':
            return 'ESA Tetra3'
        elif self.solver == 'solver2':
            return 'Cedar Tetra3'
        elif self.solver == 'solver3':
            return 'astrometry.net'
        else:
            return 'Undefined solver'

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
            pattern_checking_stars=pattern_checking_stars,
            # match_radius=0.01,  # Default 0.01
            # match_threshold=1e-3,
            solve_timeout=self.cfg.solve_timeout, # 5000.0, # Default None milliseconds
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
            resize_mode='downscale',
            level_filter: int = 9,
            ring_filter_type = 'mean'
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
            level_filter (int, optional): level_filter size
            ring_filter_type (str, optional):  Source Ring Background Estimation Type: mean|median

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
                    level_filter=level_filter,
                    ring_filter_type=ring_filter_type
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
                level_filter=level_filter,
                ring_filter_type=ring_filter_type
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

    # TODO: Remove remove_unmatched
    def remove_unmatched(self, precomputed_star_centroids, matched_centroids, epsilon=10.0):
        """
        Filter and reorder using Hungarian algorithm for optimal assignment.
        """
        if precomputed_star_centroids.size == 0 or not matched_centroids:
            return []

        matched_arr = np.array(matched_centroids)
        precomputed_yx = precomputed_star_centroids[:, :2]

        # Calculate distance matrix
        diff = precomputed_yx[:, np.newaxis, :] - matched_arr[np.newaxis, :, :]
        distances = np.sqrt(np.sum(diff ** 2, axis=2))

        # Use Hungarian algorithm for optimal assignment
        row_ind, col_ind = linear_sum_assignment(distances)

        result = [None] * len(matched_arr)

        for i, j in zip(row_ind, col_ind):
            if distances[i, j] <= epsilon:
                result[j] = precomputed_star_centroids[i].tolist()

        return result

    # TODO: Remove calculate_rms_errors_from_centroids_old
    def calculate_rms_errors_from_centroids_old(self,
                                            precomputed_star_centroids,
                                            result,
                                            image_size,
                                            plate_scale_arcsec_per_pix=None
                                            ):
        """
        Compute RA, Dec, and Roll RMS directly from tetra3 pixel-space outputs.

        Assumptions / Inputs (from tetra3.solve_from_centroids(..., return_matches=True)):
          - result['matched_centroids']     : (N,2) measured (y, x) pixels
          - result['matched_predicted_xy']  : (N,2) predicted (y, x) pixels (tetra3's image-plane match)
          - result['Roll']                  : roll [deg], image axes vs local sky tangent-plane (east/north)
          - Optional: result['FOV']         : width field of view [deg] (used if plate_scale not given)
        image_size : (H, W)                 : original image shape in pixels
        plate_scale_arcsec_per_pix : float  : arcsec per pixel; if None, estimated as (FOV_deg * 3600) / W

        Returns:
          dict with:
            - RA_RMS_arcsec, Dec_RMS_arcsec, Roll_RMS_arcsec
            # - Pixel_RMS_x, Pixel_RMS_y
            # - plate_scale_arcsec_per_pix
            # - (optional) per_star: dx_pix, dy_pix, ra_err_arcsec, dec_err_arcsec, roll_err_arcsec

        Notes:
          • Rotation by Roll decomposes pixel residuals into local RA/Dec directions (tangent-plane).
          • Roll RMS uses small-rotation model: α_i ≈ (r_perp · e) / |r|^2 about image center.
        """
        H, W = image_size

        # Plate scale
        if plate_scale_arcsec_per_pix is None:
            if 'FOV' not in result:
                raise ValueError("Provide plate_scale_arcsec_per_pix or include result['FOV'].")
            plate_scale_arcsec_per_pix = (float(result['FOV']) * 3600.0) / float(W)

        # Measured and predicted in (y, x)
        matched_precomputed_star_centroids = self.remove_unmatched(precomputed_star_centroids, result['matched_centroids'], epsilon=2.0)
        meas_yx = np.asarray(matched_precomputed_star_centroids, dtype=float)[:, :2]
        pred_yx = np.asarray(result['matched_centroids'], dtype=float) # Our precomputed_star_centroids

        if True:
            self.log.debug(f'precomputed_star_centroids: {precomputed_star_centroids}')
            self.log.debug(f'meas_yx: {meas_yx}')
            self.log.debug(f'pred_yx: {pred_yx}')

        if meas_yx.shape != pred_yx.shape:
            raise ValueError("matched_centroids and matched_predicted_xy shapes differ.")

        # Pixel residuals (y down, x right)
        dy = meas_yx[:, 0] - pred_yx[:, 0]
        dx = meas_yx[:, 1] - pred_yx[:, 1]

        # Rotate residuals by tetra3 Roll so x' ~ RA (east), y' ~ Dec (north)
        R = np.deg2rad(float(result['Roll']))
        c, s = np.cos(R), np.sin(R)
        ex_ra = c * dx + s * dy
        ey_dec = -s * dx + c * dy

        # Convert to arcsec (signs irrelevant for RMS)
        ra_err_arcsec = ex_ra * plate_scale_arcsec_per_pix
        dec_err_arcsec = (-ey_dec) * plate_scale_arcsec_per_pix

        # Pixel RMS (debug)
        if False:
            pixel_rms_x = float(np.sqrt(np.mean(dx ** 2)))
            pixel_rms_y = float(np.sqrt(np.mean(dy ** 2)))

        # Roll RMS from small-rotation model about image center
        cx, cy = W / 2.0, H / 2.0
        rx = pred_yx[:, 1] - cx  # x offset (pixels)
        ry = pred_yx[:, 0] - cy  # y offset (pixels)
        r2 = rx * rx + ry * ry
        mask = r2 > 1e-9
        # α_i ≈ (r_perp · e)/|r|^2, with r_perp = [-ry, rx]
        r_perp_dot_e = (-ry[mask]) * dx[mask] + (rx[mask]) * dy[mask]
        alpha_rad_i = np.zeros_like(dx)
        alpha_rad_i[mask] = r_perp_dot_e / r2[mask]
        roll_err_arcsec = alpha_rad_i * (180 / np.pi) * 3600.0

        # RMS values
        RA_RMS_arcsec = float(np.sqrt(np.mean(ra_err_arcsec ** 2)))
        Dec_RMS_arcsec = float(np.sqrt(np.mean(dec_err_arcsec ** 2)))
        Roll_RMS_arcsec = float(np.sqrt(np.mean(roll_err_arcsec ** 2)))

        out = {
            "RA_RMS": RA_RMS_arcsec,
            "Dec_RMS": Dec_RMS_arcsec,
            "Roll_RMS": Roll_RMS_arcsec,
            # "Pixel_RMS_x": pixel_rms_x,
            # "Pixel_RMS_y": pixel_rms_y,
            # "plate_scale_arcsec_per_pix": float(plate_scale_arcsec_per_pix),
        }

        return out

    def calculate_rms_errors_from_centroids_chatgpt_windell(self,
                                            result,
                                            image_size
                                            ):
        """
        Compute RA, Dec, and Roll RMS (arcseconds) from tetra3 outputs.

        Expects (from tetra3.solve_from_centroids(..., return_matches=True)):
          - result['matched_centroids'] : (N,2) matched image centroids in (y, x) pixels
          - result['matched_stars']     : (N,2 or N,3) matched (RA_deg, Dec_deg [, mag])
          - result['RA'], result['Dec'] : field center (deg)
          - result['Roll']              : roll (deg), image axes vs local East/North
          - result['FOV']               : horizontal field of view (deg)
          - result.get('distortion')    : optional single scalar (neg=barrel, pos=pincushion),
                                          defined at radius = image width / 2

        image_size : (H, W) in pixels

        Returns:
          {
            "RA_RMS":   <arcsec>,
            "Dec_RMS":  <arcsec>,
            "Roll_RMS": <arcsec>
          }
        """
        ARCSEC_PER_RAD = 206264.80624709636

        # ---------- helpers ----------
        def _gnomonic_project(ra, dec, ra0, dec0):
            """(ra,dec)->(xi,eta) on tangent plane at (ra0,dec0), radians."""
            dra = ra - ra0
            dra = (dra + np.pi) % (2 * np.pi) - np.pi  # wrap to [-pi, pi]
            sd, cd = np.sin(dec), np.cos(dec)
            s0, c0 = np.sin(dec0), np.cos(dec0)
            denom = s0 * sd + c0 * cd * np.cos(dra)
            xi = (cd * np.sin(dra)) / denom
            eta = (c0 * sd - s0 * cd * np.cos(dra)) / denom
            return xi, eta

        def _best_fit_theta(xi_ref, eta_ref, xi_meas, eta_meas):
            """Small in-plane rotation theta (radians) mapping ref->meas after mean-centering."""
            X = np.stack([xi_ref, eta_ref], axis=1)
            Y = np.stack([xi_meas, eta_meas], axis=1)
            Xc = X - X.mean(axis=0, keepdims=True)
            Yc = Y - Y.mean(axis=0, keepdims=True)
            H = Xc.T @ Yc
            return np.arctan2(H[0, 1] - H[1, 0], H[0, 0] + H[1, 1])

        def _rotate(xi, eta, theta):
            c, s = np.cos(theta), np.sin(theta)
            return c * xi - s * eta, s * xi + c * eta

        def _pixels_to_plane_fov(x, y, W, H, fovx_rad):
            """
            Pinhole (no distortion): normalize by half-size and scale by tan(FOV/2).
            Uses horizontal FOV; vertical FOV is inferred via aspect ratio (square pixels).
            y is flipped to make +eta point North (up).
            """
            cx, cy = (W - 1) / 2.0, (H - 1) / 2.0
            nx = (x - cx) / (W / 2.0)  # [-1,1]
            ny = (y - cy) / (H / 2.0)  # [-1,1]
            ny = -ny  # flip so +eta is up/North
            xi = np.tan(0.5 * fovx_rad) * nx
            # infer vertical FOV from aspect ratio: FOVy = FOVx * (H/W)
            fovy_rad = fovx_rad * (H / float(W))
            eta = np.tan(0.5 * fovy_rad) * ny
            return xi, eta

        def _undistort_oneparam_xy(x, y, W, H, d):
            """
            Invert tetra3's single-parameter radial distortion.
            Model (normalized by R = W/2): r_d = r_u * (1 + d * r_u^2)
            Given distorted pixels (x,y), solve for undistorted radius r_u via Newton, then rescale.
            """
            if d is None or abs(d) < 1e-12:
                return x, y  # no meaningful distortion

            cx, cy = (W - 1) / 2.0, (H - 1) / 2.0
            R = W / 2.0  # reference radius per tetra3 convention
            rx = x - cx
            ry = y - cy
            r = np.sqrt(rx * rx + ry * ry)

            outx, outy = x.copy(), y.copy()
            nz = r > 1e-12
            if not np.any(nz):
                return outx, outy

            r_d_norm = (r[nz] / R)

            # Newton iterations to invert: u + d*u^3 = r_d_norm  (solve for u)
            u = r_d_norm.copy()
            for _ in range(5):
                f = u * (1 + d * u * u) - r_d_norm
                fp = 1 + 3 * d * u * u
                u -= f / fp

            scale = np.ones_like(r)
            scale[nz] = (u / r_d_norm)  # r_u / r_d
            outx[nz] = cx + scale[nz] * rx[nz]
            outy[nz] = cy + scale[nz] * ry[nz]
            return outx, outy

        # ---------- parse inputs ----------
        H, W = image_size
        if 'matched_centroids' not in result or 'matched_stars' not in result:
            raise ValueError(
                "result must contain 'matched_centroids' and 'matched_stars' (tetra3 return_matches=True).")

        # Matched centroids: (y,x) -> (x,y)
        yx = np.asarray(result['matched_centroids'], dtype=float)
        xy = yx[:, [1, 0]]
        if xy.shape[0] < 2:
            return {"RA_RMS": np.nan, "Dec_RMS": np.nan, "Roll_RMS": np.nan}

        # Matched stars: RA/Dec in degrees (ignore magnitude if present)
        ms = np.asarray(result['matched_stars'], dtype=float)
        radec_deg = ms[:, :2]

        # Center & roll (deg -> rad)
        ra0_deg = float(result.get('RA', result.get('RA0')))
        dec0_deg = float(result.get('Dec', result.get('Dec0')))
        roll_deg = float(result['Roll'])
        ra0, dec0, roll = np.deg2rad([ra0_deg, dec0_deg, roll_deg])

        # Horizontal FOV only (deg) from tetra3; vertical inferred by aspect
        if 'FOV' not in result:
            raise ValueError("result must contain 'FOV' (horizontal field of view, degrees).")
        fovx_rad = np.deg2rad(float(result['FOV']))

        # Optional single-parameter distortion (neg=barrel, pos=pincushion)
        d = result.get('distortion', None)
        d = None if d is None else float(d)

        # ---------- pixels -> (undistorted) plane ----------
        x = xy[:, 0].copy()
        y = xy[:, 1].copy()
        if d is not None:
            x, y = _undistort_oneparam_xy(x, y, W, H, d)

        # Map to tangent plane using FOV
        xi_m, eta_m = _pixels_to_plane_fov(x, y, W, H, fovx_rad)

        # Un-roll by estimated roll so axes are East/North
        cR, sR = np.cos(-roll), np.sin(-roll)
        xi_m, eta_m = cR * xi_m - sR * eta_m, sR * xi_m + cR * eta_m

        # ---------- catalog -> plane ----------
        ra_rad = np.deg2rad(radec_deg[:, 0])
        dec_rad = np.deg2rad(radec_deg[:, 1])
        xi_c, eta_c = _gnomonic_project(ra_rad, dec_rad, ra0, dec0)

        # Best-fit tiny residual in-plane rotation (residual roll)
        theta = _best_fit_theta(xi_c, eta_c, xi_m, eta_m)

        # Plane residuals
        xi_c_rot, eta_c_rot = _rotate(xi_c, eta_c, theta)
        dxi = xi_m - xi_c_rot
        deta = eta_m - eta_c_rot

        # Plane -> RA/Dec residuals (small-angle)
        cosd0 = np.cos(dec0)
        dra_rad = dxi / cosd0
        ddec_rad = deta

        # RMS (arcseconds)
        RA_RMS_arcsec = float(np.sqrt(np.mean(dra_rad ** 2)) * ARCSEC_PER_RAD)
        Dec_RMS_arcsec = float(np.sqrt(np.mean(ddec_rad ** 2)) * ARCSEC_PER_RAD)

        # Roll RMS: local rotational component about boresight
        v2 = xi_c_rot ** 2 + eta_c_rot ** 2
        mask = v2 > 1e-16
        rot_local = np.zeros_like(v2)
        rot_local[mask] = (xi_c_rot[mask] * deta[mask] - eta_c_rot[mask] * dxi[mask]) / v2[mask]  # radians
        Roll_RMS_arcsec = float(np.sqrt(np.mean(rot_local[mask] ** 2)) * ARCSEC_PER_RAD) if np.any(mask) else np.nan

        return {"RA_RMS": RA_RMS_arcsec, "Dec_RMS": Dec_RMS_arcsec, "Roll_RMS": Roll_RMS_arcsec}

    def calculate_rms_errors_from_centroids(self, result, image_size):
        """
        Compute RA, Dec, and Roll RMS (arcseconds) from tetra3 outputs.

        Args:
            result: Dictionary from tetra3.solve_from_centroids(return_matches=True)
            image_size: (height, width) in pixels

        Returns:
            Dictionary with RA_RMS, Dec_RMS, Roll_RMS in arcseconds
        """
        # Constants
        ARCSEC_PER_RAD = 206264.80624709636

        def gnomonic_project(ra, dec, ra0, dec0):
            """(ra,dec)->(xi,eta) on tangent plane at (ra0,dec0), radians."""
            dra = ra - ra0
            dra = (dra + np.pi) % (2 * np.pi) - np.pi  # wrap to [-pi, pi]
            sin_dec, cos_dec = np.sin(dec), np.cos(dec)
            sin_dec0, cos_dec0 = np.sin(dec0), np.cos(dec0)
            denominator = sin_dec0 * sin_dec + cos_dec0 * cos_dec * np.cos(dra)
            xi = (cos_dec * np.sin(dra)) / denominator
            eta = (cos_dec0 * sin_dec - sin_dec0 * cos_dec * np.cos(dra)) / denominator
            return xi, eta

        def best_fit_rotation(xi_ref, eta_ref, xi_meas, eta_meas):
            """Find small in-plane rotation theta (radians) mapping ref->meas."""
            X = np.column_stack([xi_ref, eta_ref])
            Y = np.column_stack([xi_meas, eta_meas])
            X_centered = X - X.mean(axis=0)
            Y_centered = Y - Y.mean(axis=0)
            H = X_centered.T @ Y_centered
            return np.arctan2(H[0, 1] - H[1, 0], H[0, 0] + H[1, 1])

        def rotate_coords(xi, eta, theta):
            """Rotate coordinates by angle theta."""
            cos_theta, sin_theta = np.cos(theta), np.sin(theta)
            return cos_theta * xi - sin_theta * eta, sin_theta * xi + cos_theta * eta

        def pixels_to_plane(x, y, width, height, fov_x_rad):
            """Convert pixels to tangent plane coordinates."""
            center_x, center_y = (width - 1) / 2.0, (height - 1) / 2.0
            norm_x = (x - center_x) / (width / 2.0)  # [-1,1]
            norm_y = (y - center_y) / (height / 2.0)  # [-1,1]
            norm_y = -norm_y  # flip so +eta is up/North

            xi = np.tan(0.5 * fov_x_rad) * norm_x
            fov_y_rad = fov_x_rad * (height / float(width))
            eta = np.tan(0.5 * fov_y_rad) * norm_y
            return xi, eta

        def undistort_xy(x, y, width, height, distortion_param):
            """Apply radial distortion correction."""
            if distortion_param is None or abs(distortion_param) < 1e-12:
                return x, y

            center_x, center_y = (width - 1) / 2.0, (height - 1) / 2.0
            ref_radius = width / 2.0
            dx = x - center_x
            dy = y - center_y
            radius = np.sqrt(dx ** 2 + dy ** 2)

            result_x, result_y = x.copy(), y.copy()
            nonzero_mask = radius > 1e-12

            if not np.any(nonzero_mask):
                return result_x, result_y

            # Newton iterations to invert distortion
            r_distorted_norm = radius[nonzero_mask] / ref_radius
            u = r_distorted_norm.copy()

            for _ in range(5):
                f = u * (1 + distortion_param * u ** 2) - r_distorted_norm
                f_prime = 1 + 3 * distortion_param * u ** 2
                u -= f / f_prime

            scale = np.ones_like(radius)
            scale[nonzero_mask] = u / r_distorted_norm
            result_x[nonzero_mask] = center_x + scale[nonzero_mask] * dx[nonzero_mask]
            result_y[nonzero_mask] = center_y + scale[nonzero_mask] * dy[nonzero_mask]

            return result_x, result_y

        # Parse inputs
        height, width = image_size

        if 'matched_centroids' not in result or 'matched_stars' not in result:
            raise ValueError("Result must contain matched_centroids and matched_stars")

        # Convert centroids from (y,x) to (x,y)
        yx_coords = np.asarray(result['matched_centroids'], dtype=float)
        xy_coords = yx_coords[:, [1, 0]]

        if xy_coords.shape[0] < 2:
            return {"RA_RMS": np.nan, "Dec_RMS": np.nan, "Roll_RMS": np.nan}

        # Extract RA/Dec coordinates
        matched_stars = np.asarray(result['matched_stars'], dtype=float)
        radec_deg = matched_stars[:, :2]

        # Get field center and orientation
        ra0_deg = float(result.get('RA', result.get('RA0')))
        dec0_deg = float(result.get('Dec', result.get('Dec0')))
        roll_deg = float(result['Roll'])
        ra0, dec0, roll = np.deg2rad([ra0_deg, dec0_deg, roll_deg])

        # Field of view
        if 'FOV' not in result:
            raise ValueError("Result must contain FOV")
        fov_x_rad = np.deg2rad(float(result['FOV']))

        # Distortion parameter
        distortion = result.get('distortion')
        distortion = None if distortion is None else float(distortion)

        # Process pixel coordinates
        x_coords, y_coords = xy_coords[:, 0].copy(), xy_coords[:, 1].copy()

        if distortion is not None:
            x_coords, y_coords = undistort_xy(x_coords, y_coords, width, height, distortion)

        # Convert to tangent plane
        xi_measured, eta_measured = pixels_to_plane(x_coords, y_coords, width, height, fov_x_rad)

        # Remove estimated roll
        cos_roll, sin_roll = np.cos(-roll), np.sin(-roll)
        xi_measured = cos_roll * xi_measured - sin_roll * eta_measured
        eta_measured = sin_roll * xi_measured + cos_roll * eta_measured

        # Convert catalog coordinates to tangent plane
        ra_rad = np.deg2rad(radec_deg[:, 0])
        dec_rad = np.deg2rad(radec_deg[:, 1])
        xi_catalog, eta_catalog = gnomonic_project(ra_rad, dec_rad, ra0, dec0)

        # Find residual rotation and calculate errors
        residual_theta = best_fit_rotation(xi_catalog, eta_catalog, xi_measured, eta_measured)
        xi_catalog_rot, eta_catalog_rot = rotate_coords(xi_catalog, eta_catalog, residual_theta)

        # Calculate residuals
        dxi = xi_measured - xi_catalog_rot
        deta = eta_measured - eta_catalog_rot

        # Convert to RA/Dec errors
        cos_dec0 = np.cos(dec0)
        dra_rad = dxi / cos_dec0
        ddec_rad = deta

        # Calculate RMS values
        ra_rms = float(np.sqrt(np.mean(dra_rad ** 2)) * ARCSEC_PER_RAD)
        dec_rms = float(np.sqrt(np.mean(ddec_rad ** 2)) * ARCSEC_PER_RAD)

        # Calculate roll error
        radius_sq = xi_catalog_rot ** 2 + eta_catalog_rot ** 2
        valid_mask = radius_sq > 1e-16

        if np.any(valid_mask):
            local_rotation = (xi_catalog_rot[valid_mask] * deta[valid_mask] -
                              eta_catalog_rot[valid_mask] * dxi[valid_mask]) / radius_sq[valid_mask]
            roll_rms = float(np.sqrt(np.mean(local_rotation ** 2)) * ARCSEC_PER_RAD)
        else:
            roll_rms = np.nan

        return {
            "RA_RMS": ra_rms,
            "Dec_RMS": dec_rms,
            "Roll_RMS": roll_rms
        }

    # TODO: Windell: The variables with hard coded values here should be put in the config file.
    # TODO:   Milan: All params when using do_astrometry from pueo_star_camera_operation_code.py pass all vars ...

    def do_astrometry(self, *args, **kwargs):
        """Wrapper function that handles both direct calls and multiprocessing pool calls"""
        # TODO: Set to False for PRODUCTION
        test = True
        if test:
            print(f"DEBUG: args received: {args}")
            print(f"DEBUG: kwargs received: {kwargs}")

            # Get the original function's parameter names
            import inspect
            sig = inspect.signature(self._do_astrometry)
            param_names = list(sig.parameters.keys())[1:]  # Skip 'self'

            print(f"DEBUG: _do_astrometry expects parameters: {param_names}")
            # print(f"DEBUG: Number of args passed: {len(args)}")
            # print(f"DEBUG: Number of kwargs passed: {len(kwargs)}")

            print(f"\n{'-' * 60}")
            print(f"DO_ASTROMETRY PARAMETERS")
            print(f"{'-' * 60}")

            max_name_length = 32
            for idx, param_name in enumerate(param_names):
                param_value = args[idx+1] if len(args) > idx+1 else None
                if param_value is None:
                    formatted_value = "None"
                elif isinstance(param_value, str):
                    formatted_value = f"'{param_value}'"
                else:
                    formatted_value = str(param_value)
                print(f"{param_name:>{max_name_length}} : {formatted_value}")

        return self._do_astrometry(*args, **kwargs)

    def _print_parameters(self, func_name, parameters):
        """Helper method to print parameters"""
        if not parameters:
            return

        max_name_length = max(len(name) for name in parameters.keys())

        print(f"\n{'-' * 60}")
        print(f"{func_name.upper()} PARAMETERS")
        print(f"{'-' * 60}")

        max_name_length = 32
        for param_name, param_value in parameters.items():
            if param_value is None:
                formatted_value = "None"
            elif isinstance(param_value, str):
                formatted_value = f"'{param_value}'"
            else:
                formatted_value = str(param_value)

            print(f"{param_name:>{max_name_length}} : {formatted_value}")

        print(f"{'-' * 60}\n")


    def _do_astrometry(
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
            solver='solver1',
            level_filter: int = 9,
            ring_filter_type = 'mean'
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
            level_filter (int): The size of the star level filter, shall be 5..199 and an odd number.
            ring_filter_type (str): Source Ring Background Estimation Type: mean|median
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
        solver_exec_time = 0.0
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

        # Save Partial image. For debugging
        if return_partial_images:
            # Create partial results folder. For debugging
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
        if self.solver in ['solver1', 'solver3']: #  or True:
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
                    level_filter,
                    ring_filter_type
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

                # Save centroids

        if self.solver == 'solver2':
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
            precomputed_star_centroids, cedar_star_centroids, resize_factor, astrometry, solver_exec_time = self.cedar.get_centroids(img_bgr, img)
            total_exec_time += (time.monotonic() - solver2_t0)
        # Solve image using precomputed centroids
        # TODO: ERROR> precomputed_star_centroids != []:
        # Comparing variable of type np.ndarray to an empty list does not really work!!!
        # To check if list is not empty you can do:
        #   1. if array.size:
        #   2. if array.any():

        # if precomputed_star_centroids != []:
        # Milan: Check if the array is not empty effectively

        # Calculate and add RMS to astrometry result
        height, width = img.shape
        image_size = (int(height / resize_factor), int(width / resize_factor))
        if isinstance(precomputed_star_centroids, np.ndarray) and precomputed_star_centroids.shape[0]:
            print("--------Get direction in the sky using solver of tetra3/astrometry.net --------")
            print(
                f'Image: {img.shape[0]}x{img.shape[1]} {img.dtype} {img.size} distortion: {distortion_calibration_params}')

            max_c = 0 or precomputed_star_centroids.shape[0]
            if distortion_calibration_params and solver == 'solver1':
                astrometry, solver_exec_time = timed_function(
                    self.tetra3_solver,
                    img,
                    precomputed_star_centroids[:max_c, :2],
                    FOV=distortion_calibration_params["FOV"],
                    distortion=distortion_calibration_params["distortion"],
                    min_pattern_checking_stars=min_pattern_checking_stars,
                    resize_factor=resize_factor
                )
            elif solver == 'solver2':
                # Astrometry already created as part of cedar_detect (self.cedar.get_centroids)
                pass
            elif solver == 'solver3':
                # Process the centroids with configuration
                astrometry = {}
                an_solver = AstrometryNet(self.cfg, self.log)
                try:
                    astrometry, solver_exec_time = timed_function(
                        an_solver.process_centroids,
                        precomputed_star_centroids[:max_c],
                        image_size,
                        output_base="6.0 - astrometry.net-solve-field-centroids",
                        output_dir=self.cfg.partial_results_path  # "./astrometry_results"
                    )

                    cprint(f"Processing successful: {astrometry['success']}, color='green")
                    if astrometry['success']:
                        self.log.debug("Solution files created:")
                        for name, path in astrometry['solution_files'].items():
                            self.log.debug(f"  {name}: {path}")
                except Exception as e:
                    cprint(f"Error processing centroids: {e}", color='red')
                    print(f"Stack trace:\n{traceback.format_exc()}")
                    self.log.error(f"Error processing centroids: {e}")
                    self.log.error(f"Stack trace:\n{traceback.format_exc()}")
            else:
                pass

            # Add execution time
            astrometry["precomputed_star_centroids"] = precomputed_star_centroids.shape[0]
            astrometry['cedar_detect'] = self.cedar.cd_solutions.copy()
            astrometry["params"] = get_params()
            astrometry["solver_exec_time"] = solver_exec_time
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

            with suppress(TypeError, ValueError, IndexError):
                #                                              DETECT                      ASTRO RESULTS
                try:
                    # TODO: Remove calculate_rms_errors_from_centroids old invocation
                    # rms = self.calculate_rms_errors_from_centroids(precomputed_star_centroids, astrometry, image_size)
                    rms = self.calculate_rms_errors_from_centroids(astrometry, image_size)
                    astrometry = astrometry | rms   # Merge - union two dicts
                except (TypeError, ValueError) as e:
                    pass
                    # self.log.error(f"Failed to compute RMS: {e}")
                    # self.log.error(f"Exception type: {type(e).__name__}")
                    # self.log.error(f"Stack trace:\n{traceback.format_exc()}")
                    # print(f"Stack trace:\n{traceback.format_exc()}")
        else:
            cprint('No centroids found, skipping astrometry solving.', color='red')
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
            solver='solver2',
            level_filter=9,
            ring_filter_type='mean',
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
