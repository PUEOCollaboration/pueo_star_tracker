# Standard imports
import os
import time
import logging
from termcolor import cprint
from time import perf_counter as precision_timestamp
from contextlib import suppress
from multiprocessing import shared_memory
# from lib.cedar_detect.python.cedar_detect_client import extract_centroids

# External imports
import cv2
import numpy as np

# Cedar Detect
import grpc

# Pueo Custom Imports
import lib.cedar_detect.python.cedar_detect_pb2 as cedar_detect_pb2
import lib.cedar_detect.python.cedar_detect_pb2_grpc as cedar_detect_pb2_grpc
from lib.common import save_to_json, get_dt, float_range
from lib.utils import read_image_grayscale, read_image_BGR, display_overlay_info, timed_function, print_img_info


class Cedar:
    """Cedar Image/Centroids/Post Filtering Class"""
    test = False
    test_data = {}
    tetra3_solver = None
    solver_name = 'Cedar Tetra3'
    return_partial_images = True
    partial_results_path = "./partial_results"
    distortion_calibration_params = {}
    min_pattern_checking_stars = 10

    def __init__(self, database_name=None, cfg=None):
        self.log = logging.getLogger('pueo')
        self.log.info('Initializing Cedar Object')

        self.cfg = cfg

        # Set up to make gRPC calls to CedarDetect centroid finder (it must be running
        # already).
        # if database_name:
        #     self.channel = grpc.insecure_channel(self.cfg.cedar_detect_host)
        #     self.stub = cedar_detect_pb2_grpc.CedarDetectStub(self.channel)

        self.cd_results = {}
        self.cd_solutions = []

    def extract_centroids(self, stub, image, sigma=None):
        """Cedar Detect Extract Centroids"""
        sigma = sigma or self.cfg.sigma
        max_size = self.cfg.max_size
        binning = self.cfg.binning
        return_binned = self.cfg.return_binned
        use_binned_for_star_candidates = self.cfg.use_binned_for_star_candidates
        detect_hot_pixels = self.cfg.detect_hot_pixels

        if self.test:
            # sigma = self.test_data.get('sigma')
            max_size = self.test_data.get('max_size')
            binning = self.test_data.get('binning')
            return_binned = self.test_data.get('return_binned')
            use_binned_for_star_candidates = self.test_data.get('use_binned_for_star_candidates')
            detect_hot_pixels = self.test_data.get('detect_hot_pixels')

        cedar_data = {
            'sigma': sigma,
            'max_size': self.cfg.max_size,
            'binning': self.cfg.binning,
            'return_binned': self.cfg.return_binned,
            'use_binned_for_star_candidates': self.cfg.use_binned_for_star_candidates,
            'detect_hot_pixels': self.cfg.detect_hot_pixels,
        }

        cprint(f'  Actual sigma: {sigma}', 'cyan')
        cr = cedar_detect_pb2.CentroidsRequest(
            # input_image=image, sigma=8.0, max_size=5, return_binned=False, use_binned_for_star_candidates=True
            input_image=image,
            sigma=sigma,  # 2.5,
            max_size=max_size,  # 5,
            binning=binning,  # 2,
            return_binned=return_binned,  # True,
            use_binned_for_star_candidates=use_binned_for_star_candidates,  # True,
            detect_hot_pixels=detect_hot_pixels  # False
        )
        return stub.ExtractCentroids(cr), cedar_data

    def resize(self, img, resize_factor=None, max_bytes=4194304):
        """
        Resizes an image proportionally so that its size in bytes (height * width) is less than or equal to max_bytes.

        Args:
            img (numpy.ndarray): The input image as a NumPy array (OpenCV image format).
            resize_factor (float): Resize factor, automatically calculated when None
            max_bytes (int): The maximum allowed size in bytes (height * width). Default is 4,194,304 bytes.

        Returns:
            tuple: A tuple containing:
                - resized_img (numpy.ndarray): The resized image.
                - resize_factor (float): The factor by which the image was resized. Returns 1 if no resizing was needed.
        """
        # Get the original dimensions of the image
        height, width = img.shape[:2]

        # Calculate the current size in bytes (height * width)
        current_size = height * width

        # If the image already fits the condition, return it as is
        if current_size <= max_bytes:
            self.log.debug(f'Image size fits max_bytes: {current_size}/{max_bytes}')
            return img, 1.0

        # Calculate the resize factor needed to meet the size constraint
        if resize_factor is None:
            resize_factor = 1 / (max_bytes / current_size) ** 0.5  # Square root to maintain aspect ratio

        # Calculate the new dimensions
        new_width = int(width / resize_factor)
        new_height = int(height / resize_factor)

        # Resize the image using OpenCV
        resized_img = cv2.resize(img, (new_width, new_height), interpolation=cv2.INTER_AREA)
        self.log.debug(f'Image resize: factor: {resize_factor} bytes: {new_width * new_height}/{max_bytes} shape: {resized_img.shape} orig: {img.shape}')
        cprint(f'Image resize: factor: {resize_factor} bytes: {new_width * new_height}/{max_bytes} shape: {resized_img.shape} orig: {img.shape}', color='yellow')
        return resized_img, resize_factor

    def cedar_image_prepare(self, img):
        """Prepare image for cedar detect step 2: Gaussian Smoothing, 3. Background subtraction."""
        # Step 2: Apply Gaussian smoothing with FWHM = 3 (sigma ≈ 1.27)
        # Sigma = fwhm/2.355 is the formula
        cprint('Pre-Cedar Detect: Applying Gaussian smoothing with FWHM', color='yellow')
        sigma_x = self.cfg.gaussian_fwhm / 2.355 #
        self.cd_results['gaussian_fwhm'] = self.cfg.gaussian_fwhm
        self.cd_results['sigma_x'] = sigma_x
        smoothed_corr = cv2.GaussianBlur(img, (0, 0), sigmaX=sigma_x)

        return smoothed_corr

        # Note: Oh i see you're code. Don't implement step 3.
        # My code only assumed a single bright source in the image to see what the effect is on a single star.
        # But also my step 3 is already what Cedar does.
        # I only have it in my code to simulate what cedar detect does.

        # Step 3: Background subtraction using 7x7 median around brightest pixel
        cprint('Pre-Cedar Detect: Applying Background subtraction using 7x7 median around brightest pixel',
               color='yellow')
        brightest_y, brightest_x = np.unravel_index(np.argmax(smoothed_corr), smoothed_corr.shape)
        half_size = 3
        y_min = max(brightest_y - half_size, 0)
        y_max = min(brightest_y + half_size + 1, smoothed_corr.shape[0])
        x_min = max(brightest_x - half_size, 0)
        x_max = min(brightest_x + half_size + 1, smoothed_corr.shape[1])
        median_val = np.median(smoothed_corr[y_min:y_max, x_min:x_max])
        subtracted_corr = smoothed_corr.astype(np.float32) - median_val
        subtracted_corr = np.clip(subtracted_corr, 0, 255).astype(np.uint8)

        return subtracted_corr

    def pixel_count_filter(self, stars, min_pixels: int = 9, max_pixels: int = 45):
        """
        Filters a list of stars based on their pixel count, keeping only those within the specified range.

        Args:
            stars: List of stars where each star is represented as [y, x, brightness, pixel_count]
            min_pixels: Minimum allowed pixel count for a star (inclusive, default: 9)
            max_pixels: Maximum allowed pixel count for a star (inclusive, default: 45)

        Returns:
            List: Filtered list of stars that have pixel counts within [min_pixels, max_pixels]

        Example:
            >>> stars = [[10, 20, 150, 8], [30, 40, 200, 10], [50, 60, 180, 50]]
            >>> pixel_count_filter(stars, 9, 45)
            [[30, 40, 200, 10]]
        """
        return [star for star in stars if min_pixels <= star[3] <= max_pixels]

    def spatial_filter(self, stars, min_distance: int = 50):
        """
        Filters a list of stars to ensure no two stars are closer than min_distance pixels.
        When two stars are too close, the brighter one is kept.

        Args:
            stars: List of stars in format [[y, x, brightness, pixel_count], ...]
            min_distance: Minimum allowed distance between stars (default: 50 pixels)

        Returns:
            Filtered list of stars
        """
        # Sort stars by brightness (descending) so we keep brighter ones when removing
        stars_sorted = sorted(stars, key=lambda s: -s[2])
        filtered = []

        min_distance_sq = min_distance ** 2
        for star in stars_sorted:
            # Check distance against all already accepted stars
            too_close = False
            for kept_star in filtered:
                dy = star[0] - kept_star[0]
                dx = star[1] - kept_star[1]
                distance_sq = dx * dx + dy * dy

                if distance_sq <= min_distance_sq:
                    too_close = True
                    break

            if not too_close:
                filtered += star,

        return filtered

    def get_centroids(self, img_bgr, img):

        # Assuming `image` is your NumPy array with shape (x, y, 3)
        def rgb_to_grayscale(image):
            # Ensure the input is a NumPy array
            if not isinstance(image, np.ndarray):
                raise ValueError("Input must be a NumPy array")

            # Ensure the input has 3 channels (RGB)
            if image.shape[2] != 3:
                raise ValueError("Input image must have 3 channels (RGB)")

            # Apply the grayscale conversion formula
            grayscale = np.dot(image[..., :3], [0.2989, 0.5870, 0.1140])

            # Convert to uint8 (if not already)
            grayscale = grayscale.astype(np.uint8)

            return grayscale

        t0 = time.monotonic()
        channel = grpc.insecure_channel(self.cfg.cedar_detect_host, options=(('grpc.enable_http_proxy', 0),))
        stub = cedar_detect_pb2_grpc.CedarDetectStub(channel)

        # cd_results: Used for REPORTING and TESTING
        self.cd_solutions = []
        self.cd_results = {}

        test = False
        if test:
            from PIL import Image
            test_image_name = r'test_images/0.7-blast_image-2020-01-06--22-43-26--602.png'
            with Image.open(str(test_image_name)) as t_img:
                t_img = t_img.convert(mode='L')
                (width, height) = (t_img.width, t_img.height)

                image = np.asarray(t_img, dtype=np.uint8)
        else:
            # image = np.asarray(img, dtype=np.uint8)
            # image = img_bgr.convert(mode='L')

            # Step 0: Convert to grayscale
            image_txt = 'Converted to grayscale'
            try:
                img_cdr = rgb_to_grayscale(img_bgr)  # Converting bgr to grayscale
            except IndexError as e:
                img_cdr = img_bgr
                image_txt = 'Kept as is'
            cprint(f'  Image: {image_txt}', 'yellow')
            if self.return_partial_images:
                cv2.imwrite(os.path.join(self.partial_results_path, "4.0 - Cedar prepare - Grayed.png"), img_cdr)

            self.log.debug('Resizing?')
            resize_factor = self.cfg.cedar_downscale
            self.cd_results['cedar_downscale'] = self.cfg.cedar_downscale

            # Step 1: Resize image to fit max size for cedar_message
            img_grayed = img_cdr.copy()
            img_cdr, resize_factor = self.resize(img_cdr, resize_factor)
            self.cd_results['actual_downscale'] = resize_factor

            if self.return_partial_images:
                cv2.imwrite(os.path.join(self.partial_results_path, "4.1 - Cedar prepare - Downscaled.png"), img_cdr)

            # Prepare image for cedar detect Step 2: Gaussian Smoothing, 3. Background subtraction.
            is_gaussian_enabled = True
            if is_gaussian_enabled:
                img_cdr = self.cedar_image_prepare(img_cdr)
                if self.return_partial_images:
                    cv2.imwrite(os.path.join(self.partial_results_path, "4.2 - Cedar prepare - GaussianBlur.png"),
                                img_cdr)
            else:
                cprint(f'  Skipping GaussianBlur', color='blue')
            image = img_cdr

        # (width, height) = image.shape[:2]

        channels = 1
        height = width = 0
        if len(image.shape) == 2:
            height, width = image.shape
        elif len(image.shape) == 3:
            height, width, channels = image.shape  # Get dimensions

        cprint(f'  Image shape: {height}x{width}x{channels} resize_factor: {resize_factor}', color='yellow')

        USE_SHMEM = False
        centroids_result = None
        rpc_duration_secs = None
        if USE_SHMEM:
            # Using shared memory. The image data is passed in a shared memory
            # object, with the gRPC request giving the name of the shared memory
            # object.

            # Set up shared memory object for passing input image to CedarDetect.
            shmem = shared_memory.SharedMemory(
                "/cedar_detect_image", create=True, size=height * width * channels)
            try:
                # Create numpy array backed by shmem.
                shimg = np.ndarray(image.shape, dtype=image.dtype, buffer=shmem.buf)
                # Copy image into shimg. This is much cheaper than passing image
                # over the gRPC call.
                shimg[:] = image[:]

                im = cedar_detect_pb2.Image(width=width, height=height, shmem_name=shmem.name)
                rpc_start = precision_timestamp()
                centroids_result = self.extract_centroids(self.stub, im)
                rpc_duration_secs = precision_timestamp() - rpc_start
            finally:
                shmem.close()
                shmem.unlink()
                del shmem
        else:
            # Not using shared memory. The image data is passed as part of the
            # gRPC request.
            # NOTE:   debug_error_string = "UNKNOWN:Error received from peer ipv4:192.168.1.75:50051
            # {created_time:"2025-03-15T22:18:41.3140333+00:00", grpc_status:11,
            # grpc_message:"Error, message length too large: found 11694397 bytes, the limit is: 4194304 bytes"}"
            # save image. For debugging
            im = cedar_detect_pb2.Image(width=width, height=height, image_data=image.tobytes())
            # Try to find optimal sigma that will find no more than 150
            min_star_candidates = 8
            max_star_candidates = 300
            min_sigma = max_sigma = self.cfg.sigma
            idx = 0
            min_idx = None
            solutions = {}
            min_star_candidates = float('inf')  # Initialize with positive infinity
            tetra3_exec_time = 0.0

            for sigma in float_range(self.cfg.sigma):
                min_sigma = sigma if idx == 0 else min_sigma
                centroids_result, cedar_data = self.extract_centroids(stub, im, sigma)
                star_candidates = len(centroids_result.star_candidates)
                tetra_centroids = []  # List of (y, x).
                precomputed_star_centroids = np.ndarray([])
                cedar_star_centroids = np.ndarray([])
                dummy_diameter = 20
                star_candidates_cnt = 0
                # Cleanup cd_results
                self.cd_results['sigma'] = sigma
                self.cd_results['sigma_x'] = None
                self.cd_results['gaussian_fwhm'] = None
                self.cd_results['solution'] = False
                self.cd_results['star_centroids'] = len(centroids_result.star_candidates)
                self.cd_results['refined_circular'] = None
                self.cd_results['refined_with_area'] = None
                self.cd_results['final_filtered_30'] = None
                self.cd_results['top_30_flux'] = None

                if len(centroids_result.star_candidates) == 0:
                    cprint('  Found no stars!', 'red')
                else:
                    cedar_centroids = []
                    for sc in centroids_result.star_candidates:
                        tetra_centroids.append(
                            [sc.centroid_position.y, sc.centroid_position.x,
                             sc.brightness, sc.pixel_count, dummy_diameter])
                        cedar_centroids.append(
                            [sc.centroid_position.y, sc.centroid_position.x, sc.brightness, sc.pixel_count])

                    # Add centroids pre-filtering
                    # Spacial filtering from the highest brightness.
                    # arcsec * 60 = arcmin, min separation of 50 pixels
                    # Step 1: Keep only stars with pixel_count between 9 and 45

                    cedar_star_centroids = np.array(tetra_centroids)
                    is_filter_v1 = False
                    if is_filter_v1:
                        star_candidates_cnt = len(tetra_centroids)
                        tetra_centroids_f1 = self.pixel_count_filter(tetra_centroids, self.cfg.pixel_count_min,
                                                                     self.cfg.pixel_count_max)
                        star_candidates_f1_cnt = len(tetra_centroids_f1)

                        # Step 2: Min distance between centroids: 25 pixels
                        tetra_centroids_f2 = self.spatial_filter(tetra_centroids_f1,
                                                                 min_distance=self.cfg.spatial_distance_px)
                        star_candidates_f12_cnt = len(tetra_centroids_f1)
                        star_candidates_f2_cnt = len(tetra_centroids_f2)

                        contains_all = all(sublist in cedar_star_centroids for sublist in tetra_centroids_f2)
                        # print(contains_all)  # True if all sub lists of list2 exist in list1

                        star_candidates = star_candidates_f2_cnt
                    else:
                        tetra_centroids_f2 = self.filter_centroids(image, cedar_star_centroids, sigma)

                    precomputed_star_centroids = np.array(tetra_centroids_f2)
                    if self.cfg.return_partial_images:
                        centroids_result_dict = {
                            'noise_estimate': centroids_result.noise_estimate,
                            'star_candidates': cedar_centroids,
                            'algorithm_time': centroids_result.algorithm_time.nanos,
                            'peak_star_pixel': centroids_result.peak_star_pixel,
                            'star_candidates_cnt': star_candidates_cnt,
                            # 'pixel_count_filter_cnt': star_candidates_f1_cnt,
                            # 'spatial_filter_cnt': star_candidates_f2_cnt,
                            'filtered_candidates': tetra_centroids
                        }
                        centroids_result_filename = f'{self.cfg.partial_results_path}5.0 - Cedar-StarCandidates-{sigma:3.1f}.json'
                        save_to_json(centroids_result_dict, centroids_result_filename)
                    # cprint(
                    #     f'  Centroids: contains all: {contains_all} {len(precomputed_star_centroids)}, star_candidates: {star_candidates_cnt} pixel filter: {star_candidates_f1_cnt} spatial filter: {star_candidates_f2_cnt} in {get_dt(t0)}.',
                    #     color='yellow')

                # Cedar solver does not support distortion!
                max_c = 0 if precomputed_star_centroids.shape == () else precomputed_star_centroids.shape[0]

                astrometry = {}
                with suppress(IndexError):

                    astrometry, tetra3_exec_time_1 = timed_function(
                        self.tetra3_solver,
                        image,
                        precomputed_star_centroids=precomputed_star_centroids[:max_c, :2],  # Take only 20 centroids
                        FOV=self.distortion_calibration_params["FOV"],
                        min_pattern_checking_stars=self.min_pattern_checking_stars,
                    )
                    tetra3_exec_time += tetra3_exec_time_1
                ra = astrometry.get('RA')
                dec = astrometry.get('Dec')
                fov = astrometry.get('FOV')
                color = 'green' if ra else 'yellow'
                cprint(f'  Cedar detect sigma: {sigma:3.1f} star_candidates: {star_candidates} solution: RA: {ra} Dec: {dec}', color=color)
                # if min_star_candidates <= star_candidates and star_candidates <= max_star_candidates:
                # blast image: the correct solution is (RA, Dec): (162.239, -48.328)

                self.cd_results['solution'] = False
                self.cd_results['RA'] = ra
                self.cd_results['dec'] = dec
                self.cd_results['FOV'] = fov
                if ra is not None:
                    self.cd_results['solution'] = True
                    solutions[idx] = {
                        'sigma': sigma,
                        'star_candidates': star_candidates,
                        'RA': ra, 'Dec': dec, 'astrometry': astrometry.copy(),
                        'precomputed_star_centroids': precomputed_star_centroids.copy(),
                        'cedar_star_centroids': cedar_star_centroids.copy(),
                    }
                    if star_candidates < min_star_candidates:
                        min_sigma = sigma
                        min_star_candidates = star_candidates
                        min_idx = idx
                    idx += 1
                    if not self.test:
                        break

                # Save solution
                self.cd_solutions.append(self.cd_results.copy())

            cprint(f'Solution summary: {self.solver_name}', color='green')
            for idx, sol in solutions.items():
                cprint(
                    f'  Cedar detect sigma [{idx}]: {sol['sigma']:3.1f} star_candidates: {sol['star_candidates']} prefiltred: {precomputed_star_centroids.shape[0]} solution: RA: {sol['RA']} Dec: {sol['Dec']}',
                    color='green')
            cprint(f'  Solution with min_idx: {min_idx}', color='blue')
            # Use the solution with minimal star_candidates
            cedar_star_centroids = np.array([])
            precomputed_star_centroids = np.array([])
            astrometry = {}
            if min_idx is not None:
                cedar_data['sigma'] = min_sigma
                cedar_star_centroids = solutions[min_idx]['cedar_star_centroids'].copy()
                precomputed_star_centroids = solutions[min_idx]['precomputed_star_centroids'].copy()
                astrometry = solutions[min_idx]['astrometry'].copy()
            # rpc_duration_secs = precision_timestamp() - rpc_start

            # if cedar_star_centroids.size > 0:
            #     precomputed_star_centroids = self.filter_centroids(img_grayed, cedar_star_centroids)

            # precomputed_star_centroids = [[y * resize_factor, x * resize_factor, b, c, f] for (y, x, b, c, f) in
            #                               precomputed_star_centroids]

            cedar_exec_time = time.monotonic() - t0
            return precomputed_star_centroids, cedar_star_centroids, resize_factor, astrometry,  tetra3_exec_time # 1.0 # resize_factor

    def filter_centroids(self, image_gray, star_candidates, sigma):
        centroids = [[entry[0], entry[1], entry[2], entry[3]] for entry in star_candidates]
        self.cd_results['sigma'] = sigma
        self.cd_results['star_centroids'] = len(centroids)

        # Parameters
        box_size = 20
        min_separation = 30
        min_area = 5
        max_area = 150
        min_circularity = 0.82
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))

        # Step 1: initialize output image
        image_colored = cv2.cvtColor(image_gray, cv2.COLOR_GRAY2BGR)

        # Stage 1: Draw red circles on all original centroids
        for y, x, *_ in centroids:
            cv2.circle(image_colored, (int(x), int(y)), 5, (0, 0, 255), 1)  # Red

        # Stage 2a: Filter by circularity
        refined_circular = []
        for cy, cx, flux, circ0 in centroids:
            cx, cy = int(cx), int(cy)
            x1, x2 = cx - box_size // 2, cx + box_size // 2
            y1, y2 = cy - box_size // 2, cy + box_size // 2
            if x1 < 0 or y1 < 0 or x2 >= image_gray.shape[1] or y2 >= image_gray.shape[0]:
                continue

            region = image_gray[y1:y2, x1:x2].copy()
            _, thresh = cv2.threshold(region, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            thresh_dilated = cv2.dilate(thresh, kernel, iterations=1)
            contours, _ = cv2.findContours(thresh_dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if not contours:
                continue

            contour = max(contours, key=cv2.contourArea)
            area = cv2.contourArea(contour)
            perimeter = cv2.arcLength(contour, True)
            if perimeter == 0:
                continue
            circularity = 4 * np.pi * area / (perimeter ** 2)

            if circularity < min_circularity:
                continue

            M = cv2.moments(contour)
            if M["m00"] == 0:
                continue
            new_cx = int(M["m10"] / M["m00"]) + x1
            new_cy = int(M["m01"] / M["m00"]) + y1

            mask = np.zeros_like(region, dtype=np.uint8)
            cv2.drawContours(mask, [contour], -1, 255, -1)
            flux = np.sum(region[mask == 255])

            refined_circular.append([new_cy, new_cx, flux, area, circularity])
            cv2.circle(image_colored, (new_cx, new_cy), 6, (255, 0, 0), 1)  # Blue for circularity

        self.cd_results['refined_circular'] = len(refined_circular)

        # Stage 2b: Filter by area
        refined_with_area = [entry for entry in refined_circular if min_area <= entry[3] <= max_area]
        self.cd_results['refined_with_area'] = len(refined_with_area)

        for y, x, *_ in refined_with_area:
            cv2.circle(image_colored, (int(x), int(y)), 8, (0, 255, 255), 1)  # Yellow for area

        # Stage 3: Spatial filtering
        # refined_sorted = sorted(refined_with_area, key=lambda x: -abs(x[2]))
        # Replace the negation with reverse sorting: to prevent RuntimeWarning
        # Replace the negation with reverse sorting:
        refined_sorted = sorted(refined_with_area, key=lambda x: abs(x[2]), reverse=True)

        final_filtered_30 = []
        for cy, cx, flux, area, circ in refined_sorted:
            if all(np.hypot(cx - fx, cy - fy) > min_separation for fx, fy, *_ in final_filtered_30):
                final_filtered_30.append([cy, cx, flux, area, circ])

        self.cd_results['final_filtered_30'] = len(final_filtered_30)
        for y, x, *_ in final_filtered_30:
            cv2.circle(image_colored, (int(x), int(y)), 10, (0, 165, 255),
                       1)  # Orange for spatially filtered  # Green for final

        # Add black rectangle background for legend
        cv2.rectangle(image_colored, (10, 5), (320, 125), (0, 0, 0), -1)

        # Add overlay legend text
        # Add overlay legend text
        cv2.putText(image_colored, f"Red: Original ({len(centroids)})", (15, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                    (0, 0, 255), 1)
        cv2.putText(image_colored, f"Blue: Passed Circularity ({len(refined_circular)})", (15, 45),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 1)
        cv2.putText(image_colored, f"Yellow: Passed Area ({len(refined_with_area)})", (15, 65),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 1)
        cv2.putText(image_colored, f"Orange: Final Filtered ({len(final_filtered_30)})", (15, 85),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 165, 255), 1)

        # Save annotated image
        # Export top 30 by flux
        # sorted_final_flux = sorted(final_filtered_30, key=lambda x: -abs(x[2]))
        sorted_final_flux = sorted(final_filtered_30, key=lambda x: abs(x[2]), reverse=True)

        top_30_flux = sorted_final_flux[:30]
        self.cd_results['top_30_flux'] = len(top_30_flux)

        # Draw arrows and labels for top 10 brightest
        for idx, (y, x, _, _, circ) in enumerate(top_30_flux[:10]):
            pt = (int(x), int(y))
            label_text = f"{idx + 1} (C={circ:.2f})"
            (text_width, _), _ = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
            offset_x = x + 25 if x + 25 + text_width < image_colored.shape[1] else x - (text_width + 25)
            offset_y = y - 25 if y - 25 > 15 else y + 25
            offset = (int(offset_x), int(offset_y))
            cv2.arrowedLine(image_colored, offset, pt, (255, 255, 255), 1, tipLength=0.2)
            # v1: cv2.putText(image_colored, label_text, (offset[0] + 5, offset[1]),
            #             cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
            # v2:
            cv2.putText(image_colored, label_text, (offset[0] + 5, offset[1]),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1,
                        cv2.LINE_AA)  # Optional anti-aliased line type

            # cv2.putText(image_colored, label_text, (offset[0] + 5, offset[1]), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

        # Draw green circles for top 30 brightest
        for y, x, *_ in top_30_flux:
            cv2.circle(image_colored, (int(x), int(y)), 12, (0, 255, 0), 1)  # Green for top 30

        # cv2.putText(image_colored, f"Green: Top 30 Brightest ({len(top_30_flux)})", (15, 105), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 1)})", (15, 105), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 1)})", (15, 105), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 1), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 1)
        cv2.putText(
            image_colored,
            f"Green: Top 30 Brightest ({len(top_30_flux)})",
            (15, 105),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 0),
            1
        )

        if self.partial_results_path:
            # cv2.imwrite(f"star_filtering_stages_real-{sigma:3.1f}.png", image_colored)
            cv2.imwrite(os.path.join(self.partial_results_path, f"4.5 - Cedar prepare - Star Filtering Stages Real-{sigma:3.1f}.png"), image_colored)

        # with open("/mnt/data/top_30_star_centroids_by_flux.txt", "w") as f:
        #     for x, y, flux, *_ in top_30_flux:
        #         f.write(f"{x}, {y}\n")
        cprint(f'  Cedar Centroids Filtering Stages: star_candidates: {len(centroids)} passed_circularity: {len(refined_circular)} passed_area: {len(refined_with_area)} final_filtered: {len(final_filtered_30)}', color='cyan')
        return np.array(top_30_flux)

# Last line
