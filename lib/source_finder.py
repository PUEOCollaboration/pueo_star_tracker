import os
from contextlib import suppress

import cv2
import numpy as np
from astropy.stats import sigma_clipped_stats
from lib.common import logit, get_dt
from functools import lru_cache
import time
import logging

from typing import Optional
import threading

class SourceFinder:
    """Source Finder Class"""

    def __init__(self, cfg, logger: Optional[logging.Logger] = None, server=None):
        self.cfg = cfg
        self.log = logger or logging.getLogger("pueo")
        self.server = server
        
    @staticmethod
    def center_roi_view(arr, frac_x=1.0, frac_y=1.0):
        """
        Return a centered ROI view (no copy) and (x0, y0) offset.
        """
        h, w = arr.shape[:2]
        fx = max(0.01, min(1.0, float(frac_x)))
        fy = max(0.01, min(1.0, float(frac_y)))

        roi_w = max(1, int(round(w * fx)))
        roi_h = max(1, int(round(h * fy)))

        x0 = (w - roi_w) // 2
        y0 = (h - roi_h) // 2
        return arr[y0:y0 + roi_h, x0:x0 + roi_w], (x0, y0)

    def threshold_with_noise(
            self,
            residual_img: np.ndarray,
            sigma_map: np.ndarray,
            *,
            k_high: float = 8.0,  # behaves like your "threshold" multiplier
            use_hysteresis: bool = False,  # False -> simple single threshold
            k_low: float = 7.0,  # only used if use_hysteresis=True
            min_area: int = 10,
            close_kernel: int = 3,  # 0/1 to disable
            sigma_floor: float = 1e-6,  # avoid div-by-zero / tiny-sigma spikes
            sigma_gauss: float = 0.0  # >0 to lightly smooth the sigma map
    ):
        """
        Build a sources mask using your per-pixel noise image (sigma_map).
        If use_hysteresis=False, this reduces to: residual > k_high * sigma_map (then clean-up).
        If use_hysteresis=True, weak= k_low*σ, strong= k_high*σ, then keep weak comps touching strong.
        Returns (mask_bool, mask_u8) where mask_u8 is 0/255 for cv2.imwrite().
        """
        assert residual_img.shape == sigma_map.shape, "residual_img and sigma_map must match shape"

        # optional mild smoothing of sigma to reduce over-structured masks
        if sigma_gauss and sigma_gauss > 0:
            sigma_map = cv2.GaussianBlur(sigma_map, (0, 0), sigma_gauss)

        # clamp sigma to avoid tiny values creating speckle
        sigma_map = np.maximum(sigma_map, sigma_floor)

        if not use_hysteresis:
            mask = residual_img > (k_high * sigma_map)
        else:
            strong = residual_img > (k_high * sigma_map)
            weak = residual_img > (k_low * sigma_map)

            # --- Faster hysteresis via morphological reconstruction (no connectedComponents) ---
            strong_u8 = strong.astype(np.uint8) * 255
            weak_u8 = weak.astype(np.uint8) * 255

            k = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
            rec = strong_u8.copy()
            while True:
                prev = rec
                rec = cv2.dilate(rec, k, iterations=1)
                rec = cv2.bitwise_and(rec, weak_u8)  # grow only inside weak
                rec = cv2.bitwise_or(rec, strong_u8)  # keep seeds on
                if cv2.countNonZero(cv2.absdiff(rec, prev)) == 0:
                    break

            mask = rec > 0

        # Morphological close to fill tiny gaps
        if close_kernel and close_kernel > 1:
            k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (close_kernel, close_kernel))
            mask = cv2.morphologyEx(mask.astype(np.uint8), cv2.MORPH_CLOSE, k).astype(bool)

        # Area filter
        if min_area and min_area > 1:
            m_u8 = mask.astype(np.uint8)
            num, labels = cv2.connectedComponents(m_u8, connectivity=8)
            if num > 1:
                # Component pixel counts (skip 0)
                counts = np.bincount(labels.ravel())
                small = (counts < min_area)
                small[0] = False
                # Remove small components
                remove = small[labels]
                mask = np.logical_and(mask, ~remove)

        mask_u8 = (mask.astype(np.uint8) * 255)
        return mask, mask_u8


    def global_background_estimation(
            self,
            img,
            sigma_clipped_sigma: float = 3.0,
            return_partial_images=False,
            partial_results_path="./partial_results/",
    ):
        """Estimate and subtract the global median background from an image using astropy's
        'sigma_clipped_stats' method to filter out noise and outliers.

        Args:
            img (numpy.ndarray): The input image (2D array).
            sigma_clipped_sigma (float, optional): The sigma value to use for sigma-clipping when calculating the
                background statistics (mean, median, standard deviation). It controls how aggressive the clipping is.
                Higher values mean less clipping (more tolerance for outliers). Defaults to 3.0.
            return_partial_images (bool, optional): If True, the function saves the background-subtracted image. Defaults to False.
            partial_results_path (str, optional): The directory path where intermediate results will be saved
                if `return_partial_images` is True. Defaults to "./partial_results/".

        Returns:
            numpy.ndarray: The image with the global background subtracted.
        """
        #  Global Background subtraction
        # TODO: Done sigma_clipped_sigma = 3.0
        mean, median, std = sigma_clipped_stats(img, sigma=sigma_clipped_sigma)
        cleaned_img = img - median

        # Save Background subtracted image
        if return_partial_images:
            cv2.imwrite(os.path.join(partial_results_path, "1.1 - Global Background subtracted image.png"), cleaned_img)

        return cleaned_img


    @lru_cache(maxsize=16)
    def _make_perimeter_kernel(self, S: int, w: int = 1, exclude_corners: bool = True) -> np.ndarray:
        """
        Build an S×S perimeter kernel of thickness w.
        If exclude_corners=True, the w×w corner blocks are set to 0.
        Returns a float32 kernel normalized to sum=1 (mean).
        """
        S = int(S)
        w = int(w)
        K = np.zeros((S, S), np.float32)

        # draw w-px border
        K[:w, :] = 1.0
        K[-w:, :] = 1.0
        K[:, :w] = 1.0
        K[:, -w:] = 1.0

        if exclude_corners:
            # remove the four corners (each w×w)
            K[:w, :w] = 0.0
            K[:w, -w:] = 0.0
            K[-w:, :w] = 0.0
            K[-w:, -w:] = 0.0

        s = K.sum()
        K /= (s if s > 0 else 1.0)

        return K


    def ring_mean_background_estimation(
            self,
            downsampled_img,
            d_small,
    ):
        """
        Per-pixel local background via a square *perimeter mean* using filter2D.
        - perimeter thickness = 1 px
        - corners excluded
        - odd window size enforced
        Returns float32 image on the same grid as `downsampled_img`.
        """
        # 1) enforce odd window
        if d_small % 2 == 0:
            d_small += 1

        I = downsampled_img.astype(np.float32, copy=False)
        S = int(d_small)

        # 2) perimeter kernel (1 px thick, corners excluded), normalized to mean
        K = self._make_perimeter_kernel(S, w=1, exclude_corners=True)

        # 3) apply with a consistent border mode
        local_levels = cv2.filter2D(I, ddepth=-1, kernel=K, borderType=cv2.BORDER_REFLECT)

        return local_levels


    def subtract_background(self, image: np.ndarray, background: np.ndarray):
        # Subtract background
        flattened_img = image.astype(np.float32) - background.astype(np.float32)

        # Clip for display (avoid negatives, keep within valid dtype range)
        if np.issubdtype(image.dtype, np.integer):
            dtype_info = np.iinfo(image.dtype)
            flattened_img = np.clip(flattened_img, 0, dtype_info.max).astype(image.dtype)

        return flattened_img

        # Ensure same dtype for OpenCV
        if background.dtype != image.dtype:
            background = background.astype(image.dtype, copy=False)
        # Saturating subtract: for uint16, result = max(image - background, 0)
        flattened_img = cv2.subtract(image, background)
        return flattened_img


    def estimate_noise_pairs(
            self,
            img: np.ndarray,
            sep: int = 5,
    ) -> float:
        """
        Estimate per-pixel noise σ using random pixel pairs separated by `sep` rows and `sep` columns.
        Implements: var(Δ) ≈ 2 σ²  =>  σ ≈ std(Δ) / sqrt(2).

        Parameters
        ----------
        img : np.ndarray
            2D image (ideally *flattened* / background-subtracted).
        sep : int
            Separation in both row and column (diagonal offset). Default 5 per the text.

        Returns
        -------
        float
            Estimated noise σ.
        """
        if img.ndim != 2:
            raise ValueError("img must be a 2D array")
        H, W = img.shape
        if H <= sep or W <= sep:
            raise ValueError(f"Image too small for sep={sep}: got {img.shape}")

        # Work in float for accurate differences; avoid copies if already float
        a = img.astype(np.float32, copy=False)

        # Build the diagonal difference field: Δ = I[y,x] - I[y+sep, x+sep]
        # Shape is (H-sep, W-sep)
        diff = a[:-sep, :-sep] - a[sep:, sep:]
        flat = diff.ravel()

        # Standard deviation with Bessel’s correction
        s = flat.std(ddof=1) if flat.size > 1 else 0.0

        # convert from difference variance to per-pixel noise
        return float(s / np.sqrt(2.0))


    def source_finder(
            self,
            img,
            log_file_path="",
            leveling_filter_downscale_factor: int = 4,
            fast=False,
            threshold: float = 3.1,
            local_sigma_cell_size=36,
            kernal_size_x=3,
            kernal_size_y=3,
            sigma_x=1,
            dst=1,
            number_sources: int = 40,
            min_size=20,
            max_size=600,
            dilate_mask_iterations=1,
            is_trail=None,
            return_partial_images=False,
            partial_results_path="./partial_results/",
            level_filter=9,
            ring_filter_type="mean",
            *,
            noise_pair_sep_down=3,
            noise_pair_sep_full=5,
            simple_threshold_k: float = 8.0,
            hyst_k_high: float = 8.0,
            hyst_k_low: float = 6.0,
            hyst_min_area: int = 10,
            hyst_close_kernel: int = 20,
            hyst_sigma_gauss: float = 0.0,
            merge_min_area: int = 10,
            merge_gap_along: int = 20,
            merge_gap_cross: int = 3,
            merge_ang_tol: int = 15,
    ):
        """Function combining the source finding pipeline for faster execution time.

        - **Local levels background estimation:** Estimate and subtract local background levels in an image by applying a leveling filter.
        - **Find sources:** Identify and mask source regions in an image using a threshold-based method.
        - **Select top sources:** Identify and select the top sources in an image based on their integrated flux and size constraints.


        Args:
            img (numpy.ndarray): The input image (2D array).
            log_file_path (str, optional): Path to a log file where the overall background level statistics
                will be written. Defaults to an empty string, which means no log will be created.
            leveling_filter_downscale_factor (int, optional): The downscaling factor to apply when creating the
                downsampled image used for local level estimation. Defaults to 4.
            fast (bool, optional): If True, global noise estimation is used. Defaults to False.
            threshold (float): The threshold value used to identify sources. Pixels whose values exceed
                `local background + threshold * local noise` will be marked as source pixels.
            local_sigma_cell_size (int, optional): The size of the cells (in pixels) used for estimating local noise
                levels. Defaults to 36.
            kernal_size_x (int, optional): The width of the Gaussian kernel used for smoothing the image. Defaults to 3.
            kernal_size_y (int, optional): The height of the Gaussian kernel used for smoothing the image. Defaults to 3.
            sigma_x (float, optional): The standard deviation in the X direction for the Gaussian kernel. Defaults to 1.
            dst (int, optional): The depth of the output image. Defaults to 1.
            number_sources (int): The number of top sources to select, based on their flux significance.
            min_size (int): The minimum number of pixels required for a source to be considered valid.
            max_size (int): The maximum number of pixels allowed for a source to be considered valid.
            dilate_mask_iterations (int, optional): The number of iterations for dilating the mask to merge nearby
                sources. A higher value merges more pixels. Defaults to 1.
            is_trail (bool, optional): _description_. Defaults to False.
            return_partial_images (bool, optional): If True, the function saves the intermediate images (local estimated
                background and background-subtracted image). Defaults to False.
            partial_results_path (str, optional): The directory path where intermediate results will be saved
            if `return_partial_images` is True. Defaults to "./partial_results/".
            level_filter (int): The size of the star level filter, shall be 5..199 and an odd number.

        Returns:
            tuple: A tuple containing three elements:
                - masked_image (numpy.ndarray): The input masked image, with only the selected top sources retained.
                - sources_mask (numpy.ndarray): A binary mask highlighting the top selected sources, if `is_trail` is True.
                - top_contours (list): A list of contours for the top selected sources.
        """
        d = int(local_sigma_cell_size)
        if d < 3:
            raise ValueError("d must be >= 3")

        # Downscale image
        logit("Downscaling image.")
        downscale_factor = leveling_filter_downscale_factor
        if downscale_factor > 1:
            logit(f"downscale_factor: {downscale_factor}")
            downsampled_img = cv2.resize(img, (img.shape[1] // downscale_factor, img.shape[0] // downscale_factor),
                                         interpolation=cv2.INTER_AREA)
            d_small = max(3, d // downscale_factor)  # shrink kernel accordingly
        else:
            downsampled_img = img
            d_small = d

        milliseconds0 = int(time.time() * 1000)
        logit("Estimating initial local levels from downscaled image.")
        initial_local_levels = self.ring_mean_background_estimation(downsampled_img, d_small)
        logit(f"Time in milliseconds since epoch {int(time.time() * 1000) - milliseconds0}")

        # Upscale back with interpolation
        logit("Upscaling initial local levels.")
        if downscale_factor > 1:
            initial_local_levels = cv2.resize(initial_local_levels, (img.shape[1], img.shape[0]),
                                              interpolation=cv2.INTER_LINEAR)

        # write background info to log
        with open(log_file_path, "a") as file:
            file.write("\n--- Initial Background Stats ---\n")
            file.write(f"Initial background level mean : {np.mean(initial_local_levels)}\n")
            file.write(f"Initial background level stdev : {np.std(initial_local_levels)}\n")
            file.write(f"p99.95 Initial background level : {np.percentile(initial_local_levels,self.cfg.percentile_threshold)}\n")
            file.write(f"Initial background level mean : {np.min(initial_local_levels)}\n")


        logit("Leveling image.")
        # --- Build cleaned image once (residual) ---
        #######
        cleaned_img = self.subtract_background(img, initial_local_levels)

        if return_partial_images:
            logit("Writing cleaned_img (leveled image) to disk.")
            cv2.imwrite(os.path.join(partial_results_path, "initial_cleaned_image.png"), cleaned_img)

        # Downscale image
        downscale_factor = leveling_filter_downscale_factor
        if downscale_factor > 1:
            downsampled_img = cv2.resize(cleaned_img, (cleaned_img.shape[1] // downscale_factor,
                                                       cleaned_img.shape[0] // downscale_factor),
                                         interpolation=cv2.INTER_AREA)
            d_small = max(3, d // downscale_factor)  # shrink kernel accordingly
        else:
            downsampled_img = cleaned_img
            d_small = d

        logit("Estimating final local levels for stats calculation.")
        final_local_levels = self.ring_mean_background_estimation(downsampled_img, d_small)

        # Upscale back with cubic spline
        if downscale_factor > 1:
            final_local_levels = cv2.resize(final_local_levels, (img.shape[1], img.shape[0]),
                                            interpolation=cv2.INTER_LINEAR)

        # write background info to log
        with open(log_file_path, "a") as file:
            file.write("\n--- Final Background Stats ---\n")
            file.write(f"Final background level mean : {np.mean(final_local_levels)}\n")
            file.write(f"Final pass background level stdev : {np.std(final_local_levels)}\n")
            file.write(f"p99.95 Final background level : {np.percentile(final_local_levels,self.cfg.percentile_threshold)}\n")
            file.write(f"Final background level mean : {np.min(final_local_levels)}\n")

        logit("Estimating noise from leveled image.")

        milliseconds0 = int(time.time() * 1000)
        logit("Fast mode noise using estimate_noise_pairs_function")
        sigma_g = self.estimate_noise_pairs(cleaned_img, sep=int(noise_pair_sep_full))
        estimated_noise = np.full_like(cleaned_img, sigma_g, dtype=np.float32)
        with open(log_file_path, "a") as file:
            file.write("\n--- Noise Estimation ---\n")
            file.write(f"sigma_estimate : {float(sigma_g):.6f}\n")

        logit(f"Time in milliseconds since epoch {int(time.time() * 1000) - milliseconds0}")

        logit("resizing noise image.")
        estimated_noise = cv2.resize(estimated_noise, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_LINEAR)

        # Build leveled residual for hysteresis thresholding
        residual_img = self.subtract_background(cleaned_img,final_local_levels)

        if return_partial_images:
            logit("Writing leveled_image.")
            cv2.imwrite(os.path.join(partial_results_path, "residual_img.png"), residual_img)

        sources_mask, sources_mask_u8 = self.threshold_with_noise(
            residual_img,
            sigma_map=estimated_noise,
            k_high=float(self.cfg.hyst_k_high),
            k_low=float(self.cfg.hyst_k_low),
            use_hysteresis=True,
            min_area=int(self.cfg.hyst_min_area),
            close_kernel=int(self.cfg.hyst_close_kernel),
            sigma_gauss=float(self.cfg.hyst_sigma_gauss),
            sigma_floor=float(self.cfg.hyst_sigma_floor),
        )
        
        before_roi_clamp = int(np.count_nonzero(sources_mask_u8))
        
        # --- Clamp hysteresis sources mask to centered ROI (for centroiding) ---
        roi_keep_frac_x = float(self.cfg.roi_keep_frac_x)  # e.g. 0.90
        roi_keep_frac_y = float(self.cfg.roi_keep_frac_y)  # e.g. 0.90


        h, w = sources_mask_u8.shape[:2]
        roi_w = max(1, int(round(w * roi_keep_frac_x)))
        roi_h = max(1, int(round(h * roi_keep_frac_y)))
        x0 = (w - roi_w) // 2
        y0 = (h - roi_h) // 2
        x1 = x0 + roi_w
        y1 = y0 + roi_h
        
        logit(f"ROI clamp box: x[{x0}:{x1}] y[{y0}:{y1}] (keep {roi_keep_frac_x*100:.1f}% x {roi_keep_frac_y*100:.1f}%)")

        # Zero out detections outside ROI (in-place)
        sources_mask_u8[:y0, :] = 0
        sources_mask_u8[y1:, :] = 0
        sources_mask_u8[:, :x0] = 0
        sources_mask_u8[:, x1:] = 0
        
        after_roi_clamp = int(np.count_nonzero(sources_mask_u8))
        logit(f"hyst mask pixels before/after ROI clamp: {before_roi_clamp} -> {after_roi_clamp}  "
        f"({(100.0*after_roi_clamp/max(before_roi_clamp,1)):.1f}% kept)")

        # Keep the boolean mask consistent with the u8 mask
        sources_mask = (sources_mask_u8 > 0)

        if return_partial_images:
            logit("writing hysteresis sources mask to disk.")
            cv2.imwrite(os.path.join(partial_results_path, "hyst_sources_mask.png"), sources_mask_u8)

        logit("Creating cleaned masked_image. Using hysteresis.")
        # Apply mask to the cleaned image for downstream display/photometry
        masked_clean_image = cv2.bitwise_and(cleaned_img, cleaned_img, mask=sources_mask_u8)

        #create simple mask for autogain autoexposure thresholding
        simple_sources_mask = residual_img > (float(self.cfg.hyst_k_low)*estimated_noise)
        simple_sources_mask_u8 = (simple_sources_mask.astype(np.uint8) * 255)
        
        # Apply simple mask to original image (for autogain/autoexposure masked percentile stats)
        masked_original_image = cv2.bitwise_and(img, img, mask=simple_sources_mask_u8)

        # Saturation threshold
        pixel_saturated_value = self.cfg.pixel_saturated_value_raw16
        
        # Circle diameter = 85% of width, and ignore 5% top + 5% bottom.
        roi_circle_diam_frac_w = float(self.cfg.roi_circle_diam_frac_w)  # e.g. 0.85
        roi_strip_frac_y = float(self.cfg.roi_strip_frac_y)              # e.g. 0.05

        h, w = img.shape[:2]
        roi_stats_mask_u8 = np.zeros((h, w), dtype=np.uint8)

        # 1) Big circle (allowed to clip at top/bottom)
        cx, cy = w // 2, h // 2
        r = int(round(0.5 * roi_circle_diam_frac_w * w))
        cv2.circle(roi_stats_mask_u8, (cx, cy), r, 255, thickness=-1)

        # 2) Remove top/bottom strips
        y0 = int(round(roi_strip_frac_y * h))
        y1 = int(round((1.0 - roi_strip_frac_y) * h))
        roi_stats_mask_u8[:y0, :] = 0
        roi_stats_mask_u8[y1:, :] = 0

        # Count simple-threshold mask pixels *inside* ROI (optional but consistent)
        n_mask_pixels = int(np.count_nonzero((simple_sources_mask_u8 > 0) & (roi_stats_mask_u8 > 0)))

        roi_img = img[roi_stats_mask_u8 > 0]
        roi_masked = masked_original_image[roi_stats_mask_u8 > 0]


        # Unmasked: p99.9 of valid (unsaturated) pixels in ROI
        valid_original = roi_img[roi_img < pixel_saturated_value]
        if valid_original.size > 0:
            p999_original = np.percentile(valid_original, self.cfg.percentile_threshold)  # percentile_threshold = 99.95
        else:
            p999_original = np.percentile(roi_img, self.cfg.percentile_threshold)

        # TODO: Create histogram!!!
        # Run: _clean_image_histogram in a thread (don't wait)
        def _clean_image_histogram():
            t0 = time.monotonic()
            self.server.utils.meta['p999_value'] = p999_original
            self.server.utils.meta['dtc'] = self.server.curr_img_dtc
            basename = self.server.curr_image_name
            target_path = self.cfg.inspection_path
            histogram_filename = self.server.utils.create_image_histogram(
                cleaned_img, basename, target_path,
                postfix="_cleaned_image_histogram",
                title="Clean Image Histogram"
            )
            self.server.utils.create_symlink(self.cfg.web_path, histogram_filename, 'last_inspection_histogram_cleaned_image.jpg')
            self.server.utils.meta['p999_value'] = -1
            self.log.debug(f'Histogram {histogram_filename} completed {get_dt(t0)}.')

        # Launch the thread and detach it (don't wait)
        threading.Thread(target=_clean_image_histogram, daemon=True).start()

        # Masked: p99.9 of valid (unsaturated) pixels in ROI
        valid_masked = roi_masked[roi_masked < pixel_saturated_value]
        if valid_masked.size > 0:
            p999_masked_original = np.percentile(valid_masked, 99.95)
        else:
            p999_masked_original = np.percentile(roi_masked, 99.95)
            
        # --- Optional: saturation/headroom stats (ROI) ---
        n_sat_roi = int(np.count_nonzero(roi_img >= pixel_saturated_value))
        sat_frac_roi = n_sat_roi / float(roi_img.size) if roi_img.size else 0.0

        valid_roi = roi_img[roi_img < pixel_saturated_value]
        max_valid_roi = int(valid_roi.max()) if valid_roi.size else int(roi_img.max())

        logit(f"sat_frac_roi: {sat_frac_roi:.6g}  n_sat_roi: {n_sat_roi}")
        logit(f"max_valid_roi: {max_valid_roi}  headroom: {int(pixel_saturated_value - max_valid_roi)} counts")
        logit(f"p999_original (ROI excl. saturated): {p999_original}")
        logit(f"p999_masked_original (ROI excl. saturated): {p999_masked_original}")
        logit(f"n_mask_pixels (ROI): {n_mask_pixels}")

        # Defer trail/still decision to astrometry.py classifier; only honor explicit override
        use_trail_mode = bool(is_trail) if is_trail is not None else False
        is_trail = use_trail_mode

        contours, _ = cv2.findContours(sources_mask_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if return_partial_images:
            bg_vis_16 = initial_local_levels.astype(np.uint16)
            cv2.imwrite(os.path.join(partial_results_path, "1.3 - Local Estimated Background.png"),
                            bg_vis_16)
            cv2.imwrite(os.path.join(partial_results_path, "1.4 - Local Background subtracted image.png"), cleaned_img)
            cv2.imwrite(os.path.join(partial_results_path, "1.5 - Masked image STILL.png"), masked_clean_image)
            cv2.imwrite(os.path.join(partial_results_path, "1.7 - Merged Mask.png"), sources_mask_u8)

        # Calculate fluxes
        logit("Filtering sources.")
        fluxes = {}
        for label, contour in enumerate(contours):
            if cv2.contourArea(contour) < min_size or cv2.contourArea(contour) > max_size:
                fluxes[label] = -1
                continue
            # Calculate enclosing Rectangle
            x, y, w, h = cv2.boundingRect(contour)
            x1, y1, x2, y2 = (x, y, x + w, y + h)
            roi = masked_clean_image[y1:y2, x1:x2]
            shifted_contour = contour - [x, y]
            # Extract a masked ROI from the cleaned image containing each segement
            mask = np.zeros((h, w), dtype=np.uint8)
            cv2.drawContours(mask, [shifted_contour], -1, (255), thickness=cv2.FILLED)
            filtered_roi = cv2.bitwise_and(roi, roi, mask=mask)
            fluxes[label] = int(np.sum(filtered_roi.astype(np.int64)))
        logit("sorting fluxes.")
        # Sort sources based on flux
        fluxes_sorted = list(sorted(fluxes.items(), key=lambda item: item[1], reverse=True))

        # define number of sources to return
        if len(fluxes_sorted) < number_sources:
            number_sources = len(fluxes_sorted)

        logit("Taking only top contours. Chopping rest off.")
        # keep only top sources
        top_sources = []
        for i in range(len(fluxes_sorted) if is_trail else number_sources):
            if fluxes_sorted[i][1] != -1:
                top_sources.append(fluxes_sorted[i][0])
        top_contours = [contours[i] for i in top_sources]

        # --- For TRAIL frames: compute "brightness" as MEAN intensity inside each merged streak ---
        if is_trail:
            brightness_means = []
            for idx, cnt in enumerate(top_contours):
                x, y, w, h = cv2.boundingRect(cnt)
                cnt_roi = cnt.copy()
                cnt_roi[:, 0, 0] -= x
                cnt_roi[:, 0, 1] -= y

                roi = masked_clean_image[y:y + h, x:x + w]
                mask = np.zeros((h, w), np.uint8)
                # Use merged contour footprint (exact union shape)
                cv2.drawContours(mask, [cnt_roi], -1, 255, thickness=cv2.FILLED)

                vals = roi[mask > 0]
                mean_intensity = float(vals.mean()) if vals.size else 0.0
                brightness_means.append((idx, mean_intensity))

            # Sort merged streaks by MEAN intensity (descending)
            brightness_sorted = sorted(brightness_means, key=lambda t: t[1], reverse=True)
            # In TRAIL mode: keep only the top-N merged streaks by MEAN intensity
            keep_n = min(number_sources, len(brightness_sorted)) if is_trail else len(brightness_sorted)
            order = [i for (i, _) in brightness_sorted[:keep_n]]
            # Reorder (and slice, in trail mode) so downstream overlay/exports follow this order
            top_contours = [top_contours[i] for i in order]

        if return_partial_images:
            sources_mask_tmp = np.zeros_like(masked_clean_image, dtype=np.uint8)
            cv2.drawContours(sources_mask_tmp, top_contours, -1, (0, 255, 255), thickness=cv2.FILLED)
            # mask sources based on flux
            cv2.imwrite(os.path.join(partial_results_path, "2 - Contoured Source Mask.png"), sources_mask_tmp)
        logit(f"is_trail: {is_trail}")

        if is_trail:
            # Log detection summary (trailed)
            with open(log_file_path, "a") as file:
                file.write("\n--- Detection Summary (trailed) ---\n")
                file.write(f"num_contours : {len(contours)}\n")
                file.write(f"num_top_contours : {len(top_contours)}\n")
                file.write(f"is_trail : {is_trail}\n")
                if contours:
                    lengths = [cv2.arcLength(c, True) for c in contours]
                    file.write(f"median_length_px : {float(np.median(lengths)):.1f}\n")
                    file.write(f"median_area_px : {float(np.median([cv2.contourArea(c) for c in contours])):.1f}\n")

        return masked_clean_image, sources_mask, top_contours, p999_original, p999_masked_original, n_mask_pixels



