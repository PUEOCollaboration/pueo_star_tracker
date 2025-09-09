import os

import cv2
import numpy as np
from astropy.convolution import convolve
from astropy.stats import SigmaClip, sigma_clipped_stats
from photutils.background import Background2D, MedianBackground, SExtractorBackground
from photutils.segmentation import detect_sources, make_2dgaussian_kernel
from scipy.signal import convolve2d
from skimage.restoration import estimate_sigma
from skimage.transform import downscale_local_mean
from numpy.lib.stride_tricks import sliding_window_view

from lib.common import logit

def _save_u16(path: str, img: np.ndarray) -> None:
    """
    Save an image as 16-bit (PNG/TIFF). If input is float32,
    it's clipped to [0, 65535] then cast to uint16.
    """
    a = img
    if a.dtype.kind == "f":
        a = np.clip(a, 0, 65535).astype(np.uint16)
    elif a.dtype == np.uint8:
        a = (a.astype(np.uint16) * 257)  # 255->65535 scaling
    elif a.dtype != np.uint16:
        a = a.astype(np.uint16, copy=False)
    cv2.imwrite(path, a)


def robust_sigma_mad(a: np.ndarray) -> float:
    """
    Compute a robust estimate of background noise sigma using MAD.

    Parameters
    ----------
    a : np.ndarray
        2D array (float32 recommended), ideally a background-subtracted image.

    Returns
    -------
    float
        Robust sigma estimate: 1.4826 * median(|a - median(a)|).
    """
    a = a.astype(np.float32, copy=False)
    med = np.median(a)
    mad = np.median(np.abs(a - med))
    return float(1.4826 * mad)


def _core_vs_perimeter_stats(img: np.ndarray, contour: np.ndarray, pad: int = 2):
    """
    Measure mean core brightness and perimeter background for a blob.

    Parameters
    ----------
    img : np.ndarray
        2D background-subtracted array (float32).
    contour : np.ndarray
        OpenCV contour (Nx1x2 int array) for the blob at full resolution.
    pad : int
        Extra pixels to pad the bounding ROI around the contour.

    Returns
    -------
    core_mean : float
        Mean intensity inside the blob’s filled contour (ROI).
    per_mean : float
        Mean intensity on a 1-pixel ring around the ROI borders (no corners).
    per_sigma : float
        Robust perimeter sigma (MAD), or std as fallback if MAD=0.
    """
    x, y, w, h = cv2.boundingRect(contour)
    x0 = max(0, x - pad);  y0 = max(0, y - pad)
    x1 = min(img.shape[1], x + w + pad);  y1 = min(img.shape[0], y + h + pad)
    roi = img[y0:y1, x0:x1]

    core = np.zeros_like(roi, dtype=np.uint8)
    shifted = contour - [x0, y0]
    cv2.drawContours(core, [shifted], -1, 255, thickness=cv2.FILLED)

    H, W = roi.shape
    if H < 3 or W < 3:
        return 0.0, 0.0, 0.0

    perim = np.zeros_like(roi, dtype=bool)
    perim[0, 1:-1]  = True; perim[-1, 1:-1] = True
    perim[1:-1, 0]  = True; perim[1:-1, -1] = True

    core_vals = roi[core > 0]
    per_vals  = roi[perim]
    if core_vals.size == 0 or per_vals.size == 0:
        return 0.0, 0.0, 0.0

    core_mean = float(core_vals.mean())
    per_mean  = float(per_vals.mean())
    per_sigma = robust_sigma_mad(per_vals.astype(np.float32))
    if per_sigma <= 0.0:
        per_sigma = float(per_vals.std(ddof=1)) if per_vals.size > 1 else 0.0
    return core_mean, per_mean, per_sigma

# TODO: Done local_sigma_cell_size = 36
def estimate_local_sigma(img: np.ndarray, cell_size: int = 36) -> np.ndarray:
    """
    Estimate a per-pixel sigma map using non-overlapping tiles with edge handling.

    Parameters
    ----------
    img : np.ndarray
        2D array (float32 recommended), typically background-subtracted.
    cell_size : int
        Tile size in pixels for local sigma estimation.

    Returns
    -------
    np.ndarray
        2D float32 sigma map resized back to image shape using INTER_NEAREST.
    """
    H, W = img.shape
    ny = int(np.ceil(H / cell_size)); nx = int(np.ceil(W / cell_size))
    sigma_tiles = np.zeros((ny, nx), dtype=np.float32)
    for by in range(ny):
        for bx in range(nx):
            y0, y1 = by * cell_size, min((by + 1) * cell_size, H)
            x0, x1 = bx * cell_size, min((bx + 1) * cell_size, W)
            cell = img[y0:y1, x0:x1]
            sigma_tiles[by, bx] = float(estimate_sigma(cell))
    return cv2.resize(sigma_tiles, (W, H), interpolation=cv2.INTER_NEAREST)

def global_background_estimation(
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
        _save_u16(os.path.join(partial_results_path, "1.1 - Global Background subtracted image.png"),
              cleaned_img.astype(np.float32))

    return cleaned_img


def create_level_filter(n: int) -> np.ndarray:
    """
    Create a local leveling filter matrix of size n×n.

    The filter has ones in the following pattern:
    - First and last rows have ones in all positions except the first and last
    - First and last columns have ones in all positions except the first and last
    - All other positions are zeros
    The matrix is then normalized by the sum of all ones.

    Args:
        n: Size of the square matrix to create (must be odd and >= 3)

    Returns:
        A normalized n×n numpy array representing the level filter

    Raises:
        ValueError: If n is even or less than 3
    """
    # Validate input
    if n % 2 == 0:
        raise ValueError("n must be odd")
    if n < 3:
        raise ValueError("n must be at least 3")

    # Initialize matrix with zeros
    matrix = np.zeros((n, n), dtype=int)

    # Set first and last rows
    matrix[0, 1:-1] = 1  # First row, middle columns
    matrix[-1, 1:-1] = 1  # Last row, middle columns

    # Set first and last columns
    matrix[1:-1, 0] = 1  # Middle rows, first column
    matrix[1:-1, -1] = 1  # Middle rows, last column

    # Calculate sum of ones for normalization
    ones_sum = matrix.sum()

    # Normalize the matrix
    normalized_matrix = matrix / ones_sum

    return normalized_matrix


def local_levels_background_estimation(
        img,
        log_file_path="",
        leveling_filter_downscale_factor: int = 4,
        return_partial_images=False,
        partial_results_path="./partial_results/",
        level_filter: int = 9,
        level_filter_type: str = 'mean'
):
    """Estimate and subtract local background levels in an image by applying a leveling
    filter.

    This function uses a 9x9 local leveling filter to estimate the local background level for each
    pixel in the image, considering pixels that are between 13 and 24 pixels away from the target
    pixel. The background is subtracted from the original image, resulting in a cleaned image.

    Ref: STARS: A software application for the EBEX autonomous daytime star cameras

    Args:
        img (numpy.ndarray): The input image (2D array).
        log_file_path (str, optional): Path to a log file where the overall background level statistics
            will be written. Defaults to an empty string, which means no log will be created.
        leveling_filter_downscale_factor (int, optional): The downscaling factor to apply when creating the
            downsampled image used for local level estimation. Defaults to 4.
        return_partial_images (bool, optional): If True, the function saves the intermediate images (local estimated
            background and background-subtracted image). Defaults to False.
        partial_results_path (str, optional): The directory path where intermediate results will be saved
        if `return_partial_images` is True. Defaults to "./partial_results/".
        level_filter (int): The size of the star level filter, shall be 5..199 and an odd number.
    Returns:
        tuple: A tuple containing two numpy arrays:
            - cleaned_img (numpy.ndarray): The image with the local background subtracted.
            - local_levels (numpy.ndarray): The estimated local background for each pixel in the image.
    """
    # (9 × 9 px) local leveling filter
    # 28 is the SUM of the ones in the actual array
    if False:
        level_filter_array = (1 / 28.0) * np.array(
            [
                [0, 1, 1, 1, 1, 1, 1, 1, 0],
                [1, 0, 0, 0, 0, 0, 0, 0, 1],
                [1, 0, 0, 0, 0, 0, 0, 0, 1],
                [1, 0, 0, 0, 0, 0, 0, 0, 1],
                [1, 0, 0, 0, 0, 0, 0, 0, 1],
                [1, 0, 0, 0, 0, 0, 0, 0, 1],
                [1, 0, 0, 0, 0, 0, 0, 0, 1],
                [1, 0, 0, 0, 0, 0, 0, 0, 1],
                [0, 1, 1, 1, 1, 1, 1, 1, 0],
            ]
        )
    level_filter_type = str(level_filter_type).lower()
    if level_filter_type == 'mean':
        level_filter_array = create_level_filter(level_filter)
        # Downsample the image using local mean
        # TODO: Done leveling_filter_downscale_factor = 4
        downscale_factor = leveling_filter_downscale_factor
        downsampled_img = downscale_local_mean(img, (downscale_factor, downscale_factor))

        # Calculate the local level of the downsampled image
        local_levels = convolve2d(downsampled_img, level_filter_array, boundary="symm", mode="same")

    elif level_filter_type == 'median':
        # Median version not implemented yet; avoid None -> resize crash.
        raise NotImplementedError("level_filter_type='median' not implemented yet")


    # Resize using nearest-neighbor interpolation
    local_levels_resized = cv2.resize(local_levels, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_LINEAR)

    cleaned_img = img - local_levels_resized

    if log_file_path:
        with open(log_file_path, "a") as file:
            file.write(f"overall background level mean : {np.mean(local_levels_resized)}\n")
            file.write(f"overall background level stdev : {np.std(local_levels_resized)}\n")

    if return_partial_images:
        _save_u16(os.path.join(partial_results_path, "1.3 - Local Estimated Background.png"),
              local_levels_resized.astype(np.float32))
        _save_u16(os.path.join(partial_results_path, "1.4 - Local Background subtracted image.png"),
              cleaned_img.astype(np.float32))

    return cleaned_img, local_levels_resized


def median_background_estimation(
        img,
        sigma_clip_sigma=3.0,
        box_size_x=50,
        box_size_y=50,
        filter_size_x=3,
        filter_size_y=3,
        return_partial_images=False,
        partial_results_path="./partial_results/",
):
    """Subtract background using a 2D median background estimation technique. This
    method uses a sigma-clipped median to estimate and subtract the background from the
    input image.

    Args:
        img (numpy.ndarray): The input grayscale image (2D array).
        sigma_clip_sigma (float, optional): The sigma value to use for sigma-clipping when calculating the
            background statistics. It controls how aggressive the clipping is. Defaults to 3.0.
        box_size_x (int, optional): The size of the box along the X-axis used to divide the image into smaller
            regions for background estimation. Defaults to 50.
        box_size_y (int, optional): The size of the box along the Y-axis used for background estimation. Defaults to 50.
        filter_size_x (int, optional): The size of the filter along the X-axis applied to the background image to
            smooth the estimated background. Defaults to 3.
        filter_size_y (int, optional): The size of the filter along the Y-axis applied to the background image. Defaults to 3.
        return_partial_images (bool, optional): If True, saves intermediate images (estimated background and
            background-subtracted image) in the specified path. Defaults to False.
        partial_results_path (str, optional): The directory path where intermediate results will be saved if
            `return_partial_images` is True. Defaults to "./partial_results/".

    Returns:
        tuple: A tuple containing two numpy arrays:
            - cleaned_img (numpy.ndarray): The image with the estimated background subtracted.
            - bkg.background (numpy.ndarray): The 2D image of the estimated background.
    """
    # TODO: Done sigma_clip_sigma = 3.0
    sigma_clip = SigmaClip(sigma=sigma_clip_sigma)
    bkg_estimator = MedianBackground()

    # Generate 2D image of the background
    # TODO: Done These hardcoded number should be in config file
    bkg = Background2D(
        img,
        box_size=(box_size_x, box_size_y),
        filter_size=(filter_size_x, filter_size_y),
        sigma_clip=sigma_clip,
        bkg_estimator=bkg_estimator,
    )
    cleaned_img = img - bkg.background

    # Display Background subtracted image
    if return_partial_images:
        _save_u16(os.path.join(partial_results_path, "1.1 - Estimated Background.png"),
              bkg.background.astype(np.float32))
        _save_u16(os.path.join(partial_results_path, "1.2 - Background subtracted image.png"),
              cleaned_img.astype(np.float32))

    return cleaned_img, bkg.background


def sextractor_background_estimation(
        img,
        sigma_clip_sigma=3.0,
        box_size_x=50,
        box_size_y=50,
        filter_size_x=3,
        filter_size_y=3,
        return_partial_images=False,
        partial_results_path="./partial_results/",
):
    """Subtract background using a 2D SExtractor background estimation method. This
    method leverages the SExtractor algorithm to estimate and subtract the background
    from the input image.

    Args:
        img (numpy.ndarray): The input grayscale image (2D array).
        sigma_clip_sigma (float, optional): The sigma value to use for sigma-clipping when calculating the
            background statistics. It controls how aggressively outliers are removed from the background estimation.
            Defaults to 3.0.
        box_size_x (int, optional): The size of the box along the X-axis used to divide the image into smaller
            regions for background estimation. Defaults to 50.
        box_size_y (int, optional): The size of the box along the Y-axis used for background estimation. Defaults to 50.
        filter_size_x (int, optional): The size of the filter along the X-axis applied to smooth the estimated
            background. Defaults to 3.
        filter_size_y (int, optional): The size of the filter along the Y-axis applied to smooth the estimated
            background. Defaults to 3.
        return_partial_images (bool, optional): If True, saves intermediate images (estimated background and
            background-subtracted image) in the specified path. Defaults to False.
        partial_results_path (str, optional): The directory path where intermediate results will be saved if
            `return_partial_images` is True. Defaults to "./partial_results/".

    Returns:
        tuple: A tuple containing two numpy arrays:
            - cleaned_img (numpy.ndarray): The image with the SExtractor-estimated background subtracted.
            - bkg.background (numpy.ndarray): The 2D image of the estimated background.
    """
    # TODO: Done same here
    sigma_clip = SigmaClip(sigma=sigma_clip_sigma)
    bkg_estimator = SExtractorBackground()

    # Generate 2D image of the background
    # TODO: Done same here
    bkg = Background2D(
        img,
        (box_size_x, box_size_y),
        filter_size=(filter_size_x, filter_size_y),
        sigma_clip=sigma_clip,
        bkg_estimator=bkg_estimator,
    )
    cleaned_img = img - bkg.background

    # Display Background subtracted image
    if return_partial_images:
        _save_u16(os.path.join(partial_results_path, "1.1 - Estimated Background.png"),
              bkg.background.astype(np.float32))
        _save_u16(os.path.join(partial_results_path, "1.2 - Background subtracted image.png"),
              cleaned_img.astype(np.float32))

    return cleaned_img, bkg.background


def find_sources(
        img,
        background_img,
        fast=False,
        threshold: float = 3.1,
        local_sigma_cell_size=36,
        kernal_size_x=3,
        kernal_size_y=3,
        sigma_x=1,
        dst=1,
        return_partial_images=False,
        partial_results_path="./partial_results/",
):
    """Identify and mask source regions in an image using a threshold-based method.

    This function smooths the input image using a Gaussian kernel and estimates the local noise
    levels. It then identifies pixels that are part of potential sources by comparing the smoothed pixel
    intensities to **background + threshold * noise**.

    Args:
        img (numpy.ndarray): The input image (2D array) in which the sources need to be identified.
        background_img (numpy.ndarray): The estimated background image (2D array).
        fast (bool, optional): If True, global noise estimation is used. Defaults to False.
        threshold (float): The threshold value used to identify sources. Pixels whose values exceed
            `local background + threshold * local noise` will be marked as source pixels.
        local_sigma_cell_size (int, optional): The size of the cells (in pixels) used for estimating local noise
            levels. Defaults to 36.
        kernal_size_x (int, optional): The width of the Gaussian kernel used for smoothing the image. Defaults to 3.
        kernal_size_y (int, optional): The height of the Gaussian kernel used for smoothing the image. Defaults to 3.
        sigma_x (float, optional): The standard deviation in the X direction for the Gaussian kernel. Defaults to 1.
        dst (int, optional): The depth of the output image. Defaults to 1.
        return_partial_images (bool, optional): If True, the function saves the intermediate masked image. Defaults to False.
        partial_results_path (str, optional): The directory path where intermediate results will be saved
            if `return_partial_images` is True. Defaults to "./partial_results/".

    Returns:
        numpy.ndarray: The masked image where the background is subtracted and only source pixels are retained,
        based on the computed threshold.
    """
    # Downsample the image using local mean
    downscale_factor = 4
    downsampled_img = downscale_local_mean(img, (downscale_factor, downscale_factor))

    # Apply a (3 × 3 px) Gaussian kernel with std = 1 px
    # TODO: Done find_sources gaussian blue kernal=(3,3), stdev = (1,1)
    ksize = (kernal_size_x, kernal_size_y)
    img_smoothed = cv2.GaussianBlur(downsampled_img, ksize, sigma_x, dst)

    if fast:
        # estimate global noise
        estimated_noise = estimate_sigma(downsampled_img)
    else:
        # Estimate the local noise
        estimated_noise = estimate_local_sigma(downsampled_img, 9 * downscale_factor)

    # create sources mask using threshold
    # Ensure provided background matches the smoothed/downsampled scale
    if background_img.shape != img_smoothed.shape:
        background_img = cv2.resize(
            background_img, (img_smoothed.shape[1], img_smoothed.shape[0]),
            interpolation=cv2.INTER_LINEAR
        )
    local_levels = background_img

    sources_mask = img_smoothed > local_levels + threshold * estimated_noise

    # Subtract background from image and mask sources
    cleaned_img = downsampled_img - local_levels
    masked_image = np.clip(cleaned_img, 0, np.inf) * sources_mask
    # Resize using nearest-neighbor interpolation
    masked_image = cv2.resize(masked_image, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_NEAREST)

    if return_partial_images:
        _save_u16(os.path.join(partial_results_path, "1.5 - Masked image.png"),
              masked_image.astype(np.float32))

    return masked_image, estimated_noise


def find_sources_photutils(img, background_img, photutils_gaussian_kernal_fwhm=3, photutils_kernal_size=5):
    """Identify sources in an image using a thresholding scheme defined by photutils.

    This function subtracts the background from the input image, and creates a segmentation map to identify source regions.
    The resulting source mask is applied to the image.

    Args:
        img (numpy.ndarray): The input image (2D array) from which sources need to be identified.
        background_img (numpy.ndarray): The estimated background image (2D array).
        photutils_gaussian_kernal_fwhm (float, optional): The full width at half maximum (FWHM) of the Gaussian kernel
            used for smoothing. This controls the spread of the kernel. Defaults to 3.0.
        photutils_kernal_size (float, optional): The size of the 2D Gaussian kernel. This determines the extent of
            smoothing over the image. Defaults to 5.0.

    Returns:
        tuple: A tuple containing:
            - masked_image (numpy.ndarray): The masked image where only source pixels are retained.
            - segment_map (photutils.segmentation.SegmentationImage): The segmentation map that marks the source regions
              in the image.
    """
    # convolve the data with a 2D Gaussian kernel
    cleaned_img = img - background_img
    # TODO: Done photutils_gaussian_kernal_fwhm = 3, photutils_kernal_size = 5
    kernel = make_2dgaussian_kernel(fwhm=photutils_gaussian_kernal_fwhm, size=photutils_kernal_size)  # FWHM = 3
    convolved_data = convolve(cleaned_img, kernel)

    # Create segmentation map
    threshold = np.std(background_img)
    segment_map = detect_sources(convolved_data, threshold, npixels=10)

    # check if sources were detected robustly
    if (segment_map is not None) and (getattr(segment_map, "nlabels", 0) > 0):
        # create sources mask
        sources_mask = np.where(segment_map.data > 0, 1, 0)
        # Mask image
        masked_image = cleaned_img * sources_mask
        return masked_image, segment_map
    else:
        return None, None  # masked_image, segment_map


def select_top_sources(
        img,
        masked_image,
        estimated_noise,
        fast,
        number_sources: int,
        min_size,
        max_size,
        dilate_mask_iterations=1,
        return_partial_images=False,
        partial_results_path="./partial_results/",
):
    """Identify and select the top sources in an image based on their integrated flux and size constraints.

    This function processes the masked image, locates source regions by identifying contiguous clumps of pixels,
    and then calculates the significance of each source based on its integrated flux (signal-to-noise ratio).
    It filters out sources based on size constraints and selects the top `number_sources` sources.
    Optionally, it saves a mask of the selected sources.

    Args:
        img (numpy.ndarray): The original input image.
        masked_image (numpy.ndarray): The background-subtracted and masked image, where only source regions
            are retained.
        estimated_noise: A 2D array representing the globally estimated noise in the image when fast is set to False.
        fast (bool, optional): If True, global noise estimation is used. Defaults to False.
        number_sources (int): The number of top sources to select, based on their flux significance.
        min_size (int): The minimum number of pixels required for a source to be considered valid.
        max_size (int): The maximum number of pixels allowed for a source to be considered valid.
        dilate_mask_iterations (int, optional): The number of iterations for dilating the mask to merge nearby
            sources. A higher value merges more pixels. Defaults to 1.
        return_partial_images (bool, optional): If True, the function saves the mask of the selected sources. Defaults to False.
        partial_results_path (str, optional): The directory path where intermediate results will be saved
            if `return_partial_images` is True. Defaults to "./partial_results/".

    Returns:
        tuple: A tuple containing three elements:
            - masked_image (numpy.ndarray): The original masked image with background subtracted.
            - sources_mask (numpy.ndarray): A mask highlighting the top selected sources.
            - top_contours (list): A list of contours for the top selected sources.
    """
    # Locate sources in the masked image
    sources_mask = (np.where(masked_image > 0, 1, 0)).astype(np.uint8) * 255
    # Dilate the sources_mask to merge nearby sources
    dilation_radius = 5
    kernel = np.ones((dilation_radius, dilation_radius), np.uint8)
    # TODO Done dilate_mask_iterations = 1
    dilated_mask = cv2.dilate(sources_mask, kernel, iterations=dilate_mask_iterations)
    # Find contours on the dilated mask
    contours, _ = cv2.findContours(dilated_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # resize the estimated noise to the image scale
    if not fast:
        estimated_noise = cv2.resize(
            estimated_noise,
            (img.shape[1], img.shape[0]),
            interpolation=cv2.INTER_NEAREST,
        )

    # Calculate significance (integrated flux / noise) for each source
    flux_noise = {}
    for label, contour in enumerate(contours):
        if cv2.contourArea(contour) <= min_size:
            flux_noise[label] = -1
            continue
        # Calculate enclosing Rectangle
        x, y, w, h = cv2.boundingRect(contour)
        x1, y1, x2, y2 = (x, y, x + w, y + h)
        roi = masked_image[y1:y2, x1:x2]
        shifted_contour = contour - [x, y]
        # Extract a masked ROI from the cleaned image containing each segement
        mask = np.zeros((h, w), dtype=np.uint8)
        cv2.drawContours(mask, [shifted_contour], -1, (255), thickness=cv2.FILLED)
        filtered_roi = cv2.bitwise_and(roi, roi, mask=mask)
        # filter out sources based on number of pixels
        pixel_count = np.sum(filtered_roi > 0)
        if pixel_count < min_size or pixel_count > max_size:
            flux_noise[label] = -1
        else:
            if fast:
                flux_noise[label] = np.sum(filtered_roi)
            else:
                # Calculate total flux (signal)
                total_flux = np.sum(filtered_roi)
                # Calculate total noise as the sum of noise values from the noise ROI
                noise_roi = estimated_noise[y1:y2, x1:x2]
                total_noise = np.sum(noise_roi[mask > 0])
                # SNR = Signal / Noise
                if total_noise > 0:
                    flux_noise[label] = total_flux / total_noise
                else:
                    flux_noise[label] = -1  # Avoid division by zero

    # Sort sources based on significance (integrated flux / noise)
    flux_noise_sorted = list(sorted(flux_noise.items(), key=lambda item: item[1], reverse=True))

    # define number of sources to return
    if len(flux_noise_sorted) < number_sources:
        number_sources = len(flux_noise_sorted)

    # keep only top sources
    top_sources = []
    for i in range(number_sources):
        if flux_noise_sorted[i][1] != -1:
            top_sources.append(flux_noise_sorted[i][0])
    top_contours = [contours[i] for i in top_sources]

    # mask sources based on significance (integrated flux / noise)
    sources_mask = np.zeros_like(masked_image, dtype=np.uint8)
    cv2.drawContours(sources_mask, top_contours, -1, (255), thickness=cv2.FILLED)

    if return_partial_images:
        _save_u16(os.path.join(partial_results_path, "2a - Source Mask (contours).png"), sources_mask)

    return masked_image, sources_mask, top_contours


def select_top_sources_photutils(
        img,
        masked_image,
        segment_map,
        number_sources: int,
        return_partial_images=False,
        partial_results_path="./partial_results/",
):
    """Select the top sources in an image using Photutils segmentation and flux
    significance.

    This function processes a segmentation map from Photutils, calculates the flux significance (signal-to-noise ratio)
    for each segment, and selects the top `number_sources` based on their flux. It creates a binary mask for the
    selected sources and applies it to the masked image to retain only the top sources.

    Args:
        img (numpy.ndarray): The original input image.
        masked_image (numpy.ndarray): The background-subtracted and masked image where source regions are retained.
        segment_map (photutils.segmentation.SegmentationImage): A Photutils segmentation map containing source segments.
        number_sources (int): The number of top sources to select based on their flux significance (signal-to-noise ratio).
        return_partial_images (bool, optional): If True, the function saves the mask of the selected sources.
            Defaults to False.
        partial_results_path (str, optional): The directory path where intermediate results will be saved if
            `return_partial_images` is True. Defaults to "./partial_results/".

    Returns:
        tuple: A tuple containing three elements:
            - masked_image (numpy.ndarray): The input masked image, with only the selected top sources retained.
            - sources_mask (numpy.ndarray): A binary mask highlighting the top selected sources.
            - top_contours (list): A list of contours for the top selected sources.
    """
    # Calculate significance (integrated flux / noise) for each source
    flux_noise = {}
    noise = estimate_sigma(img)
    # Calculate significance
    for segment in segment_map.segments:
        label = segment.label
        segment_cutout = segment.make_cutout(masked_image, True)
        flux_noise[label] = np.sum(segment_cutout) / noise

    # Sort sources based on significance (integrated flux / noise)
    flux_noise_sorted = list(sorted(flux_noise.items(), key=lambda item: item[1], reverse=True))
    top_sources = []

    # define max number of sources to return
    if len(flux_noise_sorted) < number_sources:
        number_sources = len(flux_noise_sorted)
    for i in range(number_sources):
        top_sources.append(flux_noise_sorted[i][0])

    # Mask creation based on significance (integrated flux / noise)
    segment_map.keep_labels(labels=top_sources)
    sources_mask = (np.where(segment_map.data > 0, 1, 0)).astype(np.uint8) * 255

    # apply sources mask to cleaned image
    masked_image = masked_image * sources_mask

    # Sources segmentation
    top_contours, _ = cv2.findContours(sources_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if return_partial_images:
        _save_u16(os.path.join(partial_results_path, "2b - Source Mask (photutils).png"), sources_mask)


    return masked_image, sources_mask, top_contours

def ring_mask(d: int) -> np.ndarray:
    """
    1-px wide square ring mask of size d×d, EXCLUDING the four corner pixels.
    d >= 3. Ring pixel count = 4*(d-2).
    """
    if d < 3:
        raise ValueError("d must be >= 3")
    m = np.zeros((d, d), dtype=bool)
    m[0, 1:-1]  = True
    m[-1,1:-1]  = True
    m[1:-1, 0]  = True
    m[1:-1,-1]  = True
    return  m

def _ring_mean_background_estimation(downsampled_img: np.ndarray, d_small: int) -> np.ndarray:
    """
    Estimate local background via a 1-px square ring mean at downsampled scale.

    Parameters
    ----------
    downsampled_img : np.ndarray
        2D image at reduced resolution (float32 recommended).
    d_small : int
        Ring kernel size at downsampled scale (>=3, odd recommended).

    Returns
    -------
    np.ndarray
        2D float32 background at downsampled resolution.
    """
    if d_small % 2 == 0:
        d_small += 1
    mask = ring_mask(d_small).astype(np.float32)
    kernel = mask / float(mask.sum())
    return cv2.filter2D(downsampled_img.astype(np.float32), -1, kernel,
                        borderType=cv2.BORDER_REFLECT)

def _ring_median_background_estimation(img_small: np.ndarray, d_small: int) -> np.ndarray:
    """
    Apply a ring median filter with a 1-pixel-wide square ring of size d.
    Returns an array the same shape as `image`.

    Works on any numeric dtype. For integer types, the median is the usual
    floor/nearest convention used by NumPy.
    """
    # Ensure padding produces correct shape
    if d_small % 2 == 0:  # even
        d_small += 1
    print(f"Downscaled filter size: {d_small}")

    # Ensure padding produces correct shape
    pad = d_small // 2

    # Ring median on downscaled image
    mask = ring_mask(d_small)
    ring_count = mask.sum()

    img_pad = np.pad(img_small, ((pad, pad), (pad, pad)), mode="reflect")
    win = sliding_window_view(img_pad, (d_small, d_small))
    ring_vals = win[..., mask]

    k = ring_count // 2
    ring_part = np.partition(ring_vals.copy(), k, axis=-1)
    med_small = ring_part[..., k]

    if np.issubdtype(img_small.dtype, np.integer):
        med_small = med_small.astype(img_small.dtype, copy=False)

    return med_small

def subtract_background(image: np.ndarray, background: np.ndarray):
    # Subtract background
    flattened_img = image.astype(np.float32) - background.astype(np.float32)

    # Clip for display (avoid negatives, keep within valid dtype range)
    if np.issubdtype(image.dtype, np.integer):
        dtype_info = np.iinfo(image.dtype)
        flattened_img = np.clip(flattened_img, 0, dtype_info.max).astype(image.dtype)

    return flattened_img

def estimate_noise_pairs(
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
    img: np.ndarray,
    log_file_path: str = "",
    leveling_filter_downscale_factor: int = 4,
    fast: bool = False,
    threshold: float = 3.0,
    local_sigma_cell_size: int = 36,
    kernal_size_x: int = 3,
    kernal_size_y: int = 3,
    sigma_x: float = 1.0,
    dst: int = 1,
    number_sources: int = 40,
    min_size: int = 20,
    max_size: int = 600,
    dilate_mask_iterations: int = 1,
    is_trail: bool = False,
    return_partial_images: bool = False,
    partial_results_path: str = "./partial_results/",
    level_filter: int = 9,
    ring_filter_type: str = 'mean'
):
    """
    Detect point sources (and optionally trails) using ring/local background,
    k·sigma thresholding, and perimeter-vetted SNR ranking.

    Parameters
    ----------
    img : np.ndarray
        Input 2D image (uint16 or float); will be processed in float32.
    log_file_path : str
        File path for logging background stats (append mode).
    leveling_filter_downscale_factor : int
        Downscale factor for background estimation speed-up (>=1).
    fast : bool
        If True, use global noise estimate; else allow local sigma maps.
    threshold : float
        Sigma multiplier k for thresholding and acceptance (N·σ).
    local_sigma_cell_size : int
        Tile size for local sigma (if used).
    kernal_size_x, kernal_size_y : int
        Gaussian kernel size (typically 3x3).
    sigma_x : float
        Gaussian std dev (pixels) used for light smoothing.
    dst : int
        OpenCV Gaussian anchor param (kept for API parity).
    number_sources : int
        Maximum number of sources to return.
    min_size, max_size : int
        Area bounds (pixels) for valid components.
    dilate_mask_iterations : int
        Iterations for dilation to merge nearby islands (point mode).
    is_trail : bool
        If True, skip opening (trail-safe). Trail extras can be added later.
    return_partial_images : bool
        If True, write intermediate images to disk.
    partial_results_path : str
        Directory for partial result outputs.
    level_filter : int
        Ring kernel size (full-res reference) before downscaling.
    ring_filter_type : str
        'mean' or 'median' ring background (mean recommended for speed).

    Returns
    -------
    masked_image : np.ndarray
        Background-subtracted image masked to selected sources (float32).
    sources_mask : Optional[np.ndarray]
        Binary mask for winners if trails mode requires it; else None.
    top_contours : List[np.ndarray]
        List of OpenCV contours for the selected top sources.
    """
    # --- Background estimation at low-res ---
    d = level_filter
    if d < 3:
        raise ValueError("level_filter must be >= 3")
    s = max(1, int(leveling_filter_downscale_factor))

    if s > 1:
        small = cv2.resize(img.astype(np.float32, copy=False),
                           (img.shape[1]//s, img.shape[0]//s),
                           interpolation=cv2.INTER_AREA)
        d_small = max(3, d // s)
        if d_small % 2 == 0:
            d_small += 1
    else:
        small = img.astype(np.float32, copy=False)
        d_small = d if (d % 2 == 1) else d + 1

    if ring_filter_type == 'mean':
        local_levels_small = _ring_mean_background_estimation(small, d_small)
    elif ring_filter_type == 'median':
        local_levels_small = _ring_median_background_estimation(small, d_small)
    else:
        raise ValueError("ring_filter_type must be 'mean' or 'median'")

    # Upscale background to full-res
    local_levels = cv2.resize(local_levels_small, (img.shape[1], img.shape[0]),
                              interpolation=cv2.INTER_CUBIC)

    # --- Light smoothing and subtraction at full-res (float32) ---
    ksize = (kernal_size_x, kernal_size_y)
    img_smoothed = cv2.GaussianBlur(img.astype(np.float32, copy=False), ksize, sigma_x, dst)
    cleaned_img  = img_smoothed - local_levels.astype(np.float32)

    if log_file_path:
        with open(log_file_path, "a") as f:
            f.write(f"overall background level mean : {np.mean(local_levels):.6f}\n")
            f.write(f"overall background level stdev : {np.std(local_levels):.6f}\n")

    if return_partial_images:
        _save_u16(os.path.join(partial_results_path, "1.3 - Local Estimated Background.png"),
              local_levels.astype(np.float32))
        _save_u16(os.path.join(partial_results_path, "1.4 - Local Background subtracted image.png"),
              cleaned_img.astype(np.float32))


    ########
    # Thresholding : find_sources
    ########
    

    # Estimate the per pair noise
    cleaned_img = np.nan_to_num(cleaned_img, copy=False, nan=0.0, posinf=0.0, neginf=0.0)
    # --- Global noise via MAD, then k·sigma threshold ---
    sigma_global = robust_sigma_mad(cleaned_img)
    print("robust_sigma_mad(cleaned_img)", sigma_global)

    thr = threshold * float(sigma_global)
    sources_mask = (cleaned_img > thr).astype(np.uint8) * 255

    # --- Morphology (point-star only) ---
    if not is_trail:
        cross = np.array([[0,1,0],[1,1,1],[0,1,0]], np.uint8)
        sources_mask = cv2.morphologyEx(sources_mask, cv2.MORPH_OPEN, cross, iterations=1)
    if not is_trail and dilate_mask_iterations > 0:
        kernel = np.ones((5, 5), np.uint8)
        sources_mask = cv2.dilate(sources_mask, kernel, iterations=dilate_mask_iterations)

    # --- Label ---
    contours, _ = cv2.findContours(sources_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # --- Rank/accept by sigma-SNR with perimeter vetting ---
    flux_noise = {}
    for label, contour in enumerate(contours):
        area = cv2.contourArea(contour)
        if area < min_size or area > max_size:
            flux_noise[label] = -1
            continue

        core_mean, per_mean, per_sigma = _core_vs_perimeter_stats(cleaned_img, contour, pad=2)
        sigma_bg = max(float(sigma_global), float(per_sigma))
        if sigma_bg <= 0.0:
            flux_noise[label] = -1
        else:
            snr_sigma = (core_mean - per_mean) / sigma_bg
            flux_noise[label] = float(snr_sigma) if snr_sigma >= threshold else -1

    # --- Select top N ---
    flux_noise_sorted = sorted(flux_noise.items(), key=lambda kv: kv[1], reverse=True)
    number_sources = min(number_sources, len(flux_noise_sorted))
    top_sources = [i for i, (lbl, score) in enumerate(flux_noise_sorted[:number_sources]) if score != -1]
    top_contours = [contours[flux_noise_sorted[i][0]] for i in top_sources]

    # --- Winners mask + masked image ---
    winners_mask = np.zeros_like(sources_mask, dtype=np.uint8)
    if top_contours:
        cv2.drawContours(winners_mask, top_contours, -1, 255, thickness=cv2.FILLED)

    masked_image = np.clip(cleaned_img, 0, np.inf) * (winners_mask > 0)

    if return_partial_images:
        _save_u16(os.path.join(partial_results_path, "2 - Source Mask (winners).png"), winners_mask)
        _save_u16(os.path.join(partial_results_path, "3 - Masked image.png"), masked_image)

    return masked_image, (None if not is_trail else winners_mask), top_contours