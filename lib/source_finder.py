import os

import cv2
import matplotlib.pyplot as plt
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


# TODO: Done local_sigma_cell_size = 36
def estimate_local_sigma(img, cell_size=36):
    """Estimate local noise levels in an image by dividing it into non- overlapping
    cells and computing the noise in each cell using skimage's `estimate_sigma`
    function.

    The estimated noise values are stored in a grid that corresponds to the image dimensions
    and are resized using nearest-neighbor interpolation to match the original image size.

    Args:
        img (numpy.ndarray): The input grayscale image (2D array) for which the local noise
            levels need to be estimated.
        cell_size (int, optional): The size of the non-overlapping square cells (in pixels)
            used for noise estimation. The image is divided into cells of shape (cell_size x cell_size).
            Defaults to 36.

    Returns:
        numpy.ndarray: A 2D array of the same shape as the input image, where each element
        contains the estimated noise value for the corresponding pixel location in the image.
    """
    # Get the dimensions of the image
    height, width = img.shape

    # Initialize an array to store the local noise estimates
    local_sigma = np.zeros((height // cell_size, width // cell_size), dtype=np.float32)

    # Loop over the image in non-overlapping (cell_size x cell_size) cells
    for i in range(0, height, cell_size):
        for j in range(0, width, cell_size):
            # Ensure the cell fits within the image bounds
            if i + cell_size <= height and j + cell_size <= width:
                # Extract the (cell_size x cell_size) cell
                cell = img[i: i + cell_size, j: j + cell_size]
                # Estimate the noise in the cell
                local_noise = estimate_sigma(cell)
                # Store the estimated noise value
                local_sigma[i // cell_size, j // cell_size] = local_noise

    # Resize using nearest-neighbor interpolation
    local_sigma = cv2.resize(local_sigma, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_LINEAR)

    # plot the estimated background noise
    # plt.imshow(local_sigma, cmap='gray')
    # plt.show(block=True)

    return local_sigma


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
        cv2.imwrite(
            os.path.join(partial_results_path, "1.1 - Global Background subtracted image.png"),
            cleaned_img,
        )

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
        # TODO: Create the local_levels
        # Code goes here:
        local_levels = None


    # Resize using nearest-neighbor interpolation
    local_levels_resized = cv2.resize(local_levels, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_LINEAR)

    cleaned_img = img - local_levels_resized

    # write background info to log
    with open(log_file_path, "a") as file:
        file.write(f"overall background level mean : {np.mean(local_levels)}\n")
        file.write(f"overall background level stdev : {np.std(local_levels)}\n")

    # Save Background subtracted image
    if return_partial_images:
        cv2.imwrite(
            os.path.join(partial_results_path, "1.3 - Local Estimated Background.png"),
            local_levels_resized,
        )
        cv2.imwrite(
            os.path.join(partial_results_path, "1.4 - Local Background subtracted image.png"),
            cleaned_img,
        )

    return cleaned_img, local_levels


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
        cv2.imwrite(
            os.path.join(partial_results_path, "1.1 - Estimated Background.png"),
            bkg.background,
        )
        cv2.imwrite(
            os.path.join(partial_results_path, "1.2 - Background subtracted image.png"),
            cleaned_img,
        )

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
        cv2.imwrite(
            os.path.join(partial_results_path, "1.1 - Estimated Background.png"),
            bkg.background,
        )
        cv2.imwrite(
            os.path.join(partial_results_path, "1.2 - Background subtracted image.png"),
            cleaned_img,
        )

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
    local_levels = background_img
    sources_mask = img_smoothed > local_levels + threshold * estimated_noise

    # Subtract background from image and mask sources
    cleaned_img = downsampled_img - local_levels
    masked_image = np.clip(cleaned_img, 0, np.inf) * sources_mask
    # Resize using nearest-neighbor interpolation
    masked_image = cv2.resize(masked_image, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_LINEAR)

    if return_partial_images:
        cv2.imwrite(os.path.join(partial_results_path, "1.5 - Masked image.png"), masked_image)

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

    # check if sources were detected
    if segment_map:
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
    contours, _ = cv2.findContours(dilated_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

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
        cv2.imwrite(os.path.join(partial_results_path, "2 - Source Mask.png"), sources_mask)

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
        cv2.imwrite(os.path.join(partial_results_path, "2 - Source Mask.png"), sources_mask)

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

def _ring_mean_background_estimation(
    downsampled_img,
    d_small,
):
    """Estimate and subtract local background levels in an image by applying a leveling
    filter.

    This function applies a ring-based local leveling filter of size `d` to estimate
    the local background level for each pixel in the image. The background is
    downscaled for speed and resized back to the original shape.

    Args:
        img (numpy.ndarray): The input image (2D array).
        d (int): Filter size (must be >= 3).
        leveling_filter_downscale_factor (int, optional): Downscaling factor to apply
            when creating the downsampled image used for local level estimation.
            Defaults to 4.

    Returns:
        numpy.ndarray: The estimated local background for each pixel in the image.
    """
    # (d × d px) local leveling filter
    mask = ring_mask(d_small)
    ring_count = float(mask.sum())
    level_filter = (1 / ring_count) * mask

    # Calculate the local level of the downsampled image
    local_levels = cv2.filter2D(downsampled_img, -1, level_filter, borderType=cv2.BORDER_REFLECT)

    return local_levels

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

import numpy as np

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
        is_trail=False,
        return_partial_images=False,
        partial_results_path="./partial_results/",
        level_filter: int = 9,
        ring_filter_type: str = 'mean'
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
    ########
    # Background Estimation
    ########
    print(f'  threshold: {threshold} level_filter: {level_filter} ring_filter_type: {ring_filter_type}')

    d = level_filter
    if d < 3:
        raise ValueError("d must be >= 3")

    # Downscale image
    downscale_factor = leveling_filter_downscale_factor
    if downscale_factor > 1:
        downsampled_img = cv2.resize(img, (img.shape[1]//downscale_factor, img.shape[0]//downscale_factor), interpolation=cv2.INTER_AREA)
        d_small = max(3, d // downscale_factor)  # shrink kernel accordingly
    else:
        downsampled_img = img
        d_small = d

    # Run local leveling
    if ring_filter_type=="median":
        print("Using Median Ring Background Estimation")
        local_levels = _ring_median_background_estimation(downsampled_img, d_small)
    elif ring_filter_type=="mean":
        print("Using Mean Ring Background Estimation")
        local_levels = _ring_mean_background_estimation(downsampled_img, d_small)

    # Upscale back with cubic spline
    if downscale_factor > 1:
        local_levels = cv2.resize(local_levels, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_CUBIC)

    # write background info to log
    with open(log_file_path, "a") as file:
        file.write(f"overall background level mean : {np.mean(local_levels)}\n")
        file.write(f"overall background level stdev : {np.std(local_levels)}\n")

    # Save Background subtracted image
    if return_partial_images:
        cleaned_img = subtract_background(img, local_levels)
        cv2.imwrite(
            os.path.join(partial_results_path, "1.3 - Local Estimated Background.png"),
            local_levels,
        )
        cv2.imwrite(
            os.path.join(partial_results_path, "1.4 - Local Background subtracted image.png"),
            cleaned_img,
        )

    ########
    # Thresholding : find_sources
    ########

    # Apply a (3 × 3 px) Gaussian kernel with std = 1 px
    # TODO: Done find_sources gaussian blue kernal=(3,3), stdev = (1,1)
    ksize = (kernal_size_x, kernal_size_y)
    img_smoothed = cv2.GaussianBlur(img, ksize, sigma_x, dst)

    # Subtract background from image
    cleaned_img = subtract_background(img_smoothed, local_levels)

    # Estimate the per pair noise
    estimated_noise = estimate_noise_pairs(img)
    print("estimate_noise_pairs(img)", estimated_noise)

    # Get local levels from cleaned image
    if ring_filter_type == 'median':
        local_levels_from_cleaned_img = _ring_median_background_estimation(cleaned_img, d)
    elif ring_filter_type == 'mean':
        local_levels_from_cleaned_img = _ring_mean_background_estimation(cleaned_img, d)
    else:
        raise ValueError(f'Invalid ring_filter_type: {ring_filter_type} expected: median|mean')

    # create sources mask using threshold
    sources_mask = cleaned_img > local_levels_from_cleaned_img + threshold * estimated_noise

    # mask sources
    masked_image = cleaned_img * sources_mask

    if return_partial_images:
        cv2.imwrite(os.path.join(partial_results_path, "1.5 - Masked image.png"), masked_image)

    ########
    # Filter sources : select_top_sources
    ########

    # Locate sources in the masked image
    sources_mask = sources_mask.astype(np.uint8) * 255

    if not is_trail:
        # Dilate the sources_mask to merge nearby sources
        # TODO: Parameterise: Parameterise
        #
        dilation_radius = 5
        kernel = np.ones((dilation_radius, dilation_radius), np.uint8)
        # TODO Done dilate_mask_iterations = 1
        sources_mask = cv2.dilate(sources_mask, kernel, iterations=dilate_mask_iterations)

    # Find contours on the mask
    contours, _ = cv2.findContours(sources_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

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
            flux_noise[label] = np.sum(filtered_roi)

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

    if not is_trail:
        if return_partial_images:
            # mask sources based on significance (integrated flux / noise)
            sources_mask = np.zeros_like(masked_image, dtype=np.uint8)
            cv2.drawContours(sources_mask, top_contours, -1, (255), thickness=cv2.FILLED)
            cv2.imwrite(os.path.join(partial_results_path, "2 - Source Mask.png"), sources_mask)

        return masked_image, None, top_contours
    else:
        # mask sources based on significance (integrated flux / noise)
        sources_mask = np.zeros_like(masked_image, dtype=np.uint8)
        cv2.drawContours(sources_mask, top_contours, -1, (255), thickness=cv2.FILLED)
        if return_partial_images:
            cv2.imwrite(os.path.join(partial_results_path, "2 - Source Mask.png"), sources_mask)

        return masked_image, sources_mask, top_contours
