import os
from contextlib import suppress

import cv2
import matplotlib.pyplot as plt
import numpy as np

import scipy
from astropy.convolution import convolve
from lmfit import Model
from photutils.segmentation import SourceCatalog, make_2dgaussian_kernel
from scipy.special import erf
from lib.utils import box_plot_compare


def remove_outliers(arr, threshold=1.5):
    """Detect and remove outliers from an array using the Interquartile Range (IQR)
    method.

    This function calculates the first and third quartiles (Q1 and Q3) of the input array, computes the
    interquartile range (IQR), and defines outliers as values that fall below `Q1 - threshold * IQR` or
    above `Q3 + threshold * IQR`. Outliers are removed from the array.

    Args:
        arr (numpy.ndarray): The input array of numerical values from which outliers will be removed.
        threshold (float, optional): The IQR multiplier used to define the outlier range.
            Defaults to 1.5 (commonly used threshold).

    Returns:
        tuple: A tuple containing:
            - filtered_arr (numpy.ndarray): The array with outliers removed.
            - lower_bound (float): The lower bound below which values are considered outliers.
            - upper_bound (float): The upper bound above which values are considered outliers.
    """
    # Calculate the first quartile (Q1) and third quartile (Q3)
    q1 = np.percentile(arr, 25)
    q3 = np.percentile(arr, 75)

    # Calculate the interquartile range (IQR)
    iqr = q3 - q1

    # Define the lower and upper bounds for outliers
    lower_bound = q1 - threshold * iqr
    upper_bound = q3 + threshold * iqr

    # Find the indices of elements outside the lower and upper bounds
    outlier_indices = np.where((arr < lower_bound) | (arr > upper_bound))

    # Remove the outliers from the array
    filtered_arr = np.delete(arr, outlier_indices)

    return filtered_arr, lower_bound, upper_bound


def filter_spikes(sources: list, radius: int) -> list:
    """Filter out spike artifacts from a list of detected sources based on their
    proximity and flux.

    This function compares the detected sources and removes those that are considered spikes. A source
    is considered a spike if it is within a specified Euclidean distance (radius) of another source
    with a higher flux. The function retains sources that meet the distance and flux criteria.

    Args:
        sources (list): A list of dictionaries where each dictionary contains details of a detected source,
            including its centroid coordinates (e.g., `{"centroid": (x, y), "flux": value}`).
        radius (int): The minimum Euclidean distance required between two sources. If two sources are
            closer than this distance and one has a higher flux, the lower flux source is considered a spike
            and will be filtered out.

    Returns:
        list: A list of filtered sources that excludes spikes.
    """
    filtered_sources = []

    radius_squared = radius ** 2  # Milan: Avoiding the Square Root
    for source_a in sources:
        should_append = True
        for source_b in sources:
            if source_a is not source_b:  # Avoid self-comparison
                dx = source_b["centroid"][0] - source_a["centroid"][0]
                dy = source_b["centroid"][1] - source_a["centroid"][1]
                distance = dx * dx + dy * dy # Milan: Avoiding the Square Root
                if distance <= radius_squared and source_b["flux"] > source_a["flux"]:
                    should_append = False
                    break
        if should_append:
            filtered_sources.append(source_a)

    return filtered_sources


def compute_centroids_from_still(
    cleaned_img,
    sources_contours,
    img,
    min_potential_source_distance=100,
    log_file_path="",
    return_partial_images=False,
    partial_results_path="./partial_results/",
):
    """Compute star centroids using intensity-weighted centroiding.

    This technique calculates the centroid of a source by using a weighted average of the pixel positions,
    where the pixel intensities act as weights within the region of interest.

    Args:
        cleaned_img (np.ndarray): The image with background removed and sources masked.
        sources_contours (list): List of contours of detected sources.
        img (np.ndarray): Original image for drawing and displaying source markers.
        min_potential_source_distance : The minimum Euclidean distance required between two sources to be considered as a potential source
            rather than a spike.
        log_file_path (str, optional): Path to log file where source statistics will be recorded. Defaults to "".
        return_partial_images (bool, optional): If True, the function will save partial images showing contours
            and detected sources. Defaults to False.
        partial_results_path (str, optional): The directory path where intermediate results will be saved
            if `return_partial_images` is True. Defaults to "./partial_results/".

    Returns:
        tuple:
            - np.ndarray: Array of precomputed star centroids with the format `[y-coordinate, x-coordinate, flux, standard deviation, diameter]`.
            - np.ndarray: Image with the detected sources marked by circles.
    """
    # radius of source circles
    sources_radius = cleaned_img.shape[0] / 80

    # Calculate centroids
    detected_sources = []
    for contour in sources_contours:
        # Calculate the minimum enclosing circle
        (x, y), radius = cv2.minEnclosingCircle(contour)
        # Calculate enclosing Rectangle
        x, y, w, h = cv2.boundingRect(contour)
        x1, y1, x2, y2 = (x, y, x + w, y + h)
        roi = cleaned_img[y1:y2, x1:x2]
        shifted_contour = contour - [x, y]
        # Extract a masked ROI from the cleaned image containing each segement
        mask = np.zeros((h, w), dtype=np.uint8)
        cv2.drawContours(mask, [shifted_contour], -1, (255), thickness=cv2.FILLED)
        img_cent = cv2.bitwise_and(roi, roi, mask=mask)
        # if return_partial_images:
        #     plt.imshow(img_cent)
        #     plt.show(block=True)
        #     plt.imshow(img[y1:y2, x1:x2])
        #     plt.show(block=True)
        # Calculate integrated flux
        flux = np.sum(img_cent)
        # Calculate centroid
        (xx, yy) = np.meshgrid(np.arange(w) + 0.5, np.arange(h) + 0.5)
        xc = np.sum(img_cent * xx) / flux
        yc = np.sum(img_cent * yy) / flux
        center = (x + xc, y + yc)
        # Calculate squared distances from centroid
        coordinates = np.argwhere(img_cent != 0)
        source_pixels = [(coord[0] + x, coord[1] + y) for coord in coordinates]
        squared_distances = [(pixel[0] - center[0]) ** 2 + (pixel[1] - center[1]) ** 2 for pixel in source_pixels]
        # Calculate variance and standard deviation
        variance = np.mean(squared_distances)
        standard_deviation = np.sqrt(variance)
        detected_sources.append({"length": radius * 2, "centroid": center, "flux": flux, "std": standard_deviation})
        # Draw contours -
        color_red = (0, 0, 255)  # Red OpenCV BGR
        if return_partial_images:
            cv2.circle(img, (int(center[0]), int(center[1])), int(sources_radius*0.6), color_red, 4)
            cv2.circle(img, (int(center[0]), int(center[1])), 1, color_red, -1)

    # filter out spikes from the detected sources
    filtred_sources = filter_spikes(detected_sources, min_potential_source_distance)

    # write centroid info to log file
    diameters = []
    for filtered_source in filtred_sources:
        diameters.append(filtered_source["length"])

    with open(log_file_path, "a") as file:
        file.write(f"number of potential sources : {len(detected_sources)}\n")
        file.write(f"number of filtred sources : {len(filtred_sources)}\n")
        file.write(f"number of rejected sources : {len(detected_sources) - len(filtred_sources)}\n")
        file.write(f"mean centroid diameter : {np.mean(diameters)}\n")

    # Draw circle around valid source
    # (OpenCV uses BGR)
    color_blue = (255, 0, 0)  # Blue
    if return_partial_images:
        for filtred_source in filtred_sources:
            cv2.circle(
                img,
                (int(filtred_source["centroid"][0]), int(filtred_source["centroid"][1])),
                int(sources_radius*0.9),
                color_blue,
                4,
            )
            cv2.circle(
                img, (int(filtred_source["centroid"][0]), int(filtred_source["centroid"][1])), 1, color_blue, -1
            )

    if return_partial_images:
        cv2.imwrite(os.path.join(partial_results_path, "3.1 - Valid Contours image.png"), img)

    # order sources by flux
    sorted_flux_x_y = sorted(filtred_sources, key=lambda x: x["flux"], reverse=True)
    # generate output
    precomputed_star_centroids = []
    for i in range(len(sorted_flux_x_y)):
        # y coordiantes, x coordinates, flux, std
        precomputed_star_centroids.append(
            [
                sorted_flux_x_y[i]["centroid"][1],
                sorted_flux_x_y[i]["centroid"][0],
                sorted_flux_x_y[i]["flux"],
                sorted_flux_x_y[i]["std"],
                sorted_flux_x_y[i]["length"],
            ]
        )

    # generate a flux based source mask
    if return_partial_images:
        # Normalize flux [0,255]
        flux_values = [d["flux"] for d in sorted_flux_x_y]
        if flux_values:
            # TODO: Orig code yields Runtime Warning, remove when when fixed/tested.
            # Issue: RuntimeWarning: divide by zero encountered in scalar divide and
            #        RuntimeWarning: invalid value encountered in multiply
            # normalized_flux = (flux_values - np.min(flux_values)) * (
            #     (255 - 160) / (np.max(flux_values) - np.min(flux_values))
            # ) + 160

            min_flux = np.min(flux_values)
            max_flux = np.max(flux_values)

            # Check if the range of flux_values is not zero
            if max_flux != min_flux:
                normalized_flux = (flux_values - min_flux) * (
                        (255 - 160) / (max_flux - min_flux)
                ) + 160
            else:
                # If all values are the same, set normalized_flux to a default value (e.g., 160)
                normalized_flux = np.full_like(flux_values, 160)

            # Draw flux point source
            flux_mask = np.zeros_like(cleaned_img)
            for i in range(len(normalized_flux)):
                with suppress(ValueError):
                    cv2.circle(
                        flux_mask,
                        (int(sorted_flux_x_y[i]["centroid"][0]), int(sorted_flux_x_y[i]["centroid"][1])),
                        10,
                        int(normalized_flux[i]),
                        cv2.FILLED,
                    )

            cv2.imwrite("./partial_results/3.2 - Flux based source mask.png", flux_mask)

    return np.array(precomputed_star_centroids), img


def trail_function(xy_tuple, x0, y0, length, angle, sigma, flux, background):
    """Computes a 2D model for simulating star trails in an image, based on Gaussian and
    error function components.

    This model simulates the intensity distribution of star trails over a set of (x, y) coordinates. The
    trail is modeled as a Gaussian profile along its width, and the flux is distributed linearly along its
    length. The model also accounts for the trail's angle, flux, and background value.

    Args:
        xy_tuple (tuple of np.ndarray): Tuple containing two arrays, representing the x and y coordinates of the image grid.
        x0 (float): X-coordinate of the star trail's center.
        y0 (float): Y-coordinate of the star trail's center.
        length (float): Length of the star trail.
        angle (float): Angle (in degrees) of the star trail with respect to the horizontal axis.
        sigma (float): Standard deviation of the Gaussian that defines the trail's width.
        flux (float): Total flux (brightness) of the star trail.
        background (float): Constant background value to be added to the entire trail.

    Returns:
        np.ndarray: Flattened array representing the computed pixel intensities of the star trail model.
    """
    x, y = xy_tuple
    dx = x - x0
    dy = y - y0
    L = length
    A = angle * np.pi / 180
    s = sigma
    F = flux
    b = background

    amp = F / (2 * L)
    g_exp = np.exp(-((dx * np.sin(A) + dy * np.cos(A)) ** 2) / 2 / s**2)
    g_erf1 = erf((dx * np.cos(A) - dy * np.sin(A) + (L / 2)) / s / 2**0.5)
    g_erf2 = erf((dx * np.cos(A) - dy * np.sin(A) - (L / 2)) / s / 2**0.5)

    f_T = b + amp * g_exp * (g_erf1 - g_erf2)
    return f_T.ravel()


def compute_centroids_from_trail(
    cleaned_img,
    sources_mask: np.ndarray,
    min_area=10,
    max_area=None,
    max_sum=None,
    min_sum=None,
    img=None,
    log_file_path="",
    return_partial_images=False,
    partial_results_path="./partial_results/",
):
    """Computes the centroids of star trails from a sources mask.

    This function analyzes a binary mask of detected sources to compute the centroids, lengths,
    and other properties of streaked sources in the given cleaned image. It employs connected
    component analysis and fits a model to each detected source, estimating its parameters and
    validating them based on specified criteria.

    Args:
        cleaned_img (np.ndarray): Image with the background removed and sources masked.
        sources_mask (np.ndarray): Binary mask representing the detected sources.
        min_area (int): Minimum area of connected components to consider as valid sources. Default is 10.
        max_area (int): Maximum area of connected components to consider as valid sources. Default is None.
        max_sum (float): Maximum sum of pixel values in connected components to consider as valid sources. Default is None.
        min_sum (float): Minimum sum of pixel values in connected components to consider as valid sources. Default is None.
        img (np.ndarray): The original image for visualization purposes (if `return_partial_images` is True).
        log_file_path (str): Path to the log file where centroid information will be written.
        return_partial_images (bool): Whether to return partial images showing the detected sources. Default is False.
        partial_results_path (str, optional): The directory path where intermediate results will be saved
            if `return_partial_images` is True. Defaults to "./partial_results/".

    Returns:
        Tuple[np.ndarray, np.ndarray]:
            - A NumPy array containing the centroids of the detected point sources.
            - The original image with drawn circles around detected sources (if `return_partial_images` is True).
    """
    # get heigh and width
    (height, width) = cleaned_img.shape
    # radius of source circles
    sources_radius = cleaned_img.shape[0] / 80

    # Label each region in the binary mask
    (labels, num_labels) = scipy.ndimage.label(sources_mask, structure=np.ones((3, 3)))
    index = np.arange(1, num_labels + 1)

    def trail_fit(a, p):
        """Calculates statistics for each labeled region in the sources mask, including
        various moments and optimized trail fit parameters.

        Args:
            a (np.ndarray): Binary mask of the connected component being analyzed.
            p (int): Flattened index of the pixels in the mask.

        Returns:
            Tuple: Contains various statistics and parameters related to the analyzed region:
                - Total flux (zeroth moment)
                - Centroid y-coordinate (first moment)
                - Centroid x-coordinate (first moment)
                - Variance along x (second moment)
                - Variance along y (second moment)
                - Covariance (second moment)
                - Area in pixels
                - Trail length
                - Trail angle
                - Standard deviation of the streak
                - Total flux of the streak
                - Background value
        """
        (y, x) = np.unravel_index(p, (height, width))
        area = len(a)
        if min_area and area < min_area:
            return (np.nan,) * 12
        if max_area and area > max_area:
            return (np.nan,) * 12
        m0 = np.sum(a)
        if min_sum and m0 < min_sum:
            return (np.nan,) * 12
        if max_sum and m0 > max_sum:
            return (np.nan,) * 12

        # get the original object data
        mask = np.zeros_like(sources_mask)
        mask[y, x] = 1
        filtered_cleaned_img = cv2.bitwise_and(cleaned_img, cleaned_img, mask=mask)
        object_img = filtered_cleaned_img[y.min() : y.max(), x.min() : x.max()]
        # if return_partial_images:
        #     plt.imshow(object_img)
        #     plt.show(block=True)
        x_len = x.max() - x.min()
        y_len = y.max() - y.min()
        d_len = (x_len**2 + y_len**2) ** 0.5

        # initialize fitting parameters
        trailmodel = Model(trail_function)
        trail_initial_guess = {
            "x0": int(x_len / 2),
            "y0": int(y_len / 2),
            "length": int(d_len * 0.8),
            "angle": -np.rad2deg(np.arctan(y_len / x_len)),
            "sigma": 2,
            "flux": m0,
            "background": 0,
        }
        trailmodel.set_param_hint("x0", min=0, max=x_len)
        trailmodel.set_param_hint("y0", min=0, max=y_len)
        trailmodel.set_param_hint("length", min=30, max=d_len)
        trailmodel.set_param_hint("sigma", min=0)
        trailmodel.set_param_hint("flux", min=0, max=m0 + 255)
        x_grid = np.linspace(0, x_len, x_len)
        y_grid = np.linspace(0, y_len, y_len)
        x_grid, y_grid = np.meshgrid(x_grid, y_grid)

        # fitting
        try:
            fit_result = trailmodel.fit(object_img.ravel(), xy_tuple=(x_grid, y_grid), **trail_initial_guess)
        except RuntimeError:  # max iterations
            return (np.nan,) * 12

        popt = fit_result.best_values
        pcov = fit_result.covar
        if pcov is None:  # bad fit
            return (np.nan,) * 12

        m1_x = popt["x0"] + x.min()
        m1_y = popt["y0"] + y.min()
        m2_xx = pcov[0, 0]
        m2_yy = pcov[1, 1]
        m2_xy = pcov[0, 1]

        # Calculate std of the streak

        # Calculate starting and ending points of the line segment
        start_x = m1_x - (popt["length"] / 2) * np.cos(np.deg2rad(360 - popt["angle"]))
        start_y = m1_y - (popt["length"] / 2) * np.sin(np.deg2rad(360 - popt["angle"]))
        end_x = m1_x + (popt["length"] / 2) * np.cos(np.deg2rad(360 - popt["angle"]))
        end_y = m1_y + (popt["length"] / 2) * np.sin(np.deg2rad(360 - popt["angle"]))

        # get line equation
        m = (end_y - start_y) / (end_x - start_x)
        b = start_y - m * start_x

        # Calculate predicted y-values
        coordinates = np.argwhere(filtered_cleaned_img != 0)
        predicted_y_values = [m * x + b for y, x in coordinates]

        # Calculate residuals and std
        squared_residuals = [(y - predicted_y) ** 2 for (y, x), predicted_y in zip(coordinates, predicted_y_values)]
        average_squared_residuals = np.mean(squared_residuals)
        std_streak = np.sqrt(average_squared_residuals)

        # draw the fitted line and centroid on the streak
        # if return_partial_images:
        #     filtered_cleaned_img_bgr = cv2.convertScaleAbs(filtered_cleaned_img)
        #     filtered_cleaned_img_bgr = cv2.cvtColor(filtered_cleaned_img_bgr, cv2.COLOR_GRAY2BGR)
        #     cv2.line(filtered_cleaned_img_bgr, (int(start_x), int(start_y)), (int(end_x), int(end_y)), (255, 0, 0), 1)
        #     cv2.circle(filtered_cleaned_img_bgr, (int(m1_x), int(m1_y)), 1, (0, 255, 0), -1)
        #     object_img = filtered_cleaned_img_bgr[y.min():y.max(), x.min():x.max()]
        #     plt.imshow(object_img)
        #     plt.show(block=True)

        return (
            m0,
            m1_y,
            m1_x,
            m2_xx,
            m2_yy,
            m2_xy,
            area,
            popt["length"],
            popt["angle"],
            std_streak,
            popt["flux"],
            popt["background"],
        )

    # Fitting curve to streaks
    tmp = scipy.ndimage.labeled_comprehension(cleaned_img, labels, index, trail_fit, "12f", None, pass_positions=True)
    valid = np.all(~np.isnan(tmp), axis=1)
    extracted = tmp[valid, :]

    # create centroids dictionary
    detected_sources = []
    for source in extracted:
        detected_sources.append(
            {"length": source[7], "centroid": (source[2], source[1]), "flux": source[0], "std": source[9]}
        )
        # Draw circle around detected sources red
        color_red = (0, 0, 255)  # Red (BGR)
        if return_partial_images:
            cv2.circle(img, (int(source[2]), int(source[1])), int(sources_radius*0.6), color_red, 2)
            cv2.circle(img, (int(source[2]), int(source[1])), 1, color_red, -1)

    if return_partial_images:
        cv2.imwrite(os.path.join(partial_results_path, "3 - Contours image.png"), img)

    # Filter detected sources based on Distribution of lengths (remove outliers)
    length_values = [item["length"] for item in detected_sources]
    length_filtred, lower_bound, upper_bound = remove_outliers(length_values)

    if return_partial_images:
        box_plot_compare(length_values, length_filtred, "4.1 - Sources Lengths")

    filtred_sources = []
    for detected_source in detected_sources:
        if detected_source["length"] < upper_bound and detected_source["length"] > lower_bound:
            filtred_sources.append(detected_source)
            # Draw circle around possible star - valid source (Red Circle)
            color_blue = (255, 0, 0)  # Red OpenCV uses BGR
            if return_partial_images:
                cv2.circle(
                    img,
                    (int(detected_source["centroid"][0]), int(detected_source["centroid"][1])),
                    int(sources_radius*0.8),
                    color_blue,
                    2,
                )
                cv2.circle(
                    img, (int(detected_source["centroid"][0]), int(detected_source["centroid"][1])), 1, color_blue, -1
                )

    if return_partial_images:
        cv2.imwrite(os.path.join(partial_results_path, "4.2 - Valid Contours image.png"), img)

    # write centroid info to log file
    with open(log_file_path, "a") as file:
        file.write(f"mean centroid diameter : {np.mean(length_values)}\n")
        file.write(f"number of potential sources : {len(length_values)}\n")
        file.write(f"number of filtred sources : {len(filtred_sources)}\n")
        file.write(f"number of rejected sources : {len(length_values) - len(filtred_sources)}\n")

    # order sources by flux
    sorted_flux_x_y = sorted(filtred_sources, key=lambda x: x["flux"], reverse=True)
    # generate output
    precomputed_star_centroids = []
    for i in range(len(sorted_flux_x_y)):
        # y coordiantes, x coordinates, flux, std
        precomputed_star_centroids.append(
            [
                sorted_flux_x_y[i]["centroid"][1],
                sorted_flux_x_y[i]["centroid"][0],
                sorted_flux_x_y[i]["flux"],
                sorted_flux_x_y[i]["std"],
            ]
        )

    # generate a flux based source mask
    if return_partial_images:
        # Normalize flux [0,255]
        flux_values = [d["flux"] for d in sorted_flux_x_y]
        normalized_flux = (flux_values - np.min(flux_values)) * (
            (255 - 160) / (np.max(flux_values) - np.min(flux_values))
        ) + 160

        # draw flux point source
        flux_mask = np.zeros_like(cleaned_img)
        for i in range(len(normalized_flux)):
            cv2.circle(
                flux_mask,
                (int(sorted_flux_x_y[i]["centroid"][0]), int(sorted_flux_x_y[i]["centroid"][1])),
                10,
                int(normalized_flux[i]),
                cv2.FILLED,
            )

        cv2.imwrite(os.path.join(partial_results_path, "6 - Flux based point mask.png"), flux_mask)

    return np.array(precomputed_star_centroids), img


def compute_centroids_photutils(cleaned_img, img_bgr, segment_map, number_sources):
    """Computes the centroids and flux of sources using photutils.

    This function uses the provided segmentation map to identify sources. It calculates the centroids and total flux of
    the sources and marks them on the original image.

    Args:
        cleaned_img (np.ndarray): Image with the background removed. It is the data used for source detection.
        img_bgr (np.ndarray): Original color image (BGR format) where detected sources will be marked.
        segment_map (np.ndarray): A labeled segmentation map that identifies different sources in the image.
        number_sources (int): The maximum number of sources to return and visualize.

    Returns:
        Tuple[np.ndarray, np.ndarray]:
            - A NumPy array containing the centroids and flux of the detected sources in the format of (y_centroid, x_centroid, segment_flux).
            - The original image with circles drawn around the detected sources.
    """
    # convolve the data with a 2D Gaussian kernel
    kernel = make_2dgaussian_kernel(3.0, size=5)  # FWHM = 3.0
    img_smoothed = convolve(cleaned_img, kernel)

    # calculate  sources centroids and flux
    cat = SourceCatalog(cleaned_img, segment_map, convolved_data=img_smoothed)
    tbl = cat.to_table()
    sources_info = list(zip(tbl["ycentroid"].data, tbl["xcentroid"].data, tbl["segment_flux"].data))
    sources_info = sorted(sources_info, key=lambda x: x[2], reverse=True)
    sources_info = np.array([list(t) for t in sources_info])[:number_sources]

    # Draw circle around detected sources
    color_blue = (255, 0, 0)  # Blue - OpenCV uses BGR
    sources_radius = cleaned_img.shape[0] / 80
    for source in sources_info:
        cv2.circle(img_bgr, (int(source[1]), int(source[0])), int(sources_radius*0.8), color_blue, 4)
        cv2.circle(img_bgr, (int(source[1]), int(source[0])), 1, color_blue, -1)

    return sources_info, img_bgr
