"""
Created on Thu May 25 15:35:38 2023

@author: Windell Egami

This Code is designed based on "Angular Velocity Determination Directly
from Star Tracker Measurements" Paper for Prof. Crassidis from equation 10 and 12

Inputs:
    1- Data of Two Star Sensors Txt files that includes [X,Y, std dev] for every Star sensor at each time
    2- Star Sensor Configurations as focal length and principal points
Outputs:
    1- Estimation of Angular Velocities in X, and Y direction (Equation 10)
    2- Error Covariance Matrix P    (Equation 12)
"""

import csv
import logging
import time

import numpy as np
from numpy.linalg import norm

import itertools
import math
from tqdm import tqdm
from lib.common import get_dt, run_with_timeout

# TODO star_tracker_body_rates_max_distance = 100
MAX_DISTANCE = 100  # Defines the maximum allowable distance for point matching between two arrays within the match_points function


class StarTrackerBodyRates:
    def __init__(self, timeout=1):
        self.log = logging.getLogger('pueo')
        self.timeout = timeout  # Set timeout in seconds
        self.log.debug(f'StarTrackerBodyRates: timeout: {self.timeout}')

    def vector_to_cross_product_matrix(self, vector):
        """
        Transforms a 3D vector into a cross-product matrix.

        Parameters:
            vector (list or numpy array): The input 3D vector.

        Returns:
            cross_product_matrix (numpy array): The cross-product matrix.
        """
        x, y, z = vector
        cpm = np.array([[0, -z, y],
                        [z, 0, -x],
                        [-y, x, 0]])
        return cpm

    def read_file_to_matrix(self, file_path):
        """
        Open txt file and get all data in Matrix.

        Parameters:
            file_path: Path for the file to be read.

        Returns:
            numeric_values: The data matrix with time.
        """
        # Open the file in read mode
        with open(file_path, 'r') as file:
            # Create a CSV reader object
            reader = csv.reader(file)
            # Initialize an empty list for the numeric values
            numeric_values = []
            # Iterate over each row in the CSV file
            for row in reader:
                # Convert each element in the row to a numeric value
                numeric_row = [float(element) for element in row]
                # Add the numeric row to the list of numeric values
                numeric_values.append(numeric_row)
        return numeric_values

    def getTriangles(self, set_X, X_combs):
        """
        Inefficient way of obtaining the lengths of each triangle's side.
        Normalized so that the minimum length is 1.
        """
        triang = []
        for p0, p1, p2 in X_combs:
            d1 = np.sqrt((set_X[p0][0] - set_X[p1][0]) ** 2 +
                         (set_X[p0][1] - set_X[p1][1]) ** 2)
            d2 = np.sqrt((set_X[p0][0] - set_X[p2][0]) ** 2 +
                         (set_X[p0][1] - set_X[p2][1]) ** 2)
            d3 = np.sqrt((set_X[p1][0] - set_X[p2][0]) ** 2 +
                         (set_X[p1][1] - set_X[p2][1]) ** 2)
            d_min = min(d1, d2, d3)
            d_unsort = [d1 / d_min, d2 / d_min, d3 / d_min]
            triang.append(sorted(d_unsort))

        return triang

    def sumTriangles(self, A_triang, B_triang):
        """
        For each normalized triangle in A, compare with each normalized triangle
        in B. find the differences between their sides, sum their absolute values,
        and select the two triangles with the smallest sum of absolute differences.
        """
        tr_sum, tr_idx = [], []
        for i, A_tr in enumerate(A_triang):
            for j, B_tr in enumerate(B_triang):
                # Absolute value of lengths differences.
                tr_diff = abs(np.array(A_tr) - np.array(B_tr))
                # Sum the differences
                tr_sum.append(sum(tr_diff))
                tr_idx.append([i, j])

        # Index of the triangles in A and B with the smallest sum of absolute
        # length differences.
        tr_idx_min = tr_idx[tr_sum.index(min(tr_sum))]
        A_idx, B_idx = tr_idx_min[0], tr_idx_min[1]
        print(f"Smallest difference: {min(tr_sum)}")

        return A_idx, B_idx

    def match_points(self, array1, array2, max_distance):
        correspondences = []

        for i, point1 in enumerate(array1):
            min_distance = float('inf')
            best_match_index = None

            for j, point2 in enumerate(array2):
                distance = np.linalg.norm(np.array(point1) - np.array(point2))
                if distance < min_distance and distance < max_distance:
                    min_distance = distance
                    best_match_index = j

            correspondences.append((i, best_match_index))

        return correspondences

    def shift_match_points(self, star_data1, star_data2, max_distance=100):
        """Match points between two sets of star data by identifying corresponding triangles.

        This function identifies triangles formed by sets of star positions, calculates centroids,
        applies a translation to align the sets, and then matches points based on proximity.
        It uses combinations of star points to form triangles, compares them, and applies a translation
        vector derived from the centroids of the best-matching triangles.
        """
        t0 = time.monotonic()
        print(f'Match points')
        set_A = [[data[0], data[1]] for data in star_data1]
        set_B = [[data[0], data[1]] for data in star_data2]

        # All possible triangles.
        A_combs = list(itertools.combinations(range(len(set_A)), 3))
        B_combs = list(itertools.combinations(range(len(set_B)), 3))

        # Obtain normalized triangles.
        A_triang, B_triang = self.getTriangles(set_A, A_combs), self.getTriangles(set_B, B_combs)

        # Index of the A and B triangles with the smallest difference.
        A_idx, B_idx = self.sumTriangles(A_triang, B_triang)

        # Indexes of points in A and B of the best match triangles.
        A_idx_pts, B_idx_pts = A_combs[A_idx], B_combs[B_idx]
        print(f'triangle A {A_idx_pts} matches triangle B {B_idx_pts}')

        # Matched points in A and B.
        matched_A = [set_A[_] for _ in A_idx_pts]
        matched_B = [set_B[_] for _ in B_idx_pts]
        print(f"A: {matched_A}")
        print(f"B: {matched_B}")

        # Calculate centroids
        centroid_A = np.mean(matched_A, axis=0)
        centroid_B = np.mean(matched_B, axis=0)

        # Calculate translation vector
        translation_vector = centroid_A - centroid_B
        print(f"Translation Vector: {translation_vector}")
        # Apply translation to points in B
        translated_B = set_B + translation_vector

        # match points based on proximity and distance
        matched_indexes = self.match_points(set_A, translated_B, max_distance)
        print(f"matched indexes: {matched_indexes}")
        matched_indexes = [(idx_A, idx_B) for idx_A, idx_B in matched_indexes if idx_B is not None]
        print(f'shift_match_points completed in {get_dt(t0)}.')
        return matched_indexes

    # TODO dt is default to 0.1 seconds

    def _angular_velocity_estimation(
            self,
            star_data1,
            star_data2,
            plate_scale,
            dt=0.1,
            is_file=False,
            max_distance=100,
            focal_ratio: float = 22692.68,
            x_pixel_count=2072,
            y_pixel_count=1441,
    ):
        """Estimate the angular velocity from star tracker measurements.

        This function computes the angular velocity based on star sensor measurements using the
        approach described in the paper "Angular Velocity Determination Directly from Star Tracker
        Measurements" by Prof. Crassidis, utilizing equations 10 and 12.

        Args:
            star_data1 (numpy.ndarray or str): Data from the first star sensor, which includes
                the coordinates and standard deviation in the format [X, Y, std_dev]. If `is_file`
                is True, this should be the path to a file containing the data.
            star_data2 (numpy.ndarray or str): Data from the second star sensor, formatted similarly
                to `star_data1`.
            plate_scale (float): The plate scale of the star tracker, used to convert pixel displacements
                to angular displacements.
            dt (float, optional): Time interval between the two images in seconds. Defaults to 0.1.
            is_file (bool, optional): Flag indicating whether the input data is a file (True) or an
                array (False). Defaults to False.
            max_distance (float, optional): Maximum distance for matching star points between the two datasets.
                Defaults to 100.
            focal_ratio (float, optional): Focal ratio of the star tracker. Defaults to 22692.68.
            x_pixel_count (int, optional): Number of pixels in the x-direction of the sensor. Defaults to 2072.
            y_pixel_count (int, optional): Number of pixels in the y-direction of the sensor. Defaults to 1441.

        Returns:
            tuple: A tuple containing:
                - omega (numpy.ndarray): The estimated angular velocity vector in degrees/second,
                  formatted as [X, Y, Z].
                - Pk (numpy.ndarray): The covariance matrix of the angular velocity estimation.
        """
        t0 = time.monotonic()
        print('angular_velocity_estimation')
        if is_file:
            Star1 = self.read_file_to_matrix(star_data1)
            Star2 = self.read_file_to_matrix(star_data2)
        else:
            Star1 = star_data1
            Star2 = star_data2

        print(f'Data: star1: {len(Star1)} star2: {len(Star2)}')

        # match points
        # max_distance was MAX_DISTANCE
        matched_idx = self.shift_match_points(Star1, Star2, max_distance=max_distance)
        print('Creating matched lists.')
        matched_Star1 = [Star1[idx_A] for idx_A, idx_B in matched_idx]
        matched_Star2 = [Star2[idx_B] for idx_A, idx_B in matched_idx]
        print(f'Data: matched_star1: {len(matched_Star1)} matched_star2: {len(matched_Star2)}')

        # TODO f is focal_ratio, x0 x pixel count 2072, y0 y pixel count 1411
        # TODO Done focal_ratio = 22692.68, x_pixel_count=2072, y_pixel_count=1441
        f = focal_ratio  # 22692.68
        x0 = x_pixel_count  # 2072
        y0 = y_pixel_count  # 1411
        # dt = dt
        Pk1 = np.array([[0, 0, 0], [0, 0, 0], [0, 0, 0]])
        sum1 = np.array([[0, 0, 0], [0, 0, 0], [0, 0, 0]])
        sum2 = np.array([0, 0, 0])

        for i in range(len(matched_Star1) - 1):
            # Effective measurement error variance in equation 10 for current star sensor
            sigma_i1 = matched_Star1[i][3]
            # Current Measurements for star sensor [EQ4, paper 1.]
            x1 = matched_Star1[i][1]
            y1 = matched_Star1[i][0]
            x2 = matched_Star2[i][1]
            y2 = matched_Star2[i][0]
            b1i = np.array(
                [-(x1 - x0), -(y1 - y0), f] / norm([(x1 - x0), (y1 - y0), f])
            )  # Measurement Vector bi of Star Sensor 1
            b2i = np.array(
                [-(x2 - x0), -(y2 - y0), f] / norm([(x2 - x0), (y2 - y0), f])
            )  # Measurement Vector bi of Star Sensor 1
            # Transformation Vectors to cross-product matrix for measurements [bi(k) X]
            b1_cross = self.vector_to_cross_product_matrix(b1i)

            # TODO: b2_cross not used
            # b2_cross = self.vector_to_cross_product_matrix(b2i)
            # covariance estimate based on centroiding
            variance1 = (2 * sigma_i1 ** 2) / (dt ** 2)
            # Summations required for Estimated Angular Velocity (Equation 10) & Error Covariance (Equation 12)
            b1_cross_tr = np.transpose(b1_cross)
            sum1 = sum1 + (1 / variance1) * np.matmul(b1_cross_tr, b1_cross)  # first summation in omega calc
            sum2 = sum2 + (1 / variance1) * np.matmul(b1_cross_tr, b2i)  # second summation in image calc
        # covariance matrix.
        try:
            Pk = np.linalg.inv(sum1)
        except Exception as e:
            self.log.error(f'Error: {e}')
            print(f'Error: {e}')
            return [0.0, 0.0, 0.0], Pk1
        # body angular velocity vector.
        omega = (1 / dt) * np.matmul(Pk, sum2)

        omega = omega * 180.0 / math.pi

        print(f'Angular Velocity Vector[X,Y,Z]: {omega} deg/sec')
        print(f'Covariance Matrix: {Pk}')
        print(f'angular_velocity_estimation completed in {get_dt(t0)}.')
        # Note: False indicate the procedure completed in full, no timeout.
        return omega, Pk, False

    def angular_velocity_estimation(
            self,
            star_data1,
            star_data2,
            plate_scale,
            dt=0.1,
            is_file=False,
            max_distance=100,
            focal_ratio: float = 22692.68,
            x_pixel_count=2072,
            y_pixel_count=1441,
    ):

        no_result = ([0., 0., 0.], [], True)
        try:
            result = run_with_timeout(
                self._angular_velocity_estimation, self.timeout,
                star_data1,
                star_data2,
                plate_scale,
                dt,
                is_file,
                max_distance,
                focal_ratio,
                x_pixel_count,
                y_pixel_count
            )
            if result is None:
                print('Timeout calculating angular velocity.')
                result = no_result
        except Exception as e:
            self.log.error(f"An error occurred: {e}")
            result = no_result
        return result

if __name__ == "__main__":
    st = StarTrackerBodyRates()
    st.angular_velocity_estimation(
        star_data1="debug/coordinates_image_1.txt",
        star_data2="debug/coordinates_image_2.txt",
        dt=0.1,
        is_file=True,
        max_distance=101,
    )
    # angular_velocity_estimation(star_data1=[[1, 2], [6, 3], [3, 3], [1, 8]], star_data2=[[1, 3], [6, 4], [3, 4], [1, 9]], dt=0.1, is_file=False, max_distance=100)
